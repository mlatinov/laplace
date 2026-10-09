//! The `.laplacelib` dialect: source files for *writing* a laplace library.
//!
//! A `.laplacelib` file is a relaxed `.laplace` file. It uses the same
//! `library { }` import block and the same `pkg::func()` call syntax --
//! literally the same parsing code, see [`parse_import_statements`] -- but
//! it drops the model-shaped blocks:
//!
//! - **allowed:** bare function definitions, an optional `functions { }`
//!   wrapper around them, and an optional `library { }` block
//! - **rejected:** `data`, `transformed data`, `parameters`,
//!   `transformed parameters`, `model`, `generated quantities` -- a library
//!   contributes functions to someone else's model; it is not a model
//!
//! The output of parsing is a *body*: the file's function definitions with
//! the `library { }` block removed and any `functions { }` wrapper unwrapped,
//! so it can be spliced straight into a consumer's `functions { }` block (or
//! into a `.stanfunctions` file) the same way a plain `.stan` package file is
//! today.
//!
//! Plain `.stan` files inside a package keep working untouched: they are
//! ordinary Stan, cannot import anything, and are passed through verbatim.

use std::ops::Range;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::parser::blocks::{find_top_level_blocks, BlockKind};
use crate::parser::identifiers::{check_identifiers, ReservedIdentifier};
use crate::parser::library_block::{parse_import_statements, ImportStatement, LibraryBlockError};
use crate::parser::macros::{find_macros, LocatedMacro, MacroDef, MacroError};
use crate::parser::origin::{apply_cuts, line_col, LineSegment};
use crate::parser::signatures::extract_signatures;
use crate::parser::template::{find_templates, LocatedTemplate, TemplateDef, TemplateError};
use crate::parser::visibility::{
    find_pub_markers, resolve_items, LibraryItem, Visibility, VisibilityError,
};

/// The file extension that marks the library dialect.
pub const LAPLACELIB_EXTENSION: &str = "laplacelib";

/// A parsed `.laplacelib` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceLibFile {
    /// Everything this file imports, in source order.
    pub imports: Vec<ImportStatement>,
    /// The file's item definitions: `library { }` removed, any
    /// `functions { }` wrapper unwrapped, and every `pub` marker stripped.
    /// Byte-for-byte the original text otherwise.
    pub body: String,
    /// Every item the file defines, with its visibility, in source order.
    /// Offsets are into `body`.
    pub items: Vec<LibraryItem>,
    /// Every `@template` the file defines. Templates are laplace's own
    /// construct, not Stan, so they are cut out of `body` entirely and
    /// carried here instead.
    pub templates: Vec<TemplateDef>,
    /// Which line of the original file each template was written on.
    /// Its own map because the definition's offsets point into a file
    /// whose text no longer survives in `body`.
    pub template_lines: std::collections::BTreeMap<String, usize>,
    /// Every `@macro` the file defines, cut out of `body` for the same
    /// reason templates are.
    pub macros: Vec<MacroDef>,
    /// Which line each macro was written on.
    pub macro_lines: std::collections::BTreeMap<String, usize>,
    /// Which lines of the original file `body`'s bytes came from, so a
    /// provenance comment or an error can name a line a reader can open.
    pub segments: Vec<LineSegment>,
}

impl LaplaceLibFile {
    /// The function names this file contributes to its package's
    /// public API.
    pub fn public_names(&self) -> Vec<String> {
        self.items
            .iter()
            .filter(|item| item.visibility.is_public())
            .map(|item| item.name.clone())
            .collect()
    }

    /// Every template the file defines, with the line it was written
    /// on, for provenance once the definition has been cut away.
    pub fn located_templates(&self, file: &str) -> Vec<LocatedTemplate> {
        self.templates
            .iter()
            .map(|def| LocatedTemplate {
                def: def.clone(),
                file: file.to_string(),
                line: self.template_lines.get(&def.name).copied().unwrap_or(1),
            })
            .collect()
    }

    /// Every macro the file defines, with the line it was written on.
    pub fn located_macros(&self, file: &str) -> Vec<LocatedMacro> {
        self.macros
            .iter()
            .map(|def| LocatedMacro {
                def: def.clone(),
                file: file.to_string(),
                line: self.macro_lines.get(&def.name).copied().unwrap_or(1),
            })
            .collect()
    }

    /// The macro names this file makes public.
    pub fn public_macros(&self) -> Vec<String> {
        self.macros
            .iter()
            .filter(|def| def.visibility.is_public())
            .map(|def| def.name.clone())
            .collect()
    }

    /// The template names this file makes public. Kept apart from
    /// `public_names` because a template is not a function: nothing
    /// calls it, a `@use` expands it.
    pub fn public_templates(&self) -> Vec<String> {
        self.templates
            .iter()
            .filter(|template| template.visibility.is_public())
            .map(|template| template.name.clone())
            .collect()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LaplaceLibError {
    #[error(
        "{path}: a `.laplacelib` file cannot contain a `{block}` block -- a library provides \
         functions to a model, it is not a model itself (move this into the `.laplace` file that \
         uses the library)"
    )]
    ForbiddenBlock { path: PathBuf, block: String },

    #[error("{path}: {source}")]
    Imports {
        path: PathBuf,
        #[source]
        source: LibraryBlockError,
    },

    #[error("{error}\n  --> {path}:{line}:{column}\n  help: {help}")]
    Visibility {
        path: PathBuf,
        line: usize,
        column: usize,
        help: &'static str,
        #[source]
        error: VisibilityError,
    },

    #[error("{error}\n  --> {path}:{line}:{column}\n  help: {help}")]
    Macro {
        path: PathBuf,
        line: usize,
        column: usize,
        help: String,
        #[source]
        error: Box<MacroError>,
    },

    #[error("{error}\n  --> {path}:{line}:{column}\n  help: {help}")]
    Template {
        path: PathBuf,
        line: usize,
        column: usize,
        help: String,
        #[source]
        error: Box<TemplateError>,
    },

    #[error("{error}\n  --> {path}:{line}:{column}\n  help: {help}")]
    ReservedIdentifier {
        path: PathBuf,
        line: usize,
        column: usize,
        help: String,
        #[source]
        error: ReservedIdentifier,
    },
}

/// Parse a `.laplacelib` file's text. `path` is used only to make errors
/// point at the offending file.
pub fn parse(path: &Path, source: &str) -> Result<LaplaceLibFile, LaplaceLibError> {
    // `__` is reserved for generated names, so reject it before anything
    // else: every later message is clearer once names are known to be
    // unambiguous.
    if let Err(error) = check_identifiers(source) {
        let (line, column) = line_col(source, error.offset);
        return Err(LaplaceLibError::ReservedIdentifier {
            path: path.to_path_buf(),
            line,
            column,
            help: error.help(),
            error,
        });
    }

    // Templates first: they are not Stan, they are cut out whole, and
    // the regions the `pub` scan looks at have to exclude their bodies.
    let mut templates = find_templates(source, &|_| false).map_err(|error| {
        let (line, column) = line_col(source, error.offset());
        LaplaceLibError::Template {
            path: path.to_path_buf(),
            line,
            column,
            help: error.help(),
            error: Box::new(error),
        }
    })?;
    let mut macros = find_macros(source, &|_| false).map_err(|error| {
        let (line, column) = line_col(source, error.offset());
        LaplaceLibError::Macro {
            path: path.to_path_buf(),
            line,
            column,
            help: error.help(),
            error: Box::new(error),
        }
    })?;
    // Every byte that belongs to a laplace-only definition.
    let definition_ranges: Vec<Range<usize>> = templates
        .iter()
        .map(|t| t.range.clone())
        .chain(macros.iter().map(|m| m.range.clone()))
        .collect();

    // A macro header ends `in model {`, which reads exactly like a
    // `model { }` block, and a template body is full of real ones.
    // Neither is a block of *this file*, so blocks found inside a
    // definition are dropped.
    let blocks: Vec<_> = find_top_level_blocks(source)
        .into_iter()
        .filter(|block| {
            !definition_ranges.iter().any(|range| {
                range.start <= block.byte_range.start && block.byte_range.start < range.end
            })
        })
        .collect();

    let mut imports = Vec::new();
    // Byte ranges to drop from the output, in ascending order: whole
    // `library { }` blocks, the `functions {` / `}` bookends of any
    // `functions { }` wrapper (its contents are kept verbatim), and every
    // `pub` marker.
    let mut cuts: Vec<Range<usize>> = Vec::new();
    // Where items may be defined, for the `pub` scan: everything outside
    // a top-level block, plus the body of a `functions { }` wrapper.
    let mut regions: Vec<Range<usize>> = Vec::new();
    let mut after_last_block = 0usize;

    for block in &blocks {
        if after_last_block < block.byte_range.start {
            regions.push(after_last_block..block.byte_range.start);
        }
        after_last_block = block.byte_range.end;

        match block.kind {
            BlockKind::Library => {
                imports.extend(
                    parse_import_statements(&source[block.body_range.clone()]).map_err(
                        |source| LaplaceLibError::Imports {
                            path: path.to_path_buf(),
                            source,
                        },
                    )?,
                );
                cuts.push(block.byte_range.clone());
            }
            BlockKind::Functions => {
                cuts.push(block.byte_range.start..block.body_range.start);
                cuts.push(block.body_range.end..block.byte_range.end);
                regions.push(block.body_range.clone());
            }
            other => {
                return Err(LaplaceLibError::ForbiddenBlock {
                    path: path.to_path_buf(),
                    block: other.keyword().to_string(),
                })
            }
        }
    }
    if after_last_block < source.len() {
        regions.push(after_last_block..source.len());
    }

    let markers = find_pub_markers(source, &regions).map_err(|error| {
        let (line, column) = line_col(source, error.offset());
        LaplaceLibError::Visibility {
            path: path.to_path_buf(),
            line,
            column,
            help: error.help(),
            error,
        }
    })?;
    cuts.extend(markers.iter().map(|m| m.cut.clone()));
    cuts.extend(definition_ranges.iter().cloned());

    // A marker sitting in front of a template belongs to the template,
    // not to any function that follows it.
    for template in &mut templates {
        if markers
            .iter()
            .any(|marker| marker.item_start == template.keyword_offset)
        {
            template.visibility = Visibility::Public;
        }
    }
    for definition in &mut macros {
        if markers
            .iter()
            .any(|marker| marker.item_start == definition.keyword_offset)
        {
            definition.visibility = Visibility::Public;
        }
    }
    // What is left belongs to a function: not the marker a template
    // just claimed, and not one written inside a template body.
    let item_starts: Vec<usize> = markers
        .iter()
        .map(|m| m.item_start)
        .filter(|start| !definition_ranges.iter().any(|range| range.contains(start)))
        .collect();

    // Two passes: the blank lines a stripped block leaves behind can only
    // be measured once the stripping is done, so the first pass finds
    // them and the second re-cuts the original with those edges added.
    // Cutting the original twice (rather than cutting the result) is what
    // keeps every surviving byte mapped straight back to its own line.
    let first = apply_cuts(source, &cuts, &item_starts);
    for edge in blank_edge_ranges(&first.text) {
        let start = first.to_original(edge.start).unwrap_or(source.len());
        let end = first.to_original(edge.end).unwrap_or(source.len());
        cuts.push(start..end);
    }
    let cut = apply_cuts(source, &cuts, &item_starts);

    let mut body = cut.text;
    // Keep the body newline-terminated so concatenation never joins the
    // last line of one file to the first line of the next.
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }

    let items = resolve_items(&extract_signatures(&body), &cut.markers).map_err(|error| {
        let (line, column) = line_col(&body, error.offset());
        LaplaceLibError::Visibility {
            path: path.to_path_buf(),
            line,
            column,
            help: error.help(),
            error,
        }
    })?;

    let template_lines = templates
        .iter()
        .map(|t| (t.name.clone(), line_col(source, t.keyword_offset).0))
        .collect();
    let macro_lines = macros
        .iter()
        .map(|m| (m.name.clone(), line_col(source, m.keyword_offset).0))
        .collect();

    Ok(LaplaceLibFile {
        imports,
        body,
        items,
        templates,
        template_lines,
        macros,
        macro_lines,
        segments: cut.segments,
    })
}

/// The leading and trailing runs of whitespace-only lines in `body`.
///
/// Removing a `library { }` block or unwrapping a `functions { }` wrapper
/// leaves the blank lines that surrounded it behind. Everything *between*
/// the first and last real lines is untouched, byte for byte -- this only
/// stops each library file from contributing a run of blank lines to the
/// compiled output.
fn blank_edge_ranges(body: &str) -> Vec<Range<usize>> {
    let mut line_starts = vec![0usize];
    for (i, b) in body.bytes().enumerate() {
        if b == b'\n' {
            line_starts.push(i + 1);
        }
    }

    let is_blank = |start: usize| {
        let end = body[start..].find('\n').map_or(body.len(), |i| start + i);
        body[start..end].trim().is_empty()
    };

    let Some(&first) = line_starts
        .iter()
        .find(|&&start| start < body.len() && !is_blank(start))
    else {
        // Nothing but blank lines: the whole body goes.
        let whole = 0..body.len();
        return if body.is_empty() {
            Vec::new()
        } else {
            vec![whole]
        };
    };
    let last = line_starts
        .iter()
        .rev()
        .find(|&&start| start < body.len() && !is_blank(start))
        .copied()
        .expect("a non-blank line exists");
    let last_end = body[last..].find('\n').map_or(body.len(), |i| last + i + 1);

    let mut ranges = Vec::new();
    if first > 0 {
        ranges.push(0..first);
    }
    if last_end < body.len() {
        ranges.push(last_end..body.len());
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::origin::{FileOrigin, PackageOrigin};

    fn parse_str(source: &str) -> Result<LaplaceLibFile, LaplaceLibError> {
        parse(Path::new("stats.laplacelib"), source)
    }

    #[test]
    fn bare_function_definitions_with_no_blocks_pass_straight_through() {
        let source =
            "// @laplace\n// @brief Adds one.\nreal add_one(real x) {\n  return x + 1;\n}\n";
        let parsed = parse_str(source).unwrap();
        assert!(parsed.imports.is_empty());
        assert_eq!(parsed.body, source);
    }

    #[test]
    fn blank_lines_left_behind_by_stripping_are_trimmed_at_the_edges_only() {
        let source = "library {\n  import stats\n}\n\n\nreal a() {\n  return 1;\n}\n\n\nreal b() {\n  return 2;\n}\n\n\n";
        let parsed = parse_str(source).unwrap();
        assert_eq!(
            parsed.body,
            // Interior blank lines survive byte for byte; only the run the
            // stripped `library { }` block left at the top, and the trailing
            // run, are removed.
            "real a() {\n  return 1;\n}\n\n\nreal b() {\n  return 2;\n}\n"
        );
    }

    #[test]
    fn a_file_with_nothing_but_a_library_block_contributes_no_body() {
        let parsed = parse_str("library {\n  import stats\n}\n").unwrap();
        assert_eq!(parsed.imports.len(), 1);
        assert_eq!(parsed.body, "");
    }

    #[test]
    fn a_library_block_is_parsed_and_removed() {
        let source = "library {\n  import stats@1.0.0\n}\n\nreal f(real x) {\n  return stats::mean_(x);\n}\n";
        let parsed = parse_str(source).unwrap();
        assert_eq!(
            parsed.imports,
            vec![ImportStatement {
                name: "stats".to_string(),
                version: Some("1.0.0".to_string()),
            }]
        );
        assert_eq!(
            parsed.body,
            "real f(real x) {\n  return stats::mean_(x);\n}\n"
        );
    }

    #[test]
    fn an_optional_functions_wrapper_is_unwrapped_keeping_its_contents_verbatim() {
        let source = "functions {\n  real f(real x) {\n    return x;\n  }\n}\n";
        let parsed = parse_str(source).unwrap();
        assert_eq!(parsed.body, "  real f(real x) {\n    return x;\n  }\n");
    }

    #[test]
    fn a_library_block_plus_a_functions_wrapper_works() {
        let source =
            "library {\n  import stats\n}\nfunctions {\n  real f() { return stats::m(); }\n}\n";
        let parsed = parse_str(source).unwrap();
        assert_eq!(parsed.imports.len(), 1);
        assert_eq!(parsed.body, "  real f() { return stats::m(); }\n");
        assert!(!parsed.body.contains("library"));
        assert!(!parsed.body.contains("functions {"));
    }

    #[test]
    fn every_model_shaped_block_is_rejected_by_name() {
        for (block, source) in [
            ("data", "data {\n  int n;\n}\n"),
            ("transformed data", "transformed data {\n  int n = 1;\n}\n"),
            ("parameters", "parameters {\n  real mu;\n}\n"),
            (
                "transformed parameters",
                "transformed parameters {\n  real nu;\n}\n",
            ),
            ("model", "model {\n  mu ~ normal(0, 1);\n}\n"),
            (
                "generated quantities",
                "generated quantities {\n  real y;\n}\n",
            ),
        ] {
            let err = parse_str(source).unwrap_err();
            assert_eq!(
                err,
                LaplaceLibError::ForbiddenBlock {
                    path: PathBuf::from("stats.laplacelib"),
                    block: block.to_string(),
                },
                "expected `{block}` to be rejected"
            );
            let rendered = err.to_string();
            assert!(rendered.contains("stats.laplacelib"), "{rendered}");
            assert!(rendered.contains(block), "{rendered}");
        }
    }

    #[test]
    fn a_malformed_import_names_the_file() {
        let source = "library {\n  import\n}\n";
        let err = parse_str(source).unwrap_err();
        assert!(matches!(err, LaplaceLibError::Imports { .. }), "{err:?}");
        assert!(err.to_string().contains("stats.laplacelib"));
    }

    #[test]
    fn a_model_block_keyword_inside_a_function_body_is_not_a_forbidden_block() {
        let source = "real f() {\n  real model_error = 1;\n  return model_error;\n}\n";
        assert_eq!(parse_str(source).unwrap().body, source);
    }

    // ---- `pub` visibility --------------------------------------------

    #[test]
    fn items_are_private_unless_marked_pub_and_pub_never_reaches_the_body() {
        let source = concat!(
            "pub real mean_(vector x) {\n",
            "  return sum_values(x) / num_elements(x);\n",
            "}\n",
            "\n",
            "real sum_values(vector x) {\n",
            "  return sum(x);\n",
            "}\n",
        );
        let parsed = parse_str(source).unwrap();

        assert_eq!(parsed.public_names(), vec!["mean_"]);
        assert_eq!(
            parsed
                .items
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>(),
            vec!["mean_", "sum_values"],
        );
        assert!(!parsed.body.contains("pub"), "{}", parsed.body);
        assert_eq!(parsed.body, source.replace("pub ", ""));
    }

    #[test]
    fn pub_works_inside_a_functions_wrapper() {
        let source = "functions {\n  pub real a() {\n    return 1;\n  }\n  real b() {\n    return 2;\n  }\n}\n";
        let parsed = parse_str(source).unwrap();
        assert_eq!(parsed.public_names(), vec!["a"]);
        assert!(!parsed.body.contains("pub"));
    }

    #[test]
    fn pub_with_a_library_block_and_a_doc_comment() {
        let source = concat!(
            "library {\n  import stats\n}\n",
            "\n",
            "// @laplace\n",
            "// @brief Fits.\n",
            "pub real fit(vector y) {\n",
            "  return stats::mean_(y);\n",
            "}\n",
        );
        let parsed = parse_str(source).unwrap();
        assert_eq!(parsed.imports.len(), 1);
        assert_eq!(parsed.public_names(), vec!["fit"]);
        assert!(parsed.body.starts_with("// @laplace"));
    }

    #[test]
    fn a_dangling_pub_names_the_file_and_the_line() {
        let source = "real a() {\n  return 1;\n}\n\npub\n// note\n";
        let err = parse_str(source).unwrap_err();
        let rendered = err.to_string();
        assert!(matches!(err, LaplaceLibError::Visibility { .. }), "{err:?}");
        assert!(rendered.contains("stats.laplacelib:"), "{rendered}");
        assert!(rendered.contains("help:"), "{rendered}");
    }

    // ---- templates ----------------------------------------------------

    const NCP: &str = r#"pub @template ncp($name: ident, $N: expr) {
  parameters {
    vector[$N] ${name}_raw;
  }
  model {
    ${name}_raw ~ std_normal();
  }
}
"#;

    #[test]
    fn a_template_is_parsed_and_cut_entirely_out_of_the_body() {
        let source = format!(
            "{NCP}
real helper(real x) {{
  return x;
}}
"
        );
        let parsed = parse_str(&source).unwrap();

        assert_eq!(parsed.templates.len(), 1);
        assert_eq!(parsed.templates[0].name, "ncp");
        assert_eq!(parsed.public_templates(), vec!["ncp"]);
        // None of it reaches the Stan body.
        assert_eq!(parsed.body, "real helper(real x) {\n  return x;\n}\n");
        assert!(!parsed.body.contains("@template"));
        assert!(!parsed.body.contains('$'));
    }

    #[test]
    fn a_template_without_pub_is_private() {
        let source = NCP.replace("pub ", "");
        let parsed = parse_str(&source).unwrap();
        assert!(parsed.public_templates().is_empty());
        assert_eq!(parsed.templates.len(), 1);
    }

    #[test]
    fn a_templates_pub_marker_does_not_leak_onto_the_next_function() {
        let source = format!(
            "{NCP}
real helper(real x) {{
  return x;
}}
"
        );
        let parsed = parse_str(&source).unwrap();
        // `helper` has no `pub` of its own, and the template's must not
        // be mistaken for one.
        assert!(
            parsed.public_names().is_empty(),
            "{:?}",
            parsed.public_names()
        );
    }

    #[test]
    fn templates_and_pub_functions_coexist() {
        let source = format!(
            "{NCP}
pub real helper(real x) {{
  return x;
}}
"
        );
        let parsed = parse_str(&source).unwrap();
        assert_eq!(parsed.public_templates(), vec!["ncp"]);
        assert_eq!(parsed.public_names(), vec!["helper"]);
    }

    #[test]
    fn a_template_body_is_not_scanned_for_pub_markers_or_signatures() {
        // `parameters { }` inside a template must not be mistaken for a
        // forbidden model-shaped block, and the body's statements must
        // not be read as function signatures.
        let parsed = parse_str(NCP).unwrap();
        assert!(parsed.items.is_empty(), "{:?}", parsed.items);
        assert_eq!(parsed.body, "");
    }

    #[test]
    fn a_template_definition_error_names_the_file_and_line() {
        let source = "pub @template bad($n: ident) {
  model {
    real tmp = 1;
  }
}
";
        let err = parse_str(source).unwrap_err();
        assert!(matches!(err, LaplaceLibError::Template { .. }), "{err:?}");
        let rendered = err.to_string();
        assert!(rendered.contains("stats.laplacelib:"), "{rendered}");
        assert!(rendered.contains("fixed name"), "{rendered}");
    }

    #[test]
    fn a_template_and_a_library_block_together_work() {
        let source = format!(
            "library {{
  import other
}}

{NCP}"
        );
        let parsed = parse_str(&source).unwrap();
        assert_eq!(parsed.imports.len(), 1);
        assert_eq!(parsed.templates.len(), 1);
        assert_eq!(parsed.body, "");
    }

    // ---- line mapping back to the original file ----------------------

    #[test]
    fn body_offsets_map_back_to_the_lines_they_were_written_on() {
        let source = concat!(
            "library {\n",        // 1
            "  import stats\n",   // 2
            "}\n",                // 3
            "\n",                 // 4
            "pub real fit() {\n", // 5
            "  return 1;\n",      // 6
            "}\n",                // 7
        );
        let parsed = parse_str(source).unwrap();
        let origin = PackageOrigin {
            files: vec![FileOrigin {
                file: "stats.laplacelib".to_string(),
                body_range: 0..parsed.body.len(),
                segments: parsed.segments.clone(),
            }],
        };

        // `fit` is the first thing in the body but line 5 of the file.
        let at = parsed.body.find("real fit").unwrap();
        assert_eq!(origin.locate(&parsed.body, at).unwrap().line, 5);
        let ret = parsed.body.find("return 1").unwrap();
        assert_eq!(origin.locate(&parsed.body, ret).unwrap().line, 6);
    }

    #[test]
    fn stripping_pub_does_not_shift_the_line_of_what_follows() {
        let source = "real a() {\n  return 1;\n}\n\npub real b() {\n  return 2;\n}\n";
        let parsed = parse_str(source).unwrap();
        let origin = PackageOrigin {
            files: vec![FileOrigin {
                file: "stats.laplacelib".to_string(),
                body_range: 0..parsed.body.len(),
                segments: parsed.segments.clone(),
            }],
        };
        let at = parsed.body.find("real b").unwrap();
        assert_eq!(origin.locate(&parsed.body, at).unwrap().line, 5);
    }

    // ---- `__` is reserved --------------------------------------------

    #[test]
    fn a_double_underscore_identifier_is_rejected_with_a_location() {
        let source = "real my__helper(real x) {\n  return x;\n}\n";
        let err = parse_str(source).unwrap_err();
        assert!(
            matches!(err, LaplaceLibError::ReservedIdentifier { .. }),
            "{err:?}"
        );
        let rendered = err.to_string();
        assert!(rendered.contains("my__helper"), "{rendered}");
        assert!(rendered.contains("stats.laplacelib:1:6"), "{rendered}");
        assert!(rendered.contains("help:"), "{rendered}");
    }

    #[test]
    fn a_double_underscore_in_a_comment_is_still_fine() {
        let source = "// compiles to stats__a\nreal a() {\n  return 1;\n}\n";
        assert!(parse_str(source).is_ok());
    }

    #[test]
    fn parsing_is_deterministic() {
        let source = "library {\n  import a\n  import b\n}\nreal f() { return 1; }\n";
        assert_eq!(parse_str(source).unwrap(), parse_str(source).unwrap());
    }
}
