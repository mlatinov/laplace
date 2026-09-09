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

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::parser::blocks::{find_top_level_blocks, BlockKind};
use crate::parser::library_block::{parse_import_statements, ImportStatement, LibraryBlockError};

/// The file extension that marks the library dialect.
pub const LAPLACELIB_EXTENSION: &str = "laplacelib";

/// A parsed `.laplacelib` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceLibFile {
    /// Everything this file imports, in source order.
    pub imports: Vec<ImportStatement>,
    /// The file's function definitions: `library { }` removed and any
    /// `functions { }` wrapper unwrapped. Byte-for-byte the original text
    /// otherwise.
    pub body: String,
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
}

/// Parse a `.laplacelib` file's text. `path` is used only to make errors
/// point at the offending file.
pub fn parse(path: &Path, source: &str) -> Result<LaplaceLibFile, LaplaceLibError> {
    let blocks = find_top_level_blocks(source);

    let mut imports = Vec::new();
    // Byte ranges to drop from the output, in ascending order: whole
    // `library { }` blocks, and the `functions {` / `}` bookends of any
    // `functions { }` wrapper (its contents are kept verbatim).
    let mut cuts: Vec<std::ops::Range<usize>> = Vec::new();

    for block in &blocks {
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
            }
            other => {
                return Err(LaplaceLibError::ForbiddenBlock {
                    path: path.to_path_buf(),
                    block: other.keyword().to_string(),
                })
            }
        }
    }

    let mut body = String::with_capacity(source.len());
    let mut cursor = 0usize;
    for cut in cuts {
        if cut.start >= cursor {
            body.push_str(&source[cursor..cut.start]);
            cursor = cut.end;
        }
    }
    body.push_str(&source[cursor..]);

    Ok(LaplaceLibFile {
        imports,
        body: trim_blank_edges(&body),
    })
}

/// Drop whitespace-only lines from both ends of a body.
///
/// Removing a `library { }` block or unwrapping a `functions { }` wrapper
/// leaves the blank lines that surrounded it behind. Everything *between*
/// the first and last real lines is untouched, byte for byte -- this only
/// stops each library file from contributing a run of blank lines to the
/// compiled output.
fn trim_blank_edges(body: &str) -> String {
    // Line-wise, so the first surviving line keeps its own indentation --
    // important for a `functions { }` wrapper, whose contents are indented.
    let lines: Vec<&str> = body.lines().collect();
    let first = lines.iter().position(|l| !l.trim().is_empty());
    let Some(first) = first else {
        return String::new();
    };
    let last = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .expect("a non-blank line exists");

    // Keep the body newline-terminated so concatenation never joins the last
    // line of one file to the first line of the next.
    let mut out = lines[first..=last].join("\n");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(source: &str) -> Result<LaplaceLibFile, LaplaceLibError> {
        parse(Path::new("stats.laplacelib"), source)
    }

    #[test]
    fn bare_function_definitions_with_no_blocks_pass_straight_through() {
        let source = "// @laplace\n// @brief Adds one.\nreal add_one(real x) {\n  return x + 1;\n}\n";
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
        assert_eq!(parsed.body, "real f(real x) {\n  return stats::mean_(x);\n}\n");
    }

    #[test]
    fn an_optional_functions_wrapper_is_unwrapped_keeping_its_contents_verbatim() {
        let source = "functions {\n  real f(real x) {\n    return x;\n  }\n}\n";
        let parsed = parse_str(source).unwrap();
        assert_eq!(parsed.body, "  real f(real x) {\n    return x;\n  }\n");
    }

    #[test]
    fn a_library_block_plus_a_functions_wrapper_works() {
        let source = "library {\n  import stats\n}\nfunctions {\n  real f() { return stats::m(); }\n}\n";
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

    #[test]
    fn parsing_is_deterministic() {
        let source = "library {\n  import a\n  import b\n}\nreal f() { return 1; }\n";
        assert_eq!(parse_str(source).unwrap(), parse_str(source).unwrap());
    }
}
