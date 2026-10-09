//! Expanding `@use` into the model's Stan blocks.
//!
//! A template's pieces go into the blocks they name, **before** the
//! user's own content in each one. That order is not cosmetic: a
//! template exists to create building blocks (`theta`) that the user's
//! own code then uses, and Stan requires a declaration before its use.
//!
//! Blocks the model does not have are created, in Stan's canonical
//! block order.
//!
//! Everything here is expressed as edits on the user's original source,
//! so the whole compiler keeps working in one coordinate system: an
//! error from any later pass still points at the line the user wrote.

use std::collections::BTreeMap;
use std::ops::Range;

use thiserror::Error;

use crate::codegen::rename::mangle_fragment;
use crate::monomorphize::TextEdit;
use crate::parser::blocks::{find_top_level_blocks, BlockKind};
use crate::parser::declarations::{declarations, identifier_uses};
use crate::parser::origin::line_col;
use crate::parser::template::{TemplateDef, UseStatement};

use super::{check_argument, find_collision, reindent, substitute, wrap, Declared, ExpandError};

/// How far into a block a piece is indented.
const BLOCK_INDENT: usize = 2;

/// A template a model may use, with everything expanding it needs.
#[derive(Debug, Clone)]
pub struct TemplateSource {
    pub package: String,
    pub version: String,
    pub def: TemplateDef,
    /// Every function name the defining package defines. Calls inside a
    /// template body resolve in the *defining* package's scope, and the
    /// expansion lands outside that package's text, so the mangling has
    /// to happen here.
    pub functions: Vec<String>,
    /// Where the template is written, for a provenance comment.
    pub origin: String,
}

/// What expansion decided.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Expansion {
    /// Edits on the project source: `@use` lines removed, pieces
    /// spliced into blocks, missing blocks created.
    pub edits: Vec<TextEdit>,
    /// Byte ranges another pass should leave alone -- the `pkg::name`
    /// in a `@use` line is not a function call.
    pub reserved: Vec<Range<usize>>,
    /// The whole `@use` statements, which are about to be replaced.
    pub use_ranges: Vec<Range<usize>>,
    /// Names the expansions declare, so the macro pass can check for
    /// collisions against them too.
    pub declared: Vec<Declared>,
    pub warnings: Vec<String>,
}

impl Expansion {
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Whether `range` lies inside something expansion owns.
    pub fn covers(&self, range: &Range<usize>) -> bool {
        self.reserved
            .iter()
            .any(|owned| owned.start <= range.start && range.end <= owned.end)
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum BlockExpandError {
    #[error(
        "`@use {package}::{template}` names a package the model does not import\n  --> \
         {location}\n  help: add `import {package}` to the `library {{ }}` block"
    )]
    PackageNotImported {
        package: String,
        template: String,
        location: String,
    },

    #[error("package `{package}` has no template `{template}`\n  --> {location}\n  help: {help}")]
    UnknownTemplate {
        package: String,
        template: String,
        help: String,
        location: String,
    },

    #[error(
        "template `{package}::{template}` is private to `{package}`\n  --> {location}\n  help: \
         only templates marked `pub` can be used outside their package"
    )]
    TemplateIsPrivate {
        package: String,
        template: String,
        location: String,
    },

    #[error(
        "`@use {package}::{template}` takes {expected} argument(s) but was given {found}\n  --> \
         {location}\n  help: the template declares {signature}"
    )]
    ArgumentCount {
        package: String,
        template: String,
        expected: usize,
        found: usize,
        signature: String,
        location: String,
    },

    #[error("{error}\n  --> {location}\n  help: {help}")]
    Argument {
        location: String,
        help: String,
        #[source]
        error: Box<ExpandError>,
    },

    #[error(
        "`{name}` is declared twice: once by {first}, and again by {second}\n  help: give one of \
         them a different name -- that is what the `ident` placeholder is for"
    )]
    Collision {
        name: String,
        first: String,
        second: String,
    },

    #[error(
        "{later} declares `{name}`, which {earlier} already uses in the same `{block}` block\n  \
         help: put the `@use` that declares `{name}` first -- within a block, expansion order is \
         the order the `@use` lines appear in"
    )]
    OutOfOrder {
        name: String,
        block: String,
        earlier: String,
        later: String,
    },
}

/// Expand every `@use` in `source`.
///
/// `imported` are the packages the model's `library { }` block names;
/// `available` is every template in the build. `file` names the source
/// in diagnostics.
pub fn expand(
    file: &str,
    source: &str,
    uses: &[UseStatement],
    imported: &[String],
    available: &[TemplateSource],
) -> Result<Expansion, BlockExpandError> {
    let mut expansion = Expansion::default();
    if uses.is_empty() {
        return Ok(expansion);
    }

    // Pieces to insert, grouped by the block they go into, in `@use`
    // order within each block.
    let mut per_block: BTreeMap<usize, Vec<(BlockKind, String, Contribution)>> = BTreeMap::new();
    let mut incoming: Vec<Declared> = Vec::new();

    for use_ in uses {
        let (line, column) = line_col(source, use_.keyword_offset);
        let location = format!("{file}:{line}:{column}");

        let provenance = format!("{file}:{line}");
        let source_template = resolve(use_, imported, available, &location)?;
        let def = &source_template.def;

        if def.params.len() != use_.args.len() {
            return Err(BlockExpandError::ArgumentCount {
                package: use_.package.clone(),
                template: use_.template.clone(),
                expected: def.params.len(),
                found: use_.args.len(),
                signature: signature_of(def),
                location,
            });
        }

        let mut bindings = Vec::with_capacity(def.params.len());
        for (param, argument) in def.params.iter().zip(&use_.args) {
            let binding =
                check_argument(param, argument).map_err(|error| BlockExpandError::Argument {
                    location: location.clone(),
                    help: error.help(),
                    error: Box::new(error),
                })?;
            bindings.push((param.name.clone(), binding));
        }

        for name in &def.unused {
            expansion.warnings.push(format!(
                "warning: template `{}::{}` declares `${name}` and never uses it\n  --> {}",
                use_.package, use_.template, source_template.origin
            ));
        }

        for piece in &def.pieces {
            let substituted =
                substitute(&piece.body, &bindings).map_err(|error| BlockExpandError::Argument {
                    location: location.clone(),
                    help: error.help(),
                    error: Box::new(error),
                })?;
            // Calls in a template body resolve in the defining
            // package's scope, and this text leaves that package.
            let mangled = mangle_fragment(
                &substituted,
                &source_template.package,
                &source_template.functions,
            );
            // The piece came from between a block's braces, so it
            // opens and closes with the newlines that hugged them.
            let body = reindent(trim_blank_lines(&mangled), BLOCK_INDENT);

            for declaration in declarations(&body).into_iter().filter(|d| d.depth == 0) {
                incoming.push(Declared {
                    name: declaration.name.clone(),
                    source: format!("`{}` at {location}", use_.label()),
                });
            }

            let text = wrap(
                &body,
                &use_.label(),
                &use_.short_label(),
                &provenance,
                BLOCK_INDENT,
            );
            per_block
                .entry(piece.block.canonical_order())
                .or_default()
                .push((
                    piece.block,
                    text,
                    Contribution {
                        label: use_.label(),
                        body,
                    },
                ));
        }

        expansion.reserved.push(use_.reference_range.clone());
        expansion.use_ranges.push(use_.range.clone());
        expansion.edits.push(TextEdit {
            range: use_.range.clone(),
            replacement: String::new(),
        });
    }

    check_collisions(file, source, &incoming)?;
    check_order(&per_block)?;
    expansion.declared = incoming;

    let blocks = existing_blocks(source);
    for (_, contributions) in per_block {
        let kind = contributions[0].0;
        let text: String = contributions
            .iter()
            .map(|(_, text, _)| text.as_str())
            .collect();
        expansion
            .edits
            .push(insertion(source, &blocks, kind, &text));
    }

    expansion
        .edits
        .sort_by_key(|edit| (edit.range.start, edit.range.end));
    Ok(expansion)
}

/// One template's contribution to one block, for the ordering check.
#[derive(Debug, Clone)]
struct Contribution {
    label: String,
    body: String,
}

fn resolve<'a>(
    use_: &UseStatement,
    imported: &[String],
    available: &'a [TemplateSource],
    location: &str,
) -> Result<&'a TemplateSource, BlockExpandError> {
    if !imported.contains(&use_.package) {
        return Err(BlockExpandError::PackageNotImported {
            package: use_.package.clone(),
            template: use_.template.clone(),
            location: location.to_string(),
        });
    }
    let in_package: Vec<&TemplateSource> = available
        .iter()
        .filter(|t| t.package == use_.package)
        .collect();
    let Some(found) = in_package
        .iter()
        .find(|t| t.def.name == use_.template)
        .copied()
    else {
        let names: Vec<&str> = in_package
            .iter()
            .filter(|t| t.def.visibility.is_public())
            .map(|t| t.def.name.as_str())
            .collect();
        return Err(BlockExpandError::UnknownTemplate {
            package: use_.package.clone(),
            template: use_.template.clone(),
            help: if names.is_empty() {
                format!("`{}` defines no public templates", use_.package)
            } else {
                format!("`{}` defines {}", use_.package, names.join(", "))
            },
            location: location.to_string(),
        });
    };
    if !found.def.visibility.is_public() {
        return Err(BlockExpandError::TemplateIsPrivate {
            package: use_.package.clone(),
            template: use_.template.clone(),
            location: location.to_string(),
        });
    }
    Ok(found)
}

fn signature_of(def: &TemplateDef) -> String {
    let params: Vec<String> = def
        .params
        .iter()
        .map(|p| format!("${}: {}", p.name, p.kind))
        .collect();
    format!("`{}({})`", def.name, params.join(", "))
}

/// Reject a name an expansion would declare twice, or that the model
/// already declares.
fn check_collisions(
    file: &str,
    source: &str,
    incoming: &[Declared],
) -> Result<(), BlockExpandError> {
    let mut existing: Vec<Declared> = Vec::new();
    for block in existing_blocks(source) {
        for declaration in declarations(&source[block.body.clone()])
            .into_iter()
            .filter(|d| d.depth == 0)
        {
            let (line, _) = line_col(source, block.body.start + declaration.range.start);
            existing.push(Declared {
                name: declaration.name,
                source: format!("{file}:{line}"),
            });
        }
    }

    if let Some((first, second)) = find_collision(&existing, incoming) {
        return Err(BlockExpandError::Collision {
            name: second.name,
            first: first.source,
            second: second.source,
        });
    }
    Ok(())
}

/// Within one block, a piece may only use names an *earlier* piece
/// declared -- expansion order is `@use` order, and Stan needs a
/// declaration before its use.
fn check_order(
    per_block: &BTreeMap<usize, Vec<(BlockKind, String, Contribution)>>,
) -> Result<(), BlockExpandError> {
    for contributions in per_block.values() {
        let kind = contributions[0].0;
        let declared: Vec<Vec<String>> = contributions
            .iter()
            .map(|(_, _, c)| declarations(&c.body).into_iter().map(|d| d.name).collect())
            .collect();

        for (index, (_, _, contribution)) in contributions.iter().enumerate() {
            let referenced: Vec<String> = identifier_uses(&contribution.body)
                .into_iter()
                .filter(|u| u.is_value_reference())
                .map(|u| u.name)
                .collect();
            for (later, names) in declared.iter().enumerate().skip(index + 1) {
                if let Some(name) = names.iter().find(|name| referenced.contains(name)) {
                    return Err(BlockExpandError::OutOfOrder {
                        name: name.clone(),
                        block: kind.keyword().to_string(),
                        earlier: format!("`{}`", contribution.label),
                        later: format!("`{}`", contributions[later].2.label),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Drop whitespace-only lines from both ends of a piece.
fn trim_blank_lines(text: &str) -> &str {
    let mut starts = vec![0usize];
    for (i, byte) in text.bytes().enumerate() {
        if byte == b'\n' && i + 1 < text.len() {
            starts.push(i + 1);
        }
    }
    let line_end = |start: usize| {
        text[start..]
            .find('\n')
            .map_or(text.len(), |i| start + i + 1)
    };
    let blank = |start: usize| text[start..line_end(start)].trim().is_empty();

    let Some(first) = starts.iter().copied().find(|&s| !blank(s)) else {
        return "";
    };
    let last = starts
        .iter()
        .copied()
        .rev()
        .find(|&s| !blank(s))
        .expect("a non-blank line exists");
    &text[first..line_end(last)]
}

/// A top-level Stan block of the model, with the byte range of its
/// contents.
#[derive(Debug, Clone)]
struct ExistingBlock {
    kind: BlockKind,
    whole: Range<usize>,
    body: Range<usize>,
}

fn existing_blocks(source: &str) -> Vec<ExistingBlock> {
    find_top_level_blocks(source)
        .into_iter()
        .filter(|block| block.kind != BlockKind::Library)
        .map(|block| ExistingBlock {
            kind: block.kind,
            whole: block.byte_range,
            body: block.body_range,
        })
        .collect()
}

/// The edit that puts `text` into the right block, creating the block
/// if the model does not have it.
fn insertion(source: &str, blocks: &[ExistingBlock], kind: BlockKind, text: &str) -> TextEdit {
    if let Some(block) = blocks.iter().find(|block| block.kind == kind) {
        // Straight after the opening brace, so the pieces precede the
        // model's own content.
        return TextEdit {
            range: block.body.start..block.body.start,
            replacement: format!("\n{text}"),
        };
    }

    let order = kind.canonical_order();
    let at = blocks
        .iter()
        .filter(|block| block.kind.canonical_order() < order)
        .map(|block| block.whole.end)
        .max()
        .or_else(|| {
            blocks
                .iter()
                .filter(|block| block.kind.canonical_order() > order)
                .map(|block| block.whole.start)
                .min()
        });

    match at {
        // After an earlier block, or before a later one.
        Some(at) if at > 0 && source[..at].ends_with('}') => TextEdit {
            range: at..at,
            replacement: format!("\n\n{} {{\n{text}}}", kind.keyword()),
        },
        Some(at) => TextEdit {
            range: at..at,
            replacement: format!("{} {{\n{text}}}\n\n", kind.keyword()),
        },
        // The model has no blocks at all yet.
        None => TextEdit {
            range: source.len()..source.len(),
            replacement: format!("{} {{\n{text}}}\n", kind.keyword()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::template::{find_templates, find_use_statements};

    const NCP: &str = r#"@template ncp($name: ident, $N: expr) {
  parameters {
    vector[$N] ${name}_raw;
    real<lower=0> ${name}_sigma;
  }
  transformed parameters {
    vector[$N] $name = ${name}_sigma * ${name}_raw;
  }
  model {
    ${name}_raw ~ std_normal();
    ${name}_sigma ~ exponential(1);
  }
}
"#;

    const OBSERVATION: &str = r#"@template observation($y: ident, $mu: expr, $sigma: expr) {
  model {
    $y ~ lognormal($mu, $sigma);
  }
  generated quantities {
    real ${y}_rep = lognormal_rng($mu, $sigma);
  }
}
"#;

    fn template(source: &str, functions: &[&str]) -> TemplateSource {
        let def = find_templates(source, &|_| true).unwrap().remove(0);
        TemplateSource {
            package: "stats".to_string(),
            version: "1.0.0".to_string(),
            def,
            functions: functions.iter().map(|f| f.to_string()).collect(),
            origin: "stats v1.0.0 (stats/stats.laplacelib:1)".to_string(),
        }
    }

    fn run(source: &str, available: &[TemplateSource]) -> Result<String, BlockExpandError> {
        let uses = find_use_statements(source).unwrap();
        let expansion = expand(
            "model.laplace",
            source,
            &uses,
            &["stats".to_string()],
            available,
        )?;
        Ok(apply(source, &expansion.edits))
    }

    fn apply(source: &str, edits: &[TextEdit]) -> String {
        let mut out = String::new();
        let mut cursor = 0usize;
        for edit in edits {
            if edit.range.start < cursor {
                continue;
            }
            out.push_str(&source[cursor..edit.range.start]);
            out.push_str(&edit.replacement);
            cursor = edit.range.end;
        }
        out.push_str(&source[cursor..]);
        out
    }

    // ---- the worked example ------------------------------------------

    #[test]
    fn the_worked_example_from_the_spec_expands_as_documented() {
        let source = concat!(
            "library {\n  import stats\n}\n",
            "\n",
            "@use stats::ncp(theta, K);\n",
            "@use stats::observation(y, mu + theta, sigma);\n",
            "\n",
            "data {\n  int K;\n  vector[K] y;\n}\n",
            "parameters {\n  real mu;\n  real<lower=0> sigma;\n}\n",
        );
        let out = run(source, &[template(NCP, &[]), template(OBSERVATION, &[])]).unwrap();

        // Pieces land before the user's own content, in `@use` order.
        assert!(
            out.contains("parameters {\n  // begin @use stats::ncp(theta, K)"),
            "{out}"
        );
        assert!(
            out.contains("  vector[K] theta_raw;\n  real<lower=0> theta_sigma;"),
            "{out}"
        );
        assert!(
            out.contains("  // end @use stats::ncp\n\n  real mu;"),
            "{out}"
        );

        // `transformed parameters` and `generated quantities` did not
        // exist and were created.
        assert!(out.contains("transformed parameters {"), "{out}");
        assert!(
            out.contains("  vector[K] theta = theta_sigma * theta_raw;"),
            "{out}"
        );
        assert!(out.contains("generated quantities {"), "{out}");

        // The `expr` argument is parenthesized; the plain one is not.
        assert!(
            out.contains("  y ~ lognormal((mu + theta), sigma);"),
            "{out}"
        );
        assert!(
            out.contains("  real y_rep = lognormal_rng((mu + theta), sigma);"),
            "{out}"
        );

        // Both `@use` lines are gone, and so is every placeholder.
        assert!(!out.contains("@use stats::ncp(theta, K);\n"), "{out}");
        assert!(!out.contains('$'), "{out}");
    }

    #[test]
    fn created_blocks_are_in_canonical_stan_order() {
        let source = concat!(
            "@use stats::ncp(theta, K);\n",
            "data {\n  int K;\n}\n",
            "model {\n}\n",
        );
        let out = run(source, &[template(NCP, &[])]).unwrap();
        let data = out.find("data {").unwrap();
        let params = out.find("parameters {").unwrap();
        let tparams = out.find("transformed parameters {").unwrap();
        let model = out.find("model {").unwrap();
        assert!(data < params, "{out}");
        assert!(params < tparams, "{out}");
        assert!(tparams < model, "{out}");
    }

    #[test]
    fn two_uses_of_one_template_with_different_names_both_expand() {
        let source = concat!(
            "@use stats::ncp(theta, K);\n",
            "@use stats::ncp(beta, P);\n",
            "data {\n  int K;\n  int P;\n}\n",
        );
        let out = run(source, &[template(NCP, &[])]).unwrap();
        assert!(out.contains("vector[K] theta_raw;"), "{out}");
        assert!(out.contains("vector[P] beta_raw;"), "{out}");
        assert!(out.contains("theta_raw ~ std_normal();"), "{out}");
        assert!(out.contains("beta_raw ~ std_normal();"), "{out}");
    }

    #[test]
    fn a_template_body_calls_its_own_packages_functions_by_mangled_name() {
        let with_helper = r#"@template t($name: ident) {
  transformed parameters {
    real $name = scale_it(1);
  }
}
"#;
        let source = "@use stats::t(theta);\ndata {\n}\n";
        let out = run(source, &[template(with_helper, &["scale_it"])]).unwrap();
        assert!(out.contains("real theta = stats__scale_it(1);"), "{out}");
    }

    #[test]
    fn a_piece_carries_no_blank_padding_from_its_block_braces() {
        let source = "@use stats::ncp(theta, K);\ndata {\n  int K;\n}\n";
        let out = run(source, &[template(NCP, &[])]).unwrap();
        assert!(
            out.contains(
                "parameters {\n  // begin @use stats::ncp(theta, K) -- model.laplace:1\n  vector[K] theta_raw;\n"
            ),
            "{out}"
        );
        assert!(
            out.contains("  real<lower=0> theta_sigma;\n  // end @use stats::ncp\n"),
            "{out}"
        );
    }

    #[test]
    fn trimming_blank_lines_keeps_the_interior_intact() {
        assert_eq!(trim_blank_lines("\n  a;\n\n  b;\n\n"), "  a;\n\n  b;\n");
        assert_eq!(trim_blank_lines("  a;\n"), "  a;\n");
        assert_eq!(trim_blank_lines("\n\n"), "");
    }

    // ---- errors ------------------------------------------------------

    fn error(source: &str, available: &[TemplateSource]) -> BlockExpandError {
        run(source, available).expect_err("should not expand")
    }

    #[test]
    fn two_uses_with_the_same_name_collide() {
        let source = "@use stats::ncp(theta, K);\n@use stats::ncp(theta, P);\ndata {\n}\n";
        let err = error(source, &[template(NCP, &[])]);
        assert!(matches!(err, BlockExpandError::Collision { .. }), "{err:?}");
        let rendered = err.to_string();
        assert!(rendered.contains("theta_raw"), "{rendered}");
        assert!(rendered.contains("model.laplace:1"), "{rendered}");
        assert!(rendered.contains("model.laplace:2"), "{rendered}");
    }

    #[test]
    fn a_name_the_model_already_declares_collides() {
        let source = concat!(
            "@use stats::ncp(theta, K);\n",
            "parameters {\n  vector[K] theta_raw;\n}\n",
        );
        let err = error(source, &[template(NCP, &[])]);
        assert!(matches!(err, BlockExpandError::Collision { .. }), "{err:?}");
        assert!(err.to_string().contains("theta_raw"), "{err}");
    }

    #[test]
    fn a_piece_using_a_name_a_later_use_declares_is_an_error() {
        // `uses_theta` reads `theta`, which `ncp` declares -- but it is
        // written first, so within `transformed parameters` it would
        // come first too.
        let uses_theta = r#"@template uses_theta($out: ident, $src: expr) {
  transformed parameters {
    real $out = $src;
  }
}
"#;
        let source = concat!(
            "@use stats::uses_theta(scaled, theta * 2);\n",
            "@use stats::ncp(theta, K);\n",
            "data {\n  int K;\n}\n",
        );
        let err = error(source, &[template(NCP, &[]), template(uses_theta, &[])]);
        assert!(
            matches!(err, BlockExpandError::OutOfOrder { .. }),
            "{err:?}"
        );
        let rendered = err.to_string();
        assert!(rendered.contains("transformed parameters"), "{rendered}");
        assert!(rendered.contains("uses_theta"), "{rendered}");
        assert!(rendered.contains("ncp"), "{rendered}");
    }

    #[test]
    fn using_a_template_from_a_package_the_model_does_not_import_is_an_error() {
        let uses = find_use_statements("@use other::ncp(theta, K);\n").unwrap();
        let err = expand(
            "model.laplace",
            "@use other::ncp(theta, K);\n",
            &uses,
            &["stats".to_string()],
            &[template(NCP, &[])],
        )
        .unwrap_err();
        assert!(
            matches!(err, BlockExpandError::PackageNotImported { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("import other"), "{err}");
    }

    #[test]
    fn an_unknown_template_lists_what_the_package_does_offer() {
        let err = error("@use stats::nope(a);\n", &[template(NCP, &[])]);
        assert!(
            matches!(err, BlockExpandError::UnknownTemplate { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("defines ncp"), "{err}");
    }

    #[test]
    fn a_private_template_cannot_be_used_from_outside() {
        let mut private = template(NCP, &[]);
        private.def.visibility = crate::parser::visibility::Visibility::Private;
        let err = error("@use stats::ncp(theta, K);\n", &[private]);
        assert!(
            matches!(err, BlockExpandError::TemplateIsPrivate { .. }),
            "{err:?}"
        );
        assert!(
            err.to_string().contains("only templates marked `pub`"),
            "{err}"
        );
    }

    #[test]
    fn the_wrong_number_of_arguments_names_the_signature() {
        let err = error("@use stats::ncp(theta);\n", &[template(NCP, &[])]);
        assert!(
            matches!(err, BlockExpandError::ArgumentCount { .. }),
            "{err:?}"
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains("takes 2 argument(s) but was given 1"),
            "{rendered}"
        );
        assert!(rendered.contains("$name: ident"), "{rendered}");
    }

    #[test]
    fn an_expression_in_an_ident_slot_is_an_error_pointing_at_the_use_line() {
        let err = error("@use stats::ncp(mu + 1, K);\n", &[template(NCP, &[])]);
        assert!(matches!(err, BlockExpandError::Argument { .. }), "{err:?}");
        let rendered = err.to_string();
        assert!(rendered.contains("not a valid identifier"), "{rendered}");
        assert!(rendered.contains("model.laplace:1"), "{rendered}");
    }

    #[test]
    fn a_statement_in_an_expr_slot_is_an_error() {
        let err = error(
            "@use stats::ncp(theta, real x = 1;);\n",
            &[template(NCP, &[])],
        );
        assert!(matches!(err, BlockExpandError::Argument { .. }), "{err:?}");
    }

    // ---- nothing to do ------------------------------------------------

    #[test]
    fn a_file_with_no_use_statements_plans_nothing() {
        let source = "data {\n  int N;\n}\nmodel {\n}\n";
        let uses = find_use_statements(source).unwrap();
        let expansion = expand("model.laplace", source, &uses, &[], &[]).unwrap();
        assert!(expansion.is_empty());
        assert_eq!(apply(source, &expansion.edits), source);
    }

    #[test]
    fn an_unused_placeholder_produces_a_warning_not_an_error() {
        let spare = r#"@template t($used: ident, $spare: expr) {
  model {
    $used ~ std_normal();
  }
}
"#;
        let source = "@use stats::t(theta, 1);\nparameters {\n  real theta;\n}\n";
        let uses = find_use_statements(source).unwrap();
        let expansion = expand(
            "model.laplace",
            source,
            &uses,
            &["stats".to_string()],
            &[template(spare, &[])],
        )
        .unwrap();
        assert_eq!(expansion.warnings.len(), 1);
        assert!(
            expansion.warnings[0].contains("$spare"),
            "{:?}",
            expansion.warnings
        );
    }

    #[test]
    fn expansion_is_deterministic() {
        let source = "@use stats::ncp(theta, K);\n@use stats::ncp(beta, P);\ndata {\n}\n";
        let available = [template(NCP, &[])];
        assert_eq!(
            run(source, &available).unwrap(),
            run(source, &available).unwrap()
        );
    }
}
