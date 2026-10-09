//! Expanding `@expand` in place, inside one block.
//!
//! Where a template is cut into pieces and spliced into several blocks,
//! a macro replaces the `@expand` line it was written on -- keeping its
//! indentation, and repeating its body once per element if the header
//! declared an `each` parameter.
//!
//! Everything is expressed as edits on the user's original source, like
//! template expansion, so the rest of the compiler keeps reporting the
//! lines the user actually wrote.

use std::ops::Range;

use thiserror::Error;

use crate::codegen::rename::mangle_fragment;
use crate::monomorphize::TextEdit;
use crate::parser::blocks::{find_top_level_blocks, BlockKind};
use crate::parser::declarations::declarations;
use crate::parser::macros::{ExpandStatement, MacroDef};
use crate::parser::origin::line_col;
use crate::parser::statements::statements;

use super::{check_argument, find_collision, reindent, substitute, wrap, Declared, ExpandError};

/// A macro a model may expand, with everything expansion needs.
#[derive(Debug, Clone)]
pub struct MacroSource {
    pub package: String,
    pub version: String,
    pub def: MacroDef,
    /// Every function name the defining package defines. Calls in a
    /// macro body resolve in the *defining* package's scope, and the
    /// expansion lands outside that package's text.
    pub functions: Vec<String>,
    /// Where the macro is written, for a warning.
    pub origin: String,
}

/// What macro expansion decided.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MacroExpansion {
    pub edits: Vec<TextEdit>,
    /// Byte ranges another pass should leave alone: the `pkg::name` in
    /// an `@expand` line is not a function call.
    pub reserved: Vec<Range<usize>>,
    /// The whole `@expand` statements, which are about to be replaced.
    pub statement_ranges: Vec<Range<usize>>,
    /// Names the expansions declare, so a later pass can keep checking
    /// for collisions against them.
    pub declared: Vec<Declared>,
    pub warnings: Vec<String>,
}

impl MacroExpansion {
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn covers(&self, range: &Range<usize>) -> bool {
        self.reserved
            .iter()
            .any(|owned| owned.start <= range.start && range.end <= owned.end)
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MacroExpandError {
    #[error(
        "`@expand {package}::{name}` names a package the model does not import\n  --> \
         {location}\n  help: add `import {package}` to the `library {{ }}` block"
    )]
    PackageNotImported {
        package: String,
        name: String,
        location: String,
    },

    #[error("package `{package}` has no macro `{name}`\n  --> {location}\n  help: {help}")]
    UnknownMacro {
        package: String,
        name: String,
        help: String,
        location: String,
    },

    #[error(
        "macro `{package}::{name}` is private to `{package}`\n  --> {location}\n  help: only \
         macros marked `pub` can be used outside their package"
    )]
    MacroIsPrivate {
        package: String,
        name: String,
        location: String,
    },

    #[error(
        "`@expand` must be inside a Stan block\n  --> {location}\n  help: a macro expands to \
         statements, so it needs a block to expand into"
    )]
    OutsideAnyBlock { location: String },

    #[error(
        "macro `{package}::{name}` cannot be expanded in `{block}`\n  --> {location}\n  help: it \
         declares `in {targets}`"
    )]
    WrongBlock {
        package: String,
        name: String,
        block: String,
        targets: String,
        location: String,
    },

    #[error(
        "`@expand {package}::{name}` takes {expected} argument(s) but was given {found}\n  --> \
         {location}\n  help: the macro declares {signature}"
    )]
    ArgumentCount {
        package: String,
        name: String,
        expected: usize,
        found: usize,
        signature: String,
        location: String,
    },

    #[error(
        "`${param}` is an `each` parameter, so it needs a list\n  --> {location}\n  help: write \
         it as `[a, b, c]`"
    )]
    NotAList { param: String, location: String },

    #[error(
        "the list for `${param}` is empty, so `@expand {package}::{name}` would produce \
         nothing\n  --> {location}\n  help: give it at least one element, or remove the \
         `@expand` line"
    )]
    EmptyList {
        param: String,
        package: String,
        name: String,
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
}

/// Expand every `@expand` in `source`.
///
/// `already_declared` are names earlier passes will introduce (a
/// template expansion's), so a macro cannot quietly collide with one.
pub fn expand(
    file: &str,
    source: &str,
    expands: &[ExpandStatement],
    imported: &[String],
    available: &[MacroSource],
    already_declared: &[Declared],
) -> Result<MacroExpansion, MacroExpandError> {
    let mut expansion = MacroExpansion::default();
    if expands.is_empty() {
        return Ok(expansion);
    }

    let blocks = find_top_level_blocks(source);
    let mut existing = existing_declarations(file, source, &blocks);
    existing.extend_from_slice(already_declared);

    for statement in expands {
        let (line, column) = line_col(source, statement.keyword_offset);
        let location = format!("{file}:{line}:{column}");
        let source_macro = resolve(statement, imported, available, &location)?;
        let def = &source_macro.def;

        // Which block is this in, and does the macro allow it?
        let block = blocks
            .iter()
            .filter(|block| block.kind != BlockKind::Library)
            .find(|block| block.body_range.contains(&statement.keyword_offset))
            .ok_or_else(|| MacroExpandError::OutsideAnyBlock {
                location: location.clone(),
            })?;
        if !def.targets.contains(&block.kind) {
            return Err(MacroExpandError::WrongBlock {
                package: statement.package.clone(),
                name: statement.name.clone(),
                block: block.kind.keyword().to_string(),
                targets: def
                    .targets
                    .iter()
                    .map(|b| b.keyword())
                    .collect::<Vec<_>>()
                    .join(", "),
                location: location.clone(),
            });
        }

        if def.params.len() != statement.args.len() {
            return Err(MacroExpandError::ArgumentCount {
                package: statement.package.clone(),
                name: statement.name.clone(),
                expected: def.params.len(),
                found: statement.args.len(),
                signature: def.signature(),
                location,
            });
        }

        for name in &def.unused {
            expansion.warnings.push(format!(
                "warning: macro `{}::{}` declares `${name}` and never uses it\n  --> {}",
                statement.package, statement.name, source_macro.origin
            ));
        }

        // The `each` parameter's list decides how many times the body
        // repeats; everything else is bound once.
        let each_values: Vec<String> = match def.each_param() {
            None => vec![String::new()],
            Some(param) => {
                let argument = &statement.args[param_index(def, &param.name)];
                let elements = ExpandStatement::list_elements(argument).ok_or_else(|| {
                    MacroExpandError::NotAList {
                        param: param.name.clone(),
                        location: location.clone(),
                    }
                })?;
                if elements.is_empty() {
                    return Err(MacroExpandError::EmptyList {
                        param: param.name.clone(),
                        package: statement.package.clone(),
                        name: statement.name.clone(),
                        location: location.clone(),
                    });
                }
                elements
            }
        };

        let mut repetitions = String::new();
        for value in &each_values {
            let mut bindings = Vec::with_capacity(def.params.len());
            for (param, argument) in def.params.iter().zip(&statement.args) {
                let text = if param.each {
                    value.as_str()
                } else {
                    argument.as_str()
                };
                let binding =
                    check_argument(param, text).map_err(|error| MacroExpandError::Argument {
                        location: location.clone(),
                        help: error.help(),
                        error: Box::new(error),
                    })?;
                bindings.push((param.name.clone(), binding));
            }

            let substituted =
                substitute(&def.body, &bindings).map_err(|error| MacroExpandError::Argument {
                    location: location.clone(),
                    help: error.help(),
                    error: Box::new(error),
                })?;
            let mangled =
                mangle_fragment(&substituted, &source_macro.package, &source_macro.functions);
            repetitions.push_str(&reindent(trim_blank_lines(&mangled), statement.indent));
        }

        let mut incoming = Vec::new();
        for declaration in declarations(&repetitions) {
            incoming.push(Declared {
                name: declaration.name.clone(),
                source: format!("`{}` at {location}", statement.label()),
            });
        }
        if let Some((first, second)) = find_collision(&existing, &incoming) {
            return Err(MacroExpandError::Collision {
                name: second.name,
                first: first.source,
                second: second.source,
            });
        }
        existing.extend(incoming.iter().cloned());
        expansion.declared.extend(incoming);

        expansion.reserved.push(statement.reference_range.clone());
        expansion.statement_ranges.push(statement.range.clone());
        expansion.edits.push(TextEdit {
            range: statement.range.clone(),
            replacement: wrap(
                &repetitions,
                &statement.label(),
                &statement.label(),
                &format!("{file}:{line}"),
                statement.indent,
            ),
        });
    }

    expansion
        .edits
        .sort_by_key(|edit| (edit.range.start, edit.range.end));
    Ok(expansion)
}

fn param_index(def: &MacroDef, name: &str) -> usize {
    def.params
        .iter()
        .position(|p| p.name == name)
        .expect("the parameter came from this macro")
}

fn resolve<'a>(
    statement: &ExpandStatement,
    imported: &[String],
    available: &'a [MacroSource],
    location: &str,
) -> Result<&'a MacroSource, MacroExpandError> {
    if !imported.contains(&statement.package) {
        return Err(MacroExpandError::PackageNotImported {
            package: statement.package.clone(),
            name: statement.name.clone(),
            location: location.to_string(),
        });
    }
    let in_package: Vec<&MacroSource> = available
        .iter()
        .filter(|m| m.package == statement.package)
        .collect();
    let Some(found) = in_package
        .iter()
        .find(|m| m.def.name == statement.name)
        .copied()
    else {
        let names: Vec<&str> = in_package
            .iter()
            .filter(|m| m.def.visibility.is_public())
            .map(|m| m.def.name.as_str())
            .collect();
        return Err(MacroExpandError::UnknownMacro {
            package: statement.package.clone(),
            name: statement.name.clone(),
            help: if names.is_empty() {
                format!("`{}` defines no public macros", statement.package)
            } else {
                format!("`{}` defines {}", statement.package, names.join(", "))
            },
            location: location.to_string(),
        });
    };
    if !found.def.visibility.is_public() {
        return Err(MacroExpandError::MacroIsPrivate {
            package: statement.package.clone(),
            name: statement.name.clone(),
            location: location.to_string(),
        });
    }
    Ok(found)
}

/// Every name the model's blocks already declare at their top level.
fn existing_declarations(
    file: &str,
    source: &str,
    blocks: &[crate::parser::blocks::TopLevelBlock],
) -> Vec<Declared> {
    let mut out = Vec::new();
    for block in blocks.iter().filter(|b| b.kind != BlockKind::Library) {
        for declaration in declarations(&source[block.body_range.clone()])
            .into_iter()
            .filter(|d| d.depth == 0)
        {
            let (line, _) = line_col(source, block.body_range.start + declaration.range.start);
            out.push(Declared {
                name: declaration.name,
                source: format!("{file}:{line}"),
            });
        }
    }
    out
}

/// Whether `body` holds any statement at all, for the empty-expansion
/// check callers may want.
pub fn statement_count(body: &str) -> usize {
    statements(body).len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::macros::{find_expand_statements, find_macros};

    const PRIORS: &str = r#"@macro priors(each $p: ident, $dist: expr) : stmt in model {
  $p ~ $dist;
}
"#;

    const Z_SCORES: &str = r#"@macro z_scores(each $p: ident, $scale: expr) : stmt in transformed parameters {
  real ${p}_z = $p / $scale;
}
"#;

    fn macro_source(source: &str, functions: &[&str]) -> MacroSource {
        let def = find_macros(source, &|_| true).unwrap().remove(0);
        MacroSource {
            package: "stats".to_string(),
            version: "1.0.0".to_string(),
            def,
            functions: functions.iter().map(|f| f.to_string()).collect(),
            origin: "stats v1.0.0 (stats/stats.laplacelib:1)".to_string(),
        }
    }

    fn run(source: &str, available: &[MacroSource]) -> Result<String, MacroExpandError> {
        let expands = find_expand_statements(source).unwrap();
        let expansion = expand(
            "model.laplace",
            source,
            &expands,
            &["stats".to_string()],
            available,
            &[],
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

    #[test]
    fn the_worked_example_from_the_spec_expands_as_documented() {
        let source = concat!(
            "parameters {\n  real alpha;\n  real beta;\n  real<lower=0> gamma;\n}\n",
            "model {\n",
            "  @expand stats::priors([alpha, beta, gamma], normal(0, 1));\n",
            "  y ~ normal(alpha + beta * x, gamma);\n",
            "}\n",
        );
        let out = run(source, &[macro_source(PRIORS, &[])]).unwrap();

        assert_eq!(
            out,
            concat!(
                "parameters {\n  real alpha;\n  real beta;\n  real<lower=0> gamma;\n}\n",
                "model {\n",
                "  // begin @expand stats::priors -- model.laplace:7\n",
                "  alpha ~ normal(0, 1);\n",
                "  beta ~ normal(0, 1);\n",
                "  gamma ~ normal(0, 1);\n",
                "  // end @expand stats::priors\n",
                "  y ~ normal(alpha + beta * x, gamma);\n",
                "}\n",
            )
        );
        // The distribution passed as an `expr` is *not* parenthesized:
        // `alpha ~ (normal(0, 1));` would not be valid Stan.
        assert!(!out.contains("~ (normal"), "{out}");
    }

    #[test]
    fn a_macro_without_each_expands_once() {
        let single = "@macro one($p: ident) : stmt in model {\n  $p ~ std_normal();\n}\n";
        let source = "model {\n  @expand stats::one(alpha);\n}\n";
        let out = run(source, &[macro_source(single, &[])]).unwrap();
        assert_eq!(out.matches("alpha ~ std_normal();").count(), 1, "{out}");
    }

    #[test]
    fn a_one_element_list_expands_once() {
        let source = "model {\n  @expand stats::priors([alpha], normal(0, 1));\n}\n";
        let out = run(source, &[macro_source(PRIORS, &[])]).unwrap();
        assert_eq!(out.matches("~ normal(0, 1);").count(), 1, "{out}");
    }

    #[test]
    fn a_declaring_macro_builds_one_name_per_element() {
        let source = concat!(
            "parameters {\n  real a;\n  real b;\n}\n",
            "transformed parameters {\n  @expand stats::z_scores([a, b], 2.0);\n}\n",
        );
        let out = run(source, &[macro_source(Z_SCORES, &[])]).unwrap();
        // `2.0` is a primary expression, so it is not parenthesized.
        assert!(out.contains("  real a_z = a / 2.0;"), "{out}");
        assert!(out.contains("  real b_z = b / 2.0;"), "{out}");
    }

    #[test]
    fn the_expansion_keeps_the_indentation_of_the_line_it_replaced() {
        let source = "model {\n  for (i in 1:2) {\n    @expand stats::priors([alpha], normal(0, 1));\n  }\n}\n";
        let out = run(source, &[macro_source(PRIORS, &[])]).unwrap();
        assert!(out.contains("    // begin @expand stats::priors"), "{out}");
        assert!(out.contains("    alpha ~ normal(0, 1);"), "{out}");
        assert!(out.contains("    // end @expand stats::priors"), "{out}");
    }

    #[test]
    fn a_macro_body_calls_its_own_packages_functions_by_mangled_name() {
        let scaled = "@macro scaled(each $p: ident) : stmt in transformed parameters {\n  real ${p}_s = half($p);\n}\n";
        let source = "transformed parameters {\n  @expand stats::scaled([a]);\n}\n";
        let out = run(source, &[macro_source(scaled, &["half"])]).unwrap();
        assert!(out.contains("real a_s = stats__half(a);"), "{out}");
    }

    #[test]
    fn two_expands_in_one_block_both_happen() {
        let source = concat!(
            "model {\n",
            "  @expand stats::priors([alpha], normal(0, 1));\n",
            "  @expand stats::priors([beta], cauchy(0, 1));\n",
            "}\n",
        );
        let out = run(source, &[macro_source(PRIORS, &[])]).unwrap();
        assert!(out.contains("alpha ~ normal(0, 1);"), "{out}");
        assert!(out.contains("beta ~ cauchy(0, 1);"), "{out}");
    }

    // ---- errors ------------------------------------------------------

    fn error(source: &str, available: &[MacroSource]) -> MacroExpandError {
        run(source, available).expect_err("should not expand")
    }

    #[test]
    fn an_empty_list_is_an_error_rather_than_a_silent_no_op() {
        let err = error(
            "model {\n  @expand stats::priors([], normal(0, 1));\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(matches!(err, MacroExpandError::EmptyList { .. }), "{err:?}");
        assert!(err.to_string().contains("would produce nothing"), "{err}");
    }

    #[test]
    fn an_each_parameter_given_a_bare_value_asks_for_a_list() {
        let err = error(
            "model {\n  @expand stats::priors(alpha, normal(0, 1));\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(matches!(err, MacroExpandError::NotAList { .. }), "{err:?}");
        assert!(err.to_string().contains("[a, b, c]"), "{err}");
    }

    #[test]
    fn expanding_in_a_block_the_macro_does_not_allow_is_an_error() {
        let err = error(
            "generated quantities {\n  @expand stats::priors([alpha], normal(0, 1));\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(
            matches!(err, MacroExpandError::WrongBlock { .. }),
            "{err:?}"
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains("cannot be expanded in `generated quantities`"),
            "{rendered}"
        );
        assert!(rendered.contains("declares `in model`"), "{rendered}");
    }

    #[test]
    fn expanding_outside_every_block_is_an_error() {
        let err = error(
            "@expand stats::priors([alpha], normal(0, 1));\nmodel {\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(
            matches!(err, MacroExpandError::OutsideAnyBlock { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_name_the_model_already_declares_collides() {
        let source = concat!(
            "parameters {\n  real a;\n  real a_z;\n}\n",
            "transformed parameters {\n  @expand stats::z_scores([a], 2.0);\n}\n",
        );
        let err = error(source, &[macro_source(Z_SCORES, &[])]);
        assert!(matches!(err, MacroExpandError::Collision { .. }), "{err:?}");
        assert!(err.to_string().contains("`a_z` is declared twice"), "{err}");
    }

    #[test]
    fn a_repeated_element_collides_with_itself() {
        let source = "transformed parameters {\n  @expand stats::z_scores([a, a], 2.0);\n}\n";
        let err = error(source, &[macro_source(Z_SCORES, &[])]);
        assert!(matches!(err, MacroExpandError::Collision { .. }), "{err:?}");
    }

    #[test]
    fn a_collision_with_a_name_an_earlier_pass_will_declare_is_caught() {
        let source = "transformed parameters {\n  @expand stats::z_scores([a], 2.0);\n}\n";
        let expands = find_expand_statements(source).unwrap();
        let err = expand(
            "model.laplace",
            source,
            &expands,
            &["stats".to_string()],
            &[macro_source(Z_SCORES, &[])],
            &[Declared {
                name: "a_z".to_string(),
                source: "`@use stats::t(a)` at model.laplace:1".to_string(),
            }],
        )
        .unwrap_err();
        assert!(matches!(err, MacroExpandError::Collision { .. }), "{err:?}");
        assert!(err.to_string().contains("@use stats::t(a)"), "{err}");
    }

    #[test]
    fn an_expression_in_an_ident_slot_is_an_error() {
        let err = error(
            "model {\n  @expand stats::priors([alpha + 1], normal(0, 1));\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(matches!(err, MacroExpandError::Argument { .. }), "{err:?}");
        assert!(err.to_string().contains("not a valid identifier"), "{err}");
    }

    #[test]
    fn the_wrong_number_of_arguments_names_the_signature() {
        let err = error(
            "model {\n  @expand stats::priors([alpha]);\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(
            matches!(err, MacroExpandError::ArgumentCount { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("each $p: ident"), "{err}");
    }

    #[test]
    fn an_unknown_macro_lists_what_the_package_does_offer() {
        let err = error(
            "model {\n  @expand stats::nope([a], 1);\n}\n",
            &[macro_source(PRIORS, &[])],
        );
        assert!(
            matches!(err, MacroExpandError::UnknownMacro { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("defines priors"), "{err}");
    }

    #[test]
    fn a_private_macro_cannot_be_expanded_from_outside() {
        let mut private = macro_source(PRIORS, &[]);
        private.def.visibility = crate::parser::visibility::Visibility::Private;
        let err = error(
            "model {\n  @expand stats::priors([a], normal(0, 1));\n}\n",
            &[private],
        );
        assert!(
            matches!(err, MacroExpandError::MacroIsPrivate { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_package_the_model_does_not_import_is_an_error() {
        let source = "model {\n  @expand other::priors([a], normal(0, 1));\n}\n";
        let expands = find_expand_statements(source).unwrap();
        let err = expand(
            "model.laplace",
            source,
            &expands,
            &["stats".to_string()],
            &[macro_source(PRIORS, &[])],
            &[],
        )
        .unwrap_err();
        assert!(
            matches!(err, MacroExpandError::PackageNotImported { .. }),
            "{err:?}"
        );
    }

    // ---- nothing to do ------------------------------------------------

    #[test]
    fn a_file_with_no_expand_statements_plans_nothing() {
        let source = "model {\n  y ~ normal(0, 1);\n}\n";
        let expands = find_expand_statements(source).unwrap();
        let expansion = expand("model.laplace", source, &expands, &[], &[], &[]).unwrap();
        assert!(expansion.is_empty());
        assert_eq!(apply(source, &expansion.edits), source);
    }

    #[test]
    fn expansion_is_deterministic() {
        let source = "model {\n  @expand stats::priors([a, b], normal(0, 1));\n}\n";
        let available = [macro_source(PRIORS, &[])];
        assert_eq!(
            run(source, &available).unwrap(),
            run(source, &available).unwrap()
        );
    }
}
