//! The checks every expandable body gets, whether it belongs to a
//! `@template` or a `@macro`.
//!
//! Both constructs substitute placeholders into Stan and splice the
//! result into someone else's model, so both need the same guarantees,
//! and they live here rather than in two places that could drift:
//!
//! - no nesting: a body may not contain `@use` or `@expand`
//! - every placeholder it uses is declared in its header
//! - **hygiene:** every variable it declares is named from an `ident`
//!   placeholder, so using it twice cannot collide
//! - **no capture:** it may not reach for one of the model's variables
//!
//! The last two are connected. Once fixed-name declarations are ruled
//! out, the only names a body can legally mention are its own
//! placeholders -- which are not identifiers at all until substitution
//! -- and its own `for` loop variables. Anything else left over is a
//! reference to a variable the body does not own, which is exactly what
//! capture is. That is why no catalogue of Stan builtins is needed: a
//! function call is excluded by being followed by `(`, and what remains
//! is Stan's own small list of reserved words.

use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::declarations::{declarations, identifier_uses, loop_variables};
use crate::parser::placeholder::{find_uses, PlaceholderDecl, PlaceholderError, PlaceholderKind};

/// Which construct a body belongs to, for error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyOwner {
    Template,
    Macro,
}

impl BodyOwner {
    pub fn describe(self) -> &'static str {
        match self {
            BodyOwner::Template => "template",
            BodyOwner::Macro => "macro",
        }
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum BodyError {
    #[error("{owner} `{name}` contains a nested `{keyword}`")]
    Nested {
        owner: &'static str,
        name: String,
        keyword: &'static str,
        offset: usize,
    },

    #[error("{owner} `{name}` uses `${param}`, which its header does not declare")]
    UndeclaredPlaceholder {
        owner: &'static str,
        name: String,
        param: String,
        offset: usize,
    },

    #[error("{owner} `{name}` declares `{declared}` with a fixed name")]
    FixedNameDeclaration {
        owner: &'static str,
        name: String,
        declared: String,
        offset: usize,
    },

    #[error(
        "{owner} `{name}` declares `{declared}`, whose name comes from `${param}` -- an \
         `{kind}` placeholder, not an `ident`"
    )]
    DeclarationFromNonIdent {
        owner: &'static str,
        name: String,
        declared: String,
        param: String,
        kind: PlaceholderKind,
        offset: usize,
    },

    #[error("{owner} `{name}` refers to `{reference}`, which it does not declare")]
    CapturedVariable {
        owner: &'static str,
        name: String,
        reference: String,
        offset: usize,
    },

    #[error("{owner} `{name}`: {source}")]
    Placeholder {
        owner: &'static str,
        name: String,
        offset: usize,
        #[source]
        source: PlaceholderError,
    },
}

impl BodyError {
    pub fn offset(&self) -> usize {
        match self {
            BodyError::Nested { offset, .. }
            | BodyError::UndeclaredPlaceholder { offset, .. }
            | BodyError::FixedNameDeclaration { offset, .. }
            | BodyError::DeclarationFromNonIdent { offset, .. }
            | BodyError::CapturedVariable { offset, .. }
            | BodyError::Placeholder { offset, .. } => *offset,
        }
    }

    pub fn help(&self) -> String {
        match self {
            BodyError::Nested { keyword, owner, .. } => {
                format!("`{keyword}` cannot appear inside a {owner} body in this version")
            }
            BodyError::UndeclaredPlaceholder { param, .. } => {
                format!("add `${param}: ident` or `${param}: expr` to the header")
            }
            BodyError::FixedNameDeclaration { declared, .. } => format!(
                "name it from an `ident` placeholder, as `${{name}}_{declared}` -- a fixed name \
                 would collide the second time this is used"
            ),
            BodyError::DeclarationFromNonIdent { param, .. } => {
                format!("declare `${param}: ident`, since it names a variable")
            }
            BodyError::CapturedVariable {
                reference, owner, ..
            } => format!(
                "a {owner} cannot reach for one of the model's own variables; pass `{reference}` \
                 in as a placeholder instead"
            ),
            BodyError::Placeholder { source, .. } => source.help(),
        }
    }
}

/// Run every shared check over one body.
///
/// `base` is the body's byte offset in the file it came from, so the
/// reported offsets are absolute. Returns the placeholder names the
/// body actually used, which the caller turns into an unused-placeholder
/// warning.
pub fn check(
    owner: BodyOwner,
    name: &str,
    params: &[PlaceholderDecl],
    body: &str,
    base: usize,
) -> Result<Vec<String>, BodyError> {
    let described = owner.describe();
    let mut used: Vec<String> = Vec::new();

    for (keyword, label) in [
        ("@use", "@use"),
        ("@template", "@template"),
        ("@expand", "@expand"),
        ("@macro", "@macro"),
    ] {
        if let Some(offset) = find_keyword(body, keyword) {
            return Err(BodyError::Nested {
                owner: described,
                name: name.to_string(),
                keyword: label,
                offset: base + offset,
            });
        }
    }

    let uses = find_uses(body).map_err(|source| BodyError::Placeholder {
        owner: described,
        name: name.to_string(),
        offset: base + source.offset(),
        source,
    })?;

    for use_ in &uses {
        if !params.iter().any(|p| p.name == use_.name) {
            return Err(BodyError::UndeclaredPlaceholder {
                owner: described,
                name: name.to_string(),
                param: use_.name.clone(),
                offset: base + use_.range.start,
            });
        }
        if !used.contains(&use_.name) {
            used.push(use_.name.clone());
        }
    }

    // Hygiene: a declared name has to be built from an `ident`
    // placeholder, or the second use redeclares it.
    for declaration in declarations(body) {
        let tokens: Vec<&crate::parser::placeholder::PlaceholderUse> = uses
            .iter()
            .filter(|use_| {
                use_.range.start >= declaration.range.start
                    && use_.range.end <= declaration.range.end
            })
            .collect();
        if tokens.is_empty() {
            return Err(BodyError::FixedNameDeclaration {
                owner: described,
                name: name.to_string(),
                declared: declaration.name.clone(),
                offset: base + declaration.range.start,
            });
        }
        for token in tokens {
            let kind = params
                .iter()
                .find(|p| p.name == token.name)
                .map(|p| p.kind)
                .expect("every use was checked against the header above");
            if kind != PlaceholderKind::Ident {
                return Err(BodyError::DeclarationFromNonIdent {
                    owner: described,
                    name: name.to_string(),
                    declared: declaration.name.clone(),
                    param: token.name.clone(),
                    kind,
                    offset: base + declaration.range.start,
                });
            }
        }
    }

    // References: with fixed-name declarations ruled out, the only legal
    // value references left are the body's own loop variables.
    let scanned = blank_ranges(body, uses.iter().map(|u| u.range.clone()));
    let loop_vars: Vec<String> = loop_variables(&scanned)
        .into_iter()
        .map(|d| d.name)
        .collect();
    for use_ in identifier_uses(&scanned) {
        if !use_.is_value_reference() || loop_vars.contains(&use_.name) {
            continue;
        }
        return Err(BodyError::CapturedVariable {
            owner: described,
            name: name.to_string(),
            reference: use_.name.clone(),
            offset: base + use_.range.start,
        });
    }

    Ok(used)
}

/// Replace each range with spaces, keeping every other byte where it is.
pub fn blank_ranges(text: &str, ranges: impl Iterator<Item = Range<usize>>) -> String {
    let mut out = text.to_string();
    for range in ranges {
        if range.end <= out.len() {
            out.replace_range(range.clone(), &" ".repeat(range.len()));
        }
    }
    out
}

/// Offset of `keyword` in `text` as a whole word, outside comments.
pub fn find_keyword(text: &str, keyword: &str) -> Option<usize> {
    let mask = CodeMask::new(text);
    let bytes = text.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(keyword) {
        let at = from + rel;
        let end = at + keyword.len();
        from = end;
        if mask.is_real(at) && (end == bytes.len() || !is_ident_char(bytes[end])) {
            return Some(at);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(name: &str) -> PlaceholderDecl {
        PlaceholderDecl {
            name: name.to_string(),
            kind: PlaceholderKind::Ident,
            each: false,
        }
    }

    fn expr(name: &str) -> PlaceholderDecl {
        PlaceholderDecl {
            name: name.to_string(),
            kind: PlaceholderKind::Expr,
            each: false,
        }
    }

    fn check_macro(params: &[PlaceholderDecl], body: &str) -> Result<Vec<String>, BodyError> {
        check(BodyOwner::Macro, "m", params, body, 0)
    }

    #[test]
    fn a_well_formed_body_passes_and_reports_what_it_used() {
        let used = check_macro(&[ident("p"), expr("dist")], "\n    $p ~ $dist;\n").unwrap();
        assert_eq!(used, vec!["p", "dist"]);
    }

    #[test]
    fn a_declared_but_unused_placeholder_is_simply_absent_from_the_result() {
        let used = check_macro(&[ident("p"), expr("spare")], "\n    $p ~ std_normal();\n").unwrap();
        assert_eq!(used, vec!["p"]);
    }

    #[test]
    fn a_placeholder_the_header_does_not_declare_is_rejected() {
        let err = check_macro(&[ident("p")], "$p ~ normal($mu, 1);").unwrap_err();
        assert!(
            matches!(err, BodyError::UndeclaredPlaceholder { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("macro `m`"), "{err}");
    }

    #[test]
    fn a_fixed_name_declaration_is_rejected() {
        let err = check_macro(&[ident("p")], "real tmp = 1;\nreal $p = tmp;").unwrap_err();
        assert!(
            matches!(err, BodyError::FixedNameDeclaration { .. }),
            "{err:?}"
        );
        assert!(err.help().contains("collide"), "{}", err.help());
    }

    #[test]
    fn a_declaration_named_from_an_expr_placeholder_is_rejected() {
        let err = check_macro(&[expr("p")], "real ${p}_z;").unwrap_err();
        assert!(
            matches!(err, BodyError::DeclarationFromNonIdent { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_declaration_named_from_an_ident_placeholder_is_fine() {
        assert!(check_macro(&[ident("p")], "real ${p}_z = 0;").is_ok());
    }

    #[test]
    fn reaching_for_an_outside_variable_is_rejected() {
        let err = check_macro(&[ident("p")], "$p ~ normal(mu, 1);").unwrap_err();
        assert!(matches!(err, BodyError::CapturedVariable { .. }), "{err:?}");
        assert!(err.to_string().contains("`mu`"), "{err}");
        assert!(err.help().contains("as a placeholder"), "{}", err.help());
    }

    #[test]
    fn calling_a_function_is_not_capture() {
        assert!(check_macro(&[ident("p")], "$p ~ std_normal();").is_ok());
    }

    #[test]
    fn a_loop_variable_is_a_legal_reference() {
        let used = check_macro(
            &[ident("p"), expr("N")],
            "for (i in 1:$N) {\n  ${p}_z[i] ~ std_normal();\n}",
        )
        .expect("`i` is declared by the loop header");
        assert_eq!(used, vec!["N", "p"]);
    }

    #[test]
    fn a_concatenated_placeholders_suffix_is_blanked_with_it() {
        // `${p}_z` is one token: if only `${p}` were blanked, the
        // leftover `_z` would look like a captured variable.
        assert!(check_macro(&[ident("p")], "${p}_z ~ std_normal();").is_ok());
    }

    #[test]
    fn every_nesting_keyword_is_rejected() {
        for keyword in [
            "@use other::t(x)",
            "@expand other::m(x)",
            "@template t($n: ident) { }",
        ] {
            let body = format!("  {keyword};\n");
            let err = check_macro(&[ident("p")], &body).unwrap_err();
            assert!(
                matches!(err, BodyError::Nested { .. }),
                "{keyword}: {err:?}"
            );
        }
    }

    #[test]
    fn the_owner_appears_in_every_message() {
        let template = check(BodyOwner::Template, "t", &[ident("p")], "real tmp;", 0).unwrap_err();
        assert!(template.to_string().contains("template `t`"), "{template}");
        let macro_ = check(BodyOwner::Macro, "m", &[ident("p")], "real tmp;", 0).unwrap_err();
        assert!(macro_.to_string().contains("macro `m`"), "{macro_}");
    }

    #[test]
    fn offsets_are_reported_relative_to_the_file() {
        let err = check(BodyOwner::Macro, "m", &[ident("p")], "real tmp;", 100).unwrap_err();
        // `tmp` starts 5 bytes into the body.
        assert_eq!(err.offset(), 105);
    }

    #[test]
    fn checking_is_deterministic() {
        let params = [ident("p"), expr("dist")];
        let body = "$p ~ $dist;";
        assert_eq!(check_macro(&params, body), check_macro(&params, body));
    }
}
