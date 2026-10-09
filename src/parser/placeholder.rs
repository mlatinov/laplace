//! Typed placeholders: `$name` in a template or macro body, declared in
//! its header as `$name: ident`.
//!
//! ```stan
//! pub @template ncp($name: ident, $N: expr) {
//!   parameters {
//!     vector[$N] ${name}_raw;
//!   }
//! }
//! ```
//!
//! Two kinds today. They exist for different reasons, and the
//! difference is what makes substitution safe:
//!
//! - **`ident`** carries a *name*. It is what lets the same template be
//!   used twice in one model without the two expansions colliding, so
//!   every variable a template declares has to be named from one.
//! - **`expr`** carries an *expression* -- a prior, a linear predictor --
//!   through the template untouched. laplace never needs to understand
//!   it, only to keep it intact and not let the surrounding operators
//!   change its meaning.
//!
//! Only `ident` placeholders may be glued to other identifier
//! characters (`${name}_raw`): the result is a new name, which only
//! makes sense for a name.
//!
//! # Scanning, not parsing
//!
//! This module finds placeholder declarations and uses. It does not
//! know what a Stan expression is; see [`crate::expand`] for how a
//! substituted value is validated and parenthesized.

use std::fmt;
use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::identifiers::RESERVED_SEPARATOR;
use crate::parser::types::split_top_level_args;

/// What a placeholder stands for.
///
/// A `type` kind is planned; the enum and every match on it are written
/// so adding one does not reshape anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderKind {
    Ident,
    Expr,
}

impl PlaceholderKind {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "ident" => Some(PlaceholderKind::Ident),
            "expr" => Some(PlaceholderKind::Expr),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PlaceholderKind::Ident => "ident",
            PlaceholderKind::Expr => "expr",
        }
    }

    /// Whether a value of this kind may be glued to other identifier
    /// characters to build a new name.
    pub fn is_concatenable(self) -> bool {
        matches!(self, PlaceholderKind::Ident)
    }

    /// Every kind laplace understands, for an error message.
    pub fn all() -> &'static [&'static str] {
        &["ident", "expr"]
    }
}

impl fmt::Display for PlaceholderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One placeholder declared in a template or macro header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceholderDecl {
    pub name: String,
    pub kind: PlaceholderKind,
    /// Whether the header marked this one as a list (`each $p: ident`).
    /// Always false for templates; macros use it in the next patch.
    pub each: bool,
}

/// One use of a placeholder in a body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceholderUse {
    pub name: String,
    /// The whole token to replace: the `$name` or `${name}` itself plus
    /// any identifier characters glued to either side.
    pub range: Range<usize>,
    /// Literal text inside `range` before the placeholder.
    pub prefix: String,
    /// Literal text inside `range` after the placeholder.
    pub suffix: String,
    /// Whether the header form `${name}` was used.
    pub braced: bool,
}

impl PlaceholderUse {
    /// Whether this use builds a new name out of the substituted value.
    pub fn is_concatenated(&self) -> bool {
        !self.prefix.is_empty() || !self.suffix.is_empty()
    }

    /// The text this use expands to, given the substituted value.
    pub fn substituted(&self, value: &str) -> String {
        format!("{}{}{}", self.prefix, value, self.suffix)
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PlaceholderError {
    #[error("`$` must be followed by a placeholder name, as `$name` or `${{name}}`")]
    Malformed { offset: usize },

    #[error("`${{` is never closed with a matching `}}`")]
    UnclosedBrace { offset: usize },

    #[error(
        "`${name}` is glued to the identifier characters before it; laplace cannot tell where \
         the name starts"
    )]
    Glued { name: String, offset: usize },

    #[error("`{text}` is not a placeholder declaration: expected `$name: {kinds}`")]
    MalformedDecl { text: String, kinds: String },

    #[error("`{kind}` is not a placeholder kind")]
    UnknownKind { kind: String, offset: usize },

    #[error("`${name}` is declared twice")]
    DuplicateDecl { name: String },
}

impl PlaceholderError {
    pub fn offset(&self) -> usize {
        match self {
            PlaceholderError::Malformed { offset }
            | PlaceholderError::UnclosedBrace { offset }
            | PlaceholderError::Glued { offset, .. }
            | PlaceholderError::UnknownKind { offset, .. } => *offset,
            PlaceholderError::MalformedDecl { .. } | PlaceholderError::DuplicateDecl { .. } => 0,
        }
    }

    pub fn help(&self) -> String {
        match self {
            PlaceholderError::Malformed { .. } => {
                "write `$name`, or `${name}` when the name is followed by more identifier \
                 characters"
                    .to_string()
            }
            PlaceholderError::UnclosedBrace { .. } => "add the closing `}`".to_string(),
            PlaceholderError::Glued { name, .. } => {
                format!("write `${{{name}}}` so the name is delimited")
            }
            PlaceholderError::MalformedDecl { kinds, .. } => {
                format!("declare each placeholder with its kind, as `$name: {kinds}`")
            }
            PlaceholderError::UnknownKind { .. } => {
                format!("the kinds are {}", PlaceholderKind::all().join(", "))
            }
            PlaceholderError::DuplicateDecl { name } => {
                format!("remove one of the two `${name}` declarations")
            }
        }
    }
}

/// Parse a header's placeholder list: the text between the parentheses
/// of `@template ncp($name: ident, $N: expr)`.
///
/// `each` is accepted on a declaration (`each $p: ident`) and recorded;
/// whether it is allowed at all is the caller's rule.
pub fn parse_declarations(text: &str) -> Result<Vec<PlaceholderDecl>, PlaceholderError> {
    let mut decls: Vec<PlaceholderDecl> = Vec::new();

    for chunk in split_top_level_args(text) {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let malformed = || PlaceholderError::MalformedDecl {
            text: chunk.to_string(),
            kinds: PlaceholderKind::all().join("` | `"),
        };

        let (each, rest) = match chunk.strip_prefix("each") {
            Some(rest) if rest.starts_with(char::is_whitespace) => (true, rest.trim_start()),
            _ => (false, chunk),
        };
        let rest = rest.strip_prefix('$').ok_or_else(malformed)?;
        let (name, kind) = rest.split_once(':').ok_or_else(malformed)?;
        let (name, kind) = (name.trim(), kind.trim());
        if !is_identifier(name) {
            return Err(malformed());
        }
        let kind = PlaceholderKind::parse(kind).ok_or(PlaceholderError::UnknownKind {
            kind: kind.to_string(),
            offset: 0,
        })?;
        if decls.iter().any(|d| d.name == name) {
            return Err(PlaceholderError::DuplicateDecl {
                name: name.to_string(),
            });
        }
        decls.push(PlaceholderDecl {
            name: name.to_string(),
            kind,
            each,
        });
    }

    Ok(decls)
}

/// Find every placeholder use in `body`, in source order.
///
/// Comments and string literals are skipped: `$name` in a comment is
/// prose, not a placeholder.
pub fn find_uses(body: &str) -> Result<Vec<PlaceholderUse>, PlaceholderError> {
    let mask = CodeMask::new(body);
    let bytes = body.as_bytes();
    let mut uses = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'$' || !mask.is_real(i) {
            i += 1;
            continue;
        }
        let dollar = i;
        let after = dollar + 1;

        let (name, braced, end) = if bytes.get(after) == Some(&b'{') {
            let close = body[after..]
                .find('}')
                .map(|rel| after + rel)
                .ok_or(PlaceholderError::UnclosedBrace { offset: dollar })?;
            let name = body[after + 1..close].trim().to_string();
            if !is_identifier(&name) {
                return Err(PlaceholderError::Malformed { offset: dollar });
            }
            (name, true, close + 1)
        } else {
            let mut end = after;
            while end < bytes.len() && is_ident_char(bytes[end]) {
                end += 1;
            }
            if end == after {
                return Err(PlaceholderError::Malformed { offset: dollar });
            }
            (body[after..end].to_string(), false, end)
        };

        // Identifier characters glued to either side become part of the
        // name the substitution builds.
        let mut token_start = dollar;
        while token_start > 0 && is_ident_char(bytes[token_start - 1]) {
            token_start -= 1;
        }
        if token_start < dollar && !braced {
            return Err(PlaceholderError::Glued {
                name,
                offset: dollar,
            });
        }
        let mut token_end = end;
        while token_end < bytes.len() && is_ident_char(bytes[token_end]) {
            token_end += 1;
        }

        uses.push(PlaceholderUse {
            name,
            prefix: body[token_start..dollar].to_string(),
            suffix: body[end..token_end].to_string(),
            braced,
            range: token_start..token_end,
        });
        i = token_end.max(end);
    }

    Ok(uses)
}

/// Whether `name` can be a Stan identifier built by concatenation.
///
/// `__` is reserved for generated names, so a concatenation that
/// produces one is rejected here rather than silently colliding with a
/// mangled name later.
pub fn is_usable_identifier(name: &str) -> bool {
    is_identifier(name) && !name.contains(RESERVED_SEPARATOR)
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decls(text: &str) -> Vec<PlaceholderDecl> {
        parse_declarations(text).expect("should parse")
    }

    fn uses(body: &str) -> Vec<PlaceholderUse> {
        find_uses(body).expect("should scan")
    }

    #[test]
    fn a_header_declares_names_and_kinds_in_order() {
        let parsed = decls("$name: ident, $N: expr");
        assert_eq!(
            parsed,
            vec![
                PlaceholderDecl {
                    name: "name".to_string(),
                    kind: PlaceholderKind::Ident,
                    each: false,
                },
                PlaceholderDecl {
                    name: "N".to_string(),
                    kind: PlaceholderKind::Expr,
                    each: false,
                },
            ]
        );
    }

    #[test]
    fn an_empty_header_declares_nothing() {
        assert!(decls("").is_empty());
        assert!(decls("   ").is_empty());
    }

    #[test]
    fn extra_whitespace_in_a_declaration_is_fine() {
        assert_eq!(decls("  $name :  ident  ")[0].kind, PlaceholderKind::Ident);
    }

    #[test]
    fn each_is_recorded_for_the_macro_patch_to_use() {
        let parsed = decls("each $p: ident, $dist: expr");
        assert!(parsed[0].each);
        assert!(!parsed[1].each);
    }

    #[test]
    fn a_declaration_without_a_dollar_is_malformed() {
        let err = parse_declarations("name: ident").unwrap_err();
        assert!(
            matches!(err, PlaceholderError::MalformedDecl { .. }),
            "{err:?}"
        );
        assert!(err.help().contains("$name:"), "{}", err.help());
    }

    #[test]
    fn a_declaration_without_a_kind_is_malformed() {
        assert!(matches!(
            parse_declarations("$name").unwrap_err(),
            PlaceholderError::MalformedDecl { .. }
        ));
    }

    #[test]
    fn an_unknown_kind_lists_the_ones_that_exist() {
        let err = parse_declarations("$T: type").unwrap_err();
        assert!(
            matches!(err, PlaceholderError::UnknownKind { .. }),
            "{err:?}"
        );
        assert!(err.help().contains("ident"), "{}", err.help());
        assert!(err.help().contains("expr"), "{}", err.help());
    }

    #[test]
    fn declaring_the_same_placeholder_twice_is_an_error() {
        let err = parse_declarations("$a: ident, $a: expr").unwrap_err();
        assert!(
            matches!(err, PlaceholderError::DuplicateDecl { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_kind_commas_inside_a_call_do_not_split_the_list() {
        // Not valid today, but the splitter must not be the thing that
        // breaks when a future kind takes arguments.
        assert_eq!(decls("$a: ident, $b: expr").len(), 2);
    }

    // ---- uses --------------------------------------------------------

    #[test]
    fn a_bare_use_is_found() {
        let body = "vector[$N] x;";
        let found = uses(body);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "N");
        assert!(!found[0].braced);
        assert!(!found[0].is_concatenated());
        assert_eq!(&body[found[0].range.clone()], "$N");
    }

    #[test]
    fn a_braced_use_is_found() {
        let body = "vector[2] ${name};";
        let found = uses(body);
        assert_eq!(found[0].name, "name");
        assert!(found[0].braced);
        assert!(!found[0].is_concatenated());
    }

    #[test]
    fn a_suffix_concatenation_captures_the_whole_token() {
        let body = "vector[$N] ${name}_raw;";
        let found = uses(body);
        assert_eq!(found.len(), 2);
        let concat = &found[1];
        assert_eq!(concat.name, "name");
        assert!(concat.is_concatenated());
        assert_eq!(concat.suffix, "_raw");
        assert_eq!(&body[concat.range.clone()], "${name}_raw");
        assert_eq!(concat.substituted("theta"), "theta_raw");
    }

    #[test]
    fn a_prefix_concatenation_captures_the_whole_token() {
        let body = "real log_${name};";
        let found = uses(body);
        assert_eq!(found[0].prefix, "log_");
        assert_eq!(found[0].substituted("theta"), "log_theta");
    }

    #[test]
    fn a_bare_use_eats_the_identifier_characters_after_it() {
        // `$name_raw` is the placeholder called `name_raw`, which is why
        // concatenation needs the braced form.
        let found = uses("real $name_raw;");
        assert_eq!(found[0].name, "name_raw");
        assert!(!found[0].is_concatenated());
    }

    #[test]
    fn a_bare_use_glued_on_the_left_is_an_error() {
        let err = find_uses("real log_$name;").unwrap_err();
        assert!(matches!(err, PlaceholderError::Glued { .. }), "{err:?}");
        assert!(err.help().contains("${name}"), "{}", err.help());
    }

    #[test]
    fn several_uses_of_one_placeholder_are_all_found() {
        let found = uses("${name}_sigma * ${name}_raw");
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|u| u.name == "name"));
        assert_eq!(found[0].suffix, "_sigma");
        assert_eq!(found[1].suffix, "_raw");
    }

    #[test]
    fn uses_in_comments_and_strings_are_ignored() {
        assert!(uses("// $name is the placeholder\nreal x;").is_empty());
        assert!(uses("print(\"$name\");").is_empty());
    }

    #[test]
    fn a_dollar_with_nothing_after_it_is_malformed() {
        assert!(matches!(
            find_uses("real x = $;").unwrap_err(),
            PlaceholderError::Malformed { .. }
        ));
    }

    #[test]
    fn an_unclosed_brace_is_reported_as_such() {
        let err = find_uses("real ${name;").unwrap_err();
        assert!(
            matches!(err, PlaceholderError::UnclosedBrace { .. }),
            "{err:?}"
        );
        assert!(err.help().contains("closing `}`"), "{}", err.help());
    }

    #[test]
    fn a_braced_use_with_a_non_identifier_inside_is_malformed() {
        assert!(find_uses("real ${a + b};").is_err());
    }

    #[test]
    fn a_body_with_no_placeholders_yields_none() {
        assert!(uses("theta_raw ~ std_normal();").is_empty());
    }

    // ---- concatenation results ---------------------------------------

    #[test]
    fn a_concatenation_must_produce_a_usable_identifier() {
        assert!(is_usable_identifier("theta_raw"));
        assert!(is_usable_identifier("_leading"));
        assert!(!is_usable_identifier("2theta"), "cannot start with a digit");
        assert!(!is_usable_identifier(""), "cannot be empty");
        assert!(!is_usable_identifier("a-b"), "not identifier characters");
    }

    #[test]
    fn a_concatenation_may_not_produce_a_reserved_double_underscore() {
        // `${name}_raw` with name `theta_` would build `theta__raw`,
        // which belongs to generated names.
        assert!(!is_usable_identifier("theta__raw"));
    }

    #[test]
    fn only_ident_placeholders_are_concatenable() {
        assert!(PlaceholderKind::Ident.is_concatenable());
        assert!(!PlaceholderKind::Expr.is_concatenable());
    }

    #[test]
    fn scanning_is_deterministic() {
        let body = "vector[$N] ${name}_raw;\n${name}_raw ~ std_normal();";
        assert_eq!(uses(body), uses(body));
    }
}
