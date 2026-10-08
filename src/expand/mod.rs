//! The substitution engine shared by templates and macros.
//!
//! Templates (`@use`) and statement macros (`@expand`) differ only in
//! *where* their body ends up: a template is cut into pieces that go
//! into several Stan blocks, a macro expands in place inside one. The
//! work either way is the same, and it lives here:
//!
//! - checking a call site's arguments against the declared placeholder
//!   kinds ([`check_argument`])
//! - typed substitution into a body ([`substitute`])
//! - collision detection against names that already exist
//!   ([`find_collision`])
//! - provenance wrapping, so a reader of the `.stan` file can see which
//!   `@use` line produced a piece ([`wrap`])
//! - re-indenting a piece to the block it is going into ([`reindent`])
//!
//! # Expressions are validated, not parsed
//!
//! An `expr` placeholder exists to carry a whole expression -- a prior,
//! a linear predictor -- through a template untouched. laplace never
//! needs to understand it. It needs exactly two guarantees, and gets
//! both without a Stan expression grammar:
//!
//! - **intact:** [`validate_expression`] rejects anything that is not
//!   one self-contained expression (unbalanced brackets, an injected
//!   statement, a sampling statement).
//! - **unchanged in meaning:** [`parenthesize`] wraps the value unless
//!   it is already a single primary expression, so surrounding
//!   operators cannot rebind it.
//!
//! That second rule is forced by what the output has to look like.
//! `mu + theta` substituted into `lognormal($mu, sigma)` must become
//! `lognormal((mu + theta), sigma)`, while `normal(0, 1)` substituted
//! into `$p ~ $dist;` must become `alpha ~ normal(0, 1);` -- *not*
//! `alpha ~ (normal(0, 1));`, which Stan rejects, because a `~`
//! right-hand side has to be a distribution rather than a parenthesized
//! expression. Parenthesizing only non-primary values produces both.

pub mod blocks;
pub mod macros;

use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::placeholder::{
    find_uses, is_usable_identifier, PlaceholderDecl, PlaceholderError, PlaceholderKind,
};

/// A value bound to a placeholder at a call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub kind: PlaceholderKind,
    /// The argument text as the caller wrote it, trimmed.
    pub value: String,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ExpandError {
    #[error("`${name}` is used but never declared")]
    UndeclaredPlaceholder { name: String, offset: usize },

    #[error(
        "`${name}` is an `{kind}` placeholder, so it cannot be joined to other identifier \
         characters"
    )]
    NotConcatenable {
        name: String,
        kind: PlaceholderKind,
        offset: usize,
    },

    #[error("substituting `${name}` would build `{built}`, which is not a usable name")]
    UnusableName {
        name: String,
        built: String,
        offset: usize,
    },

    #[error("`{argument}` is not a valid identifier, which `${name}` requires")]
    NotAnIdentifier { name: String, argument: String },

    #[error("`{argument}` is not a single expression, which `${name}` requires: {reason}")]
    NotAnExpression {
        name: String,
        argument: String,
        reason: &'static str,
    },

    #[error(transparent)]
    Placeholder(#[from] PlaceholderError),
}

impl ExpandError {
    pub fn offset(&self) -> usize {
        match self {
            ExpandError::UndeclaredPlaceholder { offset, .. }
            | ExpandError::NotConcatenable { offset, .. }
            | ExpandError::UnusableName { offset, .. } => *offset,
            ExpandError::Placeholder(inner) => inner.offset(),
            ExpandError::NotAnIdentifier { .. } | ExpandError::NotAnExpression { .. } => 0,
        }
    }

    pub fn help(&self) -> String {
        match self {
            ExpandError::UndeclaredPlaceholder { name, .. } => {
                format!("declare it in the header, as `${name}: ident` or `${name}: expr`")
            }
            ExpandError::NotConcatenable { name, .. } => format!(
                "only an `ident` placeholder can build a new name; declare `${name}: ident`, or \
                 stop joining it to other characters"
            ),
            ExpandError::UnusableName { .. } => {
                "a name must start with a letter or `_`, contain only letters, digits and `_`, \
                 and may not contain `__`"
                    .to_string()
            }
            ExpandError::NotAnIdentifier { .. } => {
                "pass a bare name here -- this slot becomes part of a declared variable's name"
                    .to_string()
            }
            ExpandError::NotAnExpression { .. } => {
                "pass one self-contained expression, with no statements and no `;`".to_string()
            }
            ExpandError::Placeholder(inner) => inner.help(),
        }
    }
}

/// Check one call-site argument against the placeholder it is being
/// bound to.
pub fn check_argument(decl: &PlaceholderDecl, argument: &str) -> Result<Binding, ExpandError> {
    let value = argument.trim().to_string();
    match decl.kind {
        PlaceholderKind::Ident => {
            if !is_usable_identifier(&value) {
                return Err(ExpandError::NotAnIdentifier {
                    name: decl.name.clone(),
                    argument: value,
                });
            }
        }
        PlaceholderKind::Expr => {
            if let Err(reason) = validate_expression(&value) {
                return Err(ExpandError::NotAnExpression {
                    name: decl.name.clone(),
                    argument: value,
                    reason,
                });
            }
        }
    }
    Ok(Binding {
        kind: decl.kind,
        value,
    })
}

/// Substitute every placeholder in `body`.
///
/// `bindings` maps placeholder name to the value bound to it. An
/// `ident` value is pasted as a name (and may be joined to surrounding
/// identifier characters); an `expr` value is parenthesized unless it is
/// already a single primary expression.
pub fn substitute(body: &str, bindings: &[(String, Binding)]) -> Result<String, ExpandError> {
    let uses = find_uses(body)?;
    let mut out = String::with_capacity(body.len());
    let mut cursor = 0usize;

    for use_ in &uses {
        let binding = bindings
            .iter()
            .find(|(name, _)| *name == use_.name)
            .map(|(_, binding)| binding)
            .ok_or_else(|| ExpandError::UndeclaredPlaceholder {
                name: use_.name.clone(),
                offset: use_.range.start,
            })?;

        if use_.is_concatenated() && !binding.kind.is_concatenable() {
            return Err(ExpandError::NotConcatenable {
                name: use_.name.clone(),
                kind: binding.kind,
                offset: use_.range.start,
            });
        }

        let replacement = match binding.kind {
            PlaceholderKind::Ident => {
                let built = use_.substituted(&binding.value);
                if !is_usable_identifier(&built) {
                    return Err(ExpandError::UnusableName {
                        name: use_.name.clone(),
                        built,
                        offset: use_.range.start,
                    });
                }
                built
            }
            PlaceholderKind::Expr => parenthesize(&binding.value),
        };

        out.push_str(&body[cursor..use_.range.start]);
        out.push_str(&replacement);
        cursor = use_.range.end;
    }
    out.push_str(&body[cursor..]);
    Ok(out)
}

/// Whether `text` is one self-contained Stan expression, as far as
/// laplace needs to care.
///
/// Deliberately a validator and not a parser: it rejects the shapes
/// that would let an `expr` argument smuggle statements into a template
/// body, and says nothing about types, operators or semantics.
pub fn validate_expression(text: &str) -> Result<(), &'static str> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("it is empty");
    }

    let mask = CodeMask::new(trimmed);
    let mut stack: Vec<u8> = Vec::new();
    for (i, byte) in trimmed.bytes().enumerate() {
        if !mask.is_real(i) {
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' => stack.push(byte),
            b')' | b']' | b'}' => {
                let expected = match byte {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                if stack.pop() != Some(expected) {
                    return Err("its brackets are not balanced");
                }
            }
            b';' => return Err("it contains `;`, so it is a statement rather than an expression"),
            b'~' => {
                return Err("it contains `~`, so it is a sampling statement rather than an \
                            expression")
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("its brackets are not balanced");
    }
    Ok(())
}

/// Wrap an expression in parentheses unless it is already a single
/// primary expression.
///
/// See the module docs: a primary value has to stay unwrapped so that a
/// distribution passed as an `expr` still works on the right of a `~`.
pub fn parenthesize(expr: &str) -> String {
    let trimmed = expr.trim();
    if is_primary(trimmed) {
        trimmed.to_string()
    } else {
        format!("({trimmed})")
    }
}

/// Whether `expr` is a single primary expression: a name, a literal, or
/// one of those followed only by call and index groups -- and so cannot
/// be rebound by any surrounding operator.
fn is_primary(expr: &str) -> bool {
    let bytes = expr.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut at = 0usize;

    // An optional leading name or number.
    while at < bytes.len() && (is_ident_char(bytes[at]) || bytes[at] == b'.') {
        at += 1;
    }

    // Then nothing but call and index groups, to the very end. Any
    // operator, comma or space at this level means it is not primary.
    while at < bytes.len() {
        let closer = match bytes[at] {
            b'(' => b')',
            b'[' => b']',
            b'{' => b'}',
            _ => return false,
        };
        let opener = bytes[at];
        let mut depth = 0i32;
        let mut i = at;
        loop {
            if i >= bytes.len() {
                return false;
            }
            if bytes[i] == opener {
                depth += 1;
            } else if bytes[i] == closer {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            i += 1;
        }
        at = i + 1;
    }
    true
}

/// A name that exists, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declared {
    pub name: String,
    /// Rendered description of where it was declared, for the message.
    pub source: String,
}

/// The first name that `incoming` would declare twice, or that already
/// exists in `existing`.
pub fn find_collision(existing: &[Declared], incoming: &[Declared]) -> Option<(Declared, Declared)> {
    for (index, candidate) in incoming.iter().enumerate() {
        if let Some(earlier) = existing.iter().find(|d| d.name == candidate.name) {
            return Some((earlier.clone(), candidate.clone()));
        }
        if let Some(earlier) = incoming[..index]
            .iter()
            .find(|d| d.name == candidate.name)
        {
            return Some((earlier.clone(), candidate.clone()));
        }
    }
    None
}

/// Wrap a generated piece in provenance comments.
///
/// `label` is how the call site read (`@use stats::ncp(theta, K)`) and
/// `short` is the part worth repeating on the closing line.
pub fn wrap(piece: &str, label: &str, short: &str, location: &str, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let mut out = String::new();
    out.push_str(&format!("{pad}// begin {label} -- {location}\n"));
    out.push_str(piece);
    if !piece.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!("{pad}// end {short}\n"));
    out
}

/// Re-indent a piece to sit at `indent` columns.
///
/// The piece comes out of a template body, where it was indented to
/// wherever the author wrote it; it goes into a block in someone else's
/// file. The common leading whitespace is removed and `indent` columns
/// are added, so the result lines up with its new surroundings.
pub fn reindent(piece: &str, indent: usize) -> String {
    let lines: Vec<&str> = piece.lines().collect();
    let common = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    let pad = " ".repeat(indent);
    let mut out = String::with_capacity(piece.len());
    for line in &lines {
        if line.trim().is_empty() {
            out.push('\n');
            continue;
        }
        out.push_str(&pad);
        out.push_str(&line[common.min(line.len() - line.trim_start().len())..]);
        out.push('\n');
    }
    out
}

/// The byte ranges of every placeholder token in `body`, so a caller
/// can blank them before scanning the body as Stan.
pub fn placeholder_ranges(body: &str) -> Result<Vec<Range<usize>>, PlaceholderError> {
    Ok(find_uses(body)?.into_iter().map(|u| u.range).collect())
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

    fn bind(pairs: &[(&str, PlaceholderKind, &str)]) -> Vec<(String, Binding)> {
        pairs
            .iter()
            .map(|(name, kind, value)| {
                (
                    name.to_string(),
                    Binding {
                        kind: *kind,
                        value: value.to_string(),
                    },
                )
            })
            .collect()
    }

    // ---- argument checking -------------------------------------------

    #[test]
    fn an_identifier_argument_is_accepted_for_an_ident_placeholder() {
        assert_eq!(
            check_argument(&ident("name"), " theta ").unwrap().value,
            "theta"
        );
    }

    #[test]
    fn a_non_identifier_argument_is_refused_for_an_ident_placeholder() {
        let err = check_argument(&ident("name"), "mu + 1").unwrap_err();
        assert!(matches!(err, ExpandError::NotAnIdentifier { .. }), "{err:?}");
        assert!(err.help().contains("bare name"), "{}", err.help());
    }

    #[test]
    fn an_ident_argument_may_not_contain_a_reserved_double_underscore() {
        assert!(check_argument(&ident("name"), "a__b").is_err());
    }

    #[test]
    fn an_expression_argument_is_accepted_for_an_expr_placeholder() {
        assert_eq!(
            check_argument(&expr("mu"), "mu + theta").unwrap().value,
            "mu + theta"
        );
        assert!(check_argument(&expr("d"), "normal(0, 1)").is_ok());
        assert!(check_argument(&expr("n"), "K").is_ok());
    }

    #[test]
    fn a_statement_is_refused_for_an_expr_placeholder() {
        let err = check_argument(&expr("mu"), "real x = 1;").unwrap_err();
        assert!(matches!(err, ExpandError::NotAnExpression { .. }), "{err:?}");
        assert!(err.to_string().contains("`;`"), "{err}");
    }

    #[test]
    fn a_sampling_statement_is_refused_for_an_expr_placeholder() {
        let err = check_argument(&expr("d"), "y ~ normal(0, 1)").unwrap_err();
        assert!(err.to_string().contains("`~`"), "{err}");
    }

    #[test]
    fn unbalanced_brackets_are_refused_for_an_expr_placeholder() {
        assert!(check_argument(&expr("e"), "f(a").is_err());
        assert!(check_argument(&expr("e"), "a)").is_err());
        assert!(check_argument(&expr("e"), "x[1").is_err());
    }

    #[test]
    fn an_empty_argument_is_refused() {
        assert!(check_argument(&expr("e"), "  ").is_err());
        assert!(check_argument(&ident("i"), "").is_err());
    }

    #[test]
    fn an_array_literal_is_a_valid_expression() {
        assert!(validate_expression("{1, 2, 3}").is_ok());
    }

    // ---- parenthesization --------------------------------------------

    #[test]
    fn a_compound_expression_is_parenthesized() {
        assert_eq!(parenthesize("mu + theta"), "(mu + theta)");
        assert_eq!(parenthesize("2 * K"), "(2 * K)");
        assert_eq!(parenthesize("-x"), "(-x)");
        assert_eq!(parenthesize("a ? b : c"), "(a ? b : c)");
    }

    #[test]
    fn a_single_primary_expression_is_left_bare() {
        // This is what keeps `$p ~ $dist;` with `normal(0, 1)` valid.
        assert_eq!(parenthesize("normal(0, 1)"), "normal(0, 1)");
        assert_eq!(parenthesize("sigma"), "sigma");
        assert_eq!(parenthesize("y[1]"), "y[1]");
        assert_eq!(parenthesize("5"), "5");
        assert_eq!(parenthesize("2.0"), "2.0");
        assert_eq!(parenthesize("{1, 2}"), "{1, 2}");
    }

    #[test]
    fn an_already_parenthesized_expression_is_not_wrapped_twice() {
        assert_eq!(parenthesize("(mu + theta)"), "(mu + theta)");
    }

    #[test]
    fn a_nested_call_is_still_primary() {
        assert_eq!(parenthesize("f(g(a), h(b))"), "f(g(a), h(b))");
    }

    #[test]
    fn two_primaries_side_by_side_are_not_primary() {
        assert_eq!(parenthesize("f(a) + g(b)"), "(f(a) + g(b))");
    }

    // ---- substitution ------------------------------------------------

    #[test]
    fn an_ident_placeholder_substitutes_as_a_name() {
        let body = "vector[$N] $name;";
        let out = substitute(
            body,
            &bind(&[
                ("N", PlaceholderKind::Expr, "K"),
                ("name", PlaceholderKind::Ident, "theta"),
            ]),
        )
        .unwrap();
        assert_eq!(out, "vector[K] theta;");
    }

    #[test]
    fn a_concatenated_ident_builds_a_new_name() {
        let body = "vector[$N] ${name}_raw;\n${name}_raw ~ std_normal();";
        let out = substitute(
            body,
            &bind(&[
                ("N", PlaceholderKind::Expr, "K"),
                ("name", PlaceholderKind::Ident, "theta"),
            ]),
        )
        .unwrap();
        assert_eq!(out, "vector[K] theta_raw;\ntheta_raw ~ std_normal();");
    }

    #[test]
    fn an_expr_placeholder_substitutes_parenthesized_when_it_has_to() {
        let body = "$y ~ lognormal($mu, $sigma);";
        let out = substitute(
            body,
            &bind(&[
                ("y", PlaceholderKind::Ident, "y"),
                ("mu", PlaceholderKind::Expr, "mu + theta"),
                ("sigma", PlaceholderKind::Expr, "sigma"),
            ]),
        )
        .unwrap();
        // Exactly the spec's worked example.
        assert_eq!(out, "y ~ lognormal((mu + theta), sigma);");
    }

    #[test]
    fn a_distribution_passed_as_an_expr_stays_usable_after_a_tilde() {
        let out = substitute(
            "$p ~ $dist;",
            &bind(&[
                ("p", PlaceholderKind::Ident, "alpha"),
                ("dist", PlaceholderKind::Expr, "normal(0, 1)"),
            ]),
        )
        .unwrap();
        assert_eq!(out, "alpha ~ normal(0, 1);");
    }

    #[test]
    fn concatenating_an_expr_placeholder_is_an_error() {
        let err = substitute(
            "real ${mu}_x;",
            &bind(&[("mu", PlaceholderKind::Expr, "a + b")]),
        )
        .unwrap_err();
        assert!(matches!(err, ExpandError::NotConcatenable { .. }), "{err:?}");
        assert!(err.help().contains("ident"), "{}", err.help());
    }

    #[test]
    fn a_use_with_no_binding_is_an_error() {
        let err = substitute("real $nope;", &bind(&[])).unwrap_err();
        assert!(
            matches!(err, ExpandError::UndeclaredPlaceholder { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_concatenation_building_a_reserved_name_is_an_error() {
        let err = substitute(
            "real ${name}_raw;",
            &bind(&[("name", PlaceholderKind::Ident, "theta_")]),
        )
        .unwrap_err();
        assert!(matches!(err, ExpandError::UnusableName { .. }), "{err:?}");
        assert!(err.to_string().contains("theta__raw"), "{err}");
    }

    #[test]
    fn a_body_with_no_placeholders_comes_back_unchanged() {
        let body = "theta_raw ~ std_normal();";
        assert_eq!(substitute(body, &bind(&[])).unwrap(), body);
    }

    #[test]
    fn substitution_is_deterministic() {
        let body = "vector[$N] ${name}_raw;";
        let bindings = bind(&[
            ("N", PlaceholderKind::Expr, "K"),
            ("name", PlaceholderKind::Ident, "theta"),
        ]);
        assert_eq!(
            substitute(body, &bindings).unwrap(),
            substitute(body, &bindings).unwrap()
        );
    }

    // ---- collisions --------------------------------------------------

    fn declared(name: &str, source: &str) -> Declared {
        Declared {
            name: name.to_string(),
            source: source.to_string(),
        }
    }

    #[test]
    fn a_name_that_already_exists_is_a_collision() {
        let existing = vec![declared("theta_raw", "model.laplace:12")];
        let incoming = vec![declared("theta_raw", "@use stats::ncp")];
        let (first, second) = find_collision(&existing, &incoming).unwrap();
        assert_eq!(first.source, "model.laplace:12");
        assert_eq!(second.source, "@use stats::ncp");
    }

    #[test]
    fn two_incoming_pieces_declaring_the_same_name_collide_with_each_other() {
        let incoming = vec![
            declared("theta_raw", "@use a"),
            declared("theta_raw", "@use b"),
        ];
        let (first, second) = find_collision(&[], &incoming).unwrap();
        assert_eq!(first.source, "@use a");
        assert_eq!(second.source, "@use b");
    }

    #[test]
    fn distinct_names_do_not_collide() {
        let existing = vec![declared("mu", "model.laplace:3")];
        let incoming = vec![declared("theta_raw", "@use a"), declared("theta", "@use a")];
        assert_eq!(find_collision(&existing, &incoming), None);
    }

    // ---- provenance and indentation ----------------------------------

    #[test]
    fn a_piece_is_wrapped_in_begin_and_end_comments() {
        let wrapped = wrap(
            "  vector[K] theta_raw;\n",
            "@use stats::ncp(theta, K)",
            "@use stats::ncp",
            "model.laplace:4",
            2,
        );
        assert_eq!(
            wrapped,
            "  // begin @use stats::ncp(theta, K) -- model.laplace:4\n  vector[K] theta_raw;\n  // end @use stats::ncp\n"
        );
    }

    #[test]
    fn reindenting_moves_a_piece_to_its_new_block() {
        let piece = "    vector[K] theta_raw;\n    real<lower=0> theta_sigma;\n";
        assert_eq!(
            reindent(piece, 2),
            "  vector[K] theta_raw;\n  real<lower=0> theta_sigma;\n"
        );
    }

    #[test]
    fn reindenting_keeps_relative_nesting() {
        let piece = "    for (i in 1:N) {\n      x[i] = 1;\n    }\n";
        assert_eq!(
            reindent(piece, 2),
            "  for (i in 1:N) {\n    x[i] = 1;\n  }\n"
        );
    }

    #[test]
    fn reindenting_drops_blank_line_padding() {
        assert_eq!(reindent("  a;\n\n  b;\n", 2), "  a;\n\n  b;\n");
    }

    #[test]
    fn placeholder_ranges_cover_every_token() {
        let body = "vector[$N] ${name}_raw;";
        let ranges = placeholder_ranges(body).unwrap();
        assert_eq!(ranges.len(), 2);
        assert_eq!(&body[ranges[1].clone()], "${name}_raw");
    }
}
