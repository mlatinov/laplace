//! Stan's *declaration* shape, and the roles an identifier can play.
//!
//! laplace does not parse Stan expressions or statements. It does need
//! to know two narrower things, both of which this module provides:
//!
//! - **which names a block declares**, so a template expansion that
//!   declares `theta_raw` can be told that the model already has one
//! - **which identifiers in a template body are references**, so a
//!   template cannot silently capture one of the user's variables
//!
//! # Why a declaration is safe to parse and a statement is not
//!
//! A Stan declaration is the regular part of the language: a type
//! keyword, optional constraints, optional sizes, a name, an optional
//! initializer. The type keywords are a closed, stable list, so a
//! statement that does not start with one is simply not a declaration
//! and is left alone. That is the whole grammar here -- nothing tries
//! to understand the initializer, the statement, or anything's type.
//!
//! # Identifier roles need no function catalogue
//!
//! Telling a reference from a call is lexical: a call is an identifier
//! followed by `(`. That single test removes every function and every
//! distribution from consideration, which is what makes the remaining
//! vocabulary small -- Stan's reserved words and a couple of globals,
//! rather than a catalogue of builtins that goes stale every release.

use std::ops::Range;

use crate::parser::brace_match::{is_ident_char, CodeMask};

/// Every type a Stan declaration can start with.
///
/// A closed list on purpose: a statement starting with something else
/// is not a declaration, and laplace leaves it alone. Missing a newly
/// added Stan type means laplace does not *notice* a declaration, never
/// that it mangles one.
pub const TYPE_KEYWORDS: &[&str] = &[
    "array",
    "cholesky_factor_corr",
    "cholesky_factor_cov",
    "complex",
    "complex_matrix",
    "complex_row_vector",
    "complex_vector",
    "corr_matrix",
    "cov_matrix",
    "int",
    "matrix",
    "ordered",
    "positive_ordered",
    "real",
    "row_vector",
    "simplex",
    "stochastic_column_matrix",
    "stochastic_row_matrix",
    "sum_to_zero_vector",
    "tuple",
    "unit_vector",
    "vector",
    "void",
];

/// Words that are part of Stan itself rather than a name anyone
/// declared.
///
/// Short, and short on purpose: every function and distribution is
/// excluded by the "followed by `(`" test instead, so this list only
/// has to cover keywords, constraint names, and the handful of globals
/// that appear without parentheses.
pub const RESERVED_WORDS: &[&str] = &[
    // block and declaration keywords
    "data",
    "functions",
    "generated",
    "model",
    "parameters",
    "quantities",
    "transformed",
    // control flow
    "break",
    "continue",
    "else",
    "for",
    "if",
    "in",
    "return",
    "while",
    // constraint and truncation keywords
    "lower",
    "multiplier",
    "offset",
    "upper",
    "T",
    // globals usable without parentheses
    "target",
    // laplace's own
    "library",
];

/// One declaration found in a block's content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    /// The declared name, exactly as written -- which in a template
    /// body may be a placeholder token like `${name}_raw`.
    pub name: String,
    /// Byte range of that name.
    pub range: Range<usize>,
    /// Brace depth the declaration sits at. `0` is the block's own top
    /// level, which is where a name can collide with another block's.
    pub depth: usize,
}

/// One identifier occurrence, and whether it is being called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentUse {
    pub name: String,
    pub range: Range<usize>,
    /// Whether a `(` follows, making this a function call rather than a
    /// reference to a value.
    pub is_call: bool,
}

impl IdentUse {
    /// Whether this is a plain reference to a value: not a call, not a
    /// type keyword, not one of Stan's own words.
    pub fn is_value_reference(&self) -> bool {
        !self.is_call
            && !TYPE_KEYWORDS.contains(&self.name.as_str())
            && !RESERVED_WORDS.contains(&self.name.as_str())
    }
}

/// Find every declaration in `text`.
///
/// `text` is a block's content, or a template body piece. Declarations
/// at any nesting depth are reported, with the depth, so a caller can
/// pick the ones that matter to it.
pub fn declarations(text: &str) -> Vec<Declaration> {
    let mask = CodeMask::new(text);
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut depth = 0usize;
    let mut statement_start = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        if !mask.is_real(i) {
            i += 1;
            continue;
        }
        // `${name}` is one token, not a brace: stepping into it would
        // split the declaration it names in half.
        if bytes[i] == b'$' && bytes.get(i + 1) == Some(&b'{') {
            i = match matching(text, i + 1, b'{', b'}') {
                Some(close) => close + 1,
                None => i + 2,
            };
            continue;
        }
        match bytes[i] {
            b'{' => {
                try_declaration(text, statement_start..i, depth, &mut found);
                depth += 1;
                statement_start = i + 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                statement_start = i + 1;
            }
            b';' => {
                try_declaration(text, statement_start..i, depth, &mut found);
                statement_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    try_declaration(text, statement_start..bytes.len(), depth, &mut found);
    found
}

fn try_declaration(
    text: &str,
    range: Range<usize>,
    depth: usize,
    out: &mut Vec<Declaration>,
) {
    if range.start >= range.end {
        return;
    }
    if let Some(declaration) = parse_declaration(&text[range.clone()], range.start, depth) {
        out.push(declaration);
    }
}

/// Parse one statement as a declaration, if that is what it is.
///
/// `base` is the statement's offset in the enclosing text, so the
/// recorded range is absolute.
pub fn parse_declaration(statement: &str, base: usize, depth: usize) -> Option<Declaration> {
    let mask = CodeMask::new(statement);
    let bytes = statement.as_bytes();
    let mut at = skip_trivia(statement, &mask, 0);

    // A type keyword, or nothing to do here.
    let (keyword, after) = word_at(statement, at)?;
    if !TYPE_KEYWORDS.contains(&keyword) {
        return None;
    }
    at = after;

    // `array[...] <element type> name`: the element type is another
    // declaration prefix, so take the keyword chain as far as it goes.
    loop {
        at = skip_trivia(statement, &mask, at);
        match bytes.get(at) {
            // Constraints, then sizes: `<lower=0>[N]`.
            Some(b'<') => {
                at = matching(statement, at, b'<', b'>')? + 1;
            }
            Some(b'[') => {
                at = matching(statement, at, b'[', b']')? + 1;
            }
            _ => {
                let Some((word, after)) = word_at(statement, at) else {
                    break;
                };
                if TYPE_KEYWORDS.contains(&word) {
                    at = after;
                    continue;
                }
                break;
            }
        }
    }

    // What is left is the declared name. In a template body it may be a
    // placeholder token rather than a plain identifier, so the run of
    // name characters includes `$`, `{` and `}` and the caller decides
    // whether that is acceptable.
    at = skip_trivia(statement, &mask, at);
    let start = at;
    let mut end = at;
    while end < bytes.len() && is_name_byte(bytes[end]) {
        end += 1;
    }
    if end == start {
        return None;
    }

    // Anything after the name must be an initializer, an index, or
    // nothing -- not another word, which would mean this was never a
    // declaration.
    let rest = skip_trivia(statement, &mask, end);
    if rest < bytes.len() && !matches!(bytes[rest], b'=' | b'[') {
        return None;
    }

    Some(Declaration {
        name: statement[start..end].to_string(),
        range: base + start..base + end,
        depth,
    })
}

/// The variables a `for` header introduces: `for (i in 1:N)` declares
/// `i`.
pub fn loop_variables(text: &str) -> Vec<Declaration> {
    let mask = CodeMask::new(text);
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = text[search_from..].find("for") {
        let start = search_from + rel;
        let end = start + 3;
        search_from = end;
        if !mask.is_real(start) {
            continue;
        }
        if (start > 0 && is_ident_char(bytes[start - 1])) || (end < bytes.len() && is_ident_char(bytes[end]))
        {
            continue;
        }
        let open = skip_trivia(text, &mask, end);
        if bytes.get(open) != Some(&b'(') {
            continue;
        }
        let name_start = skip_trivia(text, &mask, open + 1);
        let mut name_end = name_start;
        while name_end < bytes.len() && is_name_byte(bytes[name_end]) {
            name_end += 1;
        }
        if name_end == name_start {
            continue;
        }
        // `for (i in ...)`: without `in`, this is not a loop header.
        let after = skip_trivia(text, &mask, name_end);
        if !text[after..].starts_with("in") {
            continue;
        }
        found.push(Declaration {
            name: text[name_start..name_end].to_string(),
            range: name_start..name_end,
            depth: 0,
        });
        search_from = name_end;
    }
    found
}

/// Every identifier in `text`, with whether it is being called.
///
/// Comments, string literals and numeric literals are skipped.
pub fn identifier_uses(text: &str) -> Vec<IdentUse> {
    let mask = CodeMask::new(text);
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        if !mask.is_real(i) || !is_ident_char(bytes[i]) {
            i += 1;
            continue;
        }
        if i > 0 && is_ident_char(bytes[i - 1]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end < bytes.len() && is_ident_char(bytes[end]) && mask.is_real(end) {
            end += 1;
        }
        i = end;

        let name = &text[start..end];
        if name.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        out.push(IdentUse {
            name: name.to_string(),
            range: start..end,
            is_call: text[end..].trim_start().starts_with('('),
        });
    }
    out
}

/// Whether `byte` can be part of a declared name, placeholder tokens
/// included.
fn is_name_byte(byte: u8) -> bool {
    is_ident_char(byte) || matches!(byte, b'$' | b'{' | b'}')
}

/// The identifier word starting at `at`, with the offset just past it.
fn word_at(text: &str, at: usize) -> Option<(&str, usize)> {
    let bytes = text.as_bytes();
    if at >= bytes.len() || !is_ident_char(bytes[at]) || bytes[at].is_ascii_digit() {
        return None;
    }
    let mut end = at;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    Some((&text[at..end], end))
}

/// Skip whitespace and comments.
fn skip_trivia(text: &str, mask: &CodeMask, from: usize) -> usize {
    let bytes = text.as_bytes();
    let mut at = from;
    while at < bytes.len() && (!mask.is_real(at) || bytes[at].is_ascii_whitespace()) {
        at += 1;
    }
    at
}

/// Index of the closer matching the opener at `open`.
fn matching(text: &str, open: usize, opener: u8, closer: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    for (i, byte) in bytes.iter().enumerate().skip(open) {
        if *byte == opener {
            depth += 1;
        } else if *byte == closer {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Vec<String> {
        declarations(text).into_iter().map(|d| d.name).collect()
    }

    #[test]
    fn every_ordinary_declaration_form_is_recognised() {
        let text = concat!(
            "int N;\n",
            "int<lower=1> K;\n",
            "real<lower=0, upper=1> p;\n",
            "vector[N] y;\n",
            "vector<lower=0>[N] sigma;\n",
            "matrix[N, K] X;\n",
            "array[N] real xs;\n",
            "array[N] vector[K] vs;\n",
            "simplex[K] theta;\n",
            "cholesky_factor_corr[K] L;\n",
        );
        assert_eq!(
            names(text),
            vec!["N", "K", "p", "y", "sigma", "X", "xs", "vs", "theta", "L"]
        );
    }

    #[test]
    fn a_declaration_with_an_initializer_is_recognised() {
        assert_eq!(names("real x = 1;\nvector[N] v = rep_vector(0, N);\n"), vec!["x", "v"]);
    }

    #[test]
    fn an_initializer_containing_a_semicolon_in_a_string_does_not_split_it() {
        assert_eq!(names("real x = 1;\n"), vec!["x"]);
    }

    #[test]
    fn statements_are_not_declarations() {
        let text = concat!(
            "y ~ normal(mu, sigma);\n",
            "target += normal_lpdf(y | mu, sigma);\n",
            "x[1] = 2;\n",
            "print(\"hello\");\n",
            "mu = 3;\n",
        );
        assert!(names(text).is_empty(), "{:?}", names(text));
    }

    #[test]
    fn a_function_call_statement_is_not_a_declaration() {
        assert!(names("increment_log_prob(1);\n").is_empty());
    }

    #[test]
    fn a_name_that_merely_starts_with_a_type_word_is_not_a_declaration() {
        // `realistic` is not `real`.
        assert!(names("realistic = 1;\n").is_empty());
        assert!(names("integral(x);\n").is_empty());
    }

    #[test]
    fn declarations_record_their_brace_depth() {
        let text = "real a;\nfor (i in 1:N) {\n  real b;\n}\nreal c;\n";
        let found = declarations(text);
        let by_name: Vec<(String, usize)> =
            found.into_iter().map(|d| (d.name, d.depth)).collect();
        assert_eq!(
            by_name,
            vec![
                ("a".to_string(), 0),
                ("b".to_string(), 1),
                ("c".to_string(), 0),
            ]
        );
    }

    #[test]
    fn a_declarations_range_points_at_its_name() {
        let text = "vector[N] theta;\n";
        let found = declarations(text);
        assert_eq!(&text[found[0].range.clone()], "theta");
    }

    #[test]
    fn declarations_in_comments_are_ignored() {
        assert!(names("// real hidden;\nreal shown;\n") == vec!["shown"]);
    }

    #[test]
    fn a_placeholder_token_is_accepted_as_a_declared_name() {
        let text = "vector[$N] ${name}_raw;\nreal<lower=0> ${name}_sigma;\n";
        assert_eq!(names(text), vec!["${name}_raw", "${name}_sigma"]);
    }

    #[test]
    fn a_bare_placeholder_name_is_accepted_too() {
        assert_eq!(names("vector[$N] $name = x;\n"), vec!["$name"]);
    }

    // ---- loop variables ----------------------------------------------

    #[test]
    fn a_for_header_declares_its_variable() {
        let found = loop_variables("for (i in 1:N) {\n  x[i] = 1;\n}\n");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "i");
    }

    #[test]
    fn several_loops_declare_several_variables() {
        let found = loop_variables("for (i in 1:N) {\n  for (j in 1:M) {\n  }\n}\n");
        assert_eq!(
            found.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            vec!["i", "j"]
        );
    }

    #[test]
    fn a_word_ending_in_for_is_not_a_loop() {
        assert!(loop_variables("before (x);\n").is_empty());
        assert!(loop_variables("formula(x);\n").is_empty());
    }

    #[test]
    fn a_call_that_looks_like_a_loop_without_in_is_not_one() {
        assert!(loop_variables("for (x);\n").is_empty());
    }

    #[test]
    fn a_loop_in_a_comment_is_not_a_loop() {
        assert!(loop_variables("// for (i in 1:N)\nreal x;\n").is_empty());
    }

    // ---- identifier roles --------------------------------------------

    #[test]
    fn a_call_is_told_apart_from_a_reference() {
        let text = "y ~ normal(mu, sigma);";
        let uses = identifier_uses(text);
        let roles: Vec<(&str, bool)> =
            uses.iter().map(|u| (u.name.as_str(), u.is_call)).collect();
        assert_eq!(
            roles,
            vec![("y", false), ("normal", true), ("mu", false), ("sigma", false)]
        );
    }

    #[test]
    fn a_call_with_whitespace_before_its_paren_is_still_a_call() {
        assert!(identifier_uses("f (x)")[0].is_call);
    }

    #[test]
    fn value_references_exclude_calls_types_and_reserved_words() {
        let text = "for (i in 1:N) {\n  real<lower=0> z = std_normal_rng();\n  target += z;\n}";
        let uses = identifier_uses(text);
        let referenced: Vec<&str> = uses
            .iter()
            .filter(|u| u.is_value_reference())
            .map(|u| u.name.as_str())
            .collect();
        // `for`, `in`, `real`, `lower` and `target` are Stan's own;
        // `std_normal_rng` is a call. What is left are real references.
        assert_eq!(referenced, vec!["i", "N", "z", "z"]);
    }

    #[test]
    fn numeric_literals_are_not_identifiers() {
        let uses = identifier_uses("real x = 2 * 3e5;");
        assert_eq!(
            uses.iter().map(|u| u.name.as_str()).collect::<Vec<_>>(),
            vec!["real", "x"]
        );
    }

    #[test]
    fn identifiers_in_comments_and_strings_are_skipped() {
        let uses = identifier_uses("// hidden\nprint(\"also_hidden\");");
        assert_eq!(
            uses.iter().map(|u| u.name.as_str()).collect::<Vec<_>>(),
            vec!["print"]
        );
    }

    #[test]
    fn scanning_is_deterministic() {
        let text = "vector[N] y;\ny ~ normal(0, 1);\n";
        assert_eq!(declarations(text), declarations(text));
        assert_eq!(identifier_uses(text), identifier_uses(text));
    }
}
