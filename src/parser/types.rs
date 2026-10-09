//! Stan types as they appear in a *function signature*, including the
//! optional size annotation laplace adds and strips.
//!
//! Stan signatures carry no sizes: `vector to_pair(real x)` says nothing
//! about how long the result is. That is fine for Stan, which only needs
//! the type to resolve a call, and useless for laplace, which has to
//! write a *sized* local declaration when it specializes a higher-order
//! function (`@wait(f) row = f(x);` has to become `vector[2] row = ...`).
//!
//! So laplace accepts a size on a signature type:
//!
//! ```stan
//! vector[2] to_pair(real x) { ... }        // laplace source
//! vector to_pair(real x) { ... }           // Stan output
//! ```
//!
//! This is deliberately **one** grammar rule, shared by two features: the
//! sized *return* types this patch needs, and the planned signature size
//! binding (`real foo(vector[N] x, vector[N] y)`), which will read sizes
//! off *parameters* instead. Only return types are acted on today;
//! parameter sizes parse and are left alone.
//!
//! This is not a Stan type checker. It splits a type into "the part Stan
//! keeps" and "the sizes laplace reads", and classifies the base type
//! well enough to say whether `.size`, `.rows` or `.cols` makes sense.

use std::fmt;

/// Which family of Stan type this is. Enough to check a `@wait`
/// accessor and to know how many sizes a type takes; deliberately not a
/// full model of Stan's type lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeCategory {
    Int,
    Real,
    Complex,
    Vector,
    RowVector,
    Matrix,
    /// `array[] real`, `array[,] vector`, ... Out of scope as a
    /// functional parameter's return type in this patch.
    Array,
    /// Anything this scanner does not recognise. Passed through
    /// untouched rather than rejected -- laplace does not own Stan's
    /// type list and a new Stan type must not break the compiler.
    Other,
}

impl TypeCategory {
    fn of(base: &str) -> Self {
        match base {
            "int" => TypeCategory::Int,
            "real" => TypeCategory::Real,
            "complex" => TypeCategory::Complex,
            "vector" => TypeCategory::Vector,
            "row_vector" => TypeCategory::RowVector,
            "matrix" => TypeCategory::Matrix,
            "array" => TypeCategory::Array,
            _ => TypeCategory::Other,
        }
    }

    /// How many sizes a declaration of this type needs: `vector[N]`,
    /// `matrix[R, C]`, `real`. `None` for a category laplace does not
    /// size itself.
    pub fn size_arity(self) -> Option<usize> {
        match self {
            TypeCategory::Int | TypeCategory::Real | TypeCategory::Complex => Some(0),
            TypeCategory::Vector | TypeCategory::RowVector => Some(1),
            TypeCategory::Matrix => Some(2),
            TypeCategory::Array | TypeCategory::Other => None,
        }
    }

    /// The `@wait(f).<accessor>` spellings this category answers to, in
    /// the order they index into the type's sizes.
    pub fn accessors(self) -> &'static [&'static str] {
        match self {
            TypeCategory::Vector | TypeCategory::RowVector => &["size"],
            TypeCategory::Matrix => &["rows", "cols"],
            _ => &[],
        }
    }

    /// How this category reads in an error message.
    pub fn describe(self) -> &'static str {
        match self {
            TypeCategory::Int => "int",
            TypeCategory::Real => "real",
            TypeCategory::Complex => "complex",
            TypeCategory::Vector => "vector",
            TypeCategory::RowVector => "row_vector",
            TypeCategory::Matrix => "matrix",
            TypeCategory::Array => "array",
            TypeCategory::Other => "unrecognised type",
        }
    }
}

/// A type as written in a signature, split into the part Stan keeps and
/// the sizes laplace reads off it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StanType {
    /// The type with its size annotation removed -- what goes into the
    /// generated `.stan`. `vector[2]` -> `vector`, `real` -> `real`.
    pub bare: String,
    /// The size expressions between the brackets, verbatim and
    /// individually trimmed. Empty when there was no annotation, which
    /// is the normal case.
    pub sizes: Vec<String>,
    pub category: TypeCategory,
}

impl StanType {
    /// Whether the type carried a size annotation.
    pub fn is_sized(&self) -> bool {
        !self.sizes.is_empty()
    }

    /// The type as a *sized* declaration: `vector` + `["2"]` ->
    /// `vector[2]`. Unsized types render as themselves, which is what a
    /// `real`/`int` return needs.
    pub fn sized(&self) -> String {
        if self.sizes.is_empty() {
            self.bare.clone()
        } else {
            format!("{}[{}]", self.bare, self.sizes.join(", "))
        }
    }

    /// Whether `self` and `other` are the same type as far as *Stan*
    /// cares: sizes are laplace's business and never part of a Stan
    /// signature, so they are not compared.
    pub fn matches(&self, other: &StanType) -> bool {
        normalize_whitespace(&self.bare) == normalize_whitespace(&other.bare)
    }
}

impl fmt::Display for StanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.sized())
    }
}

/// Parse a signature type: a base type, optionally followed by a size
/// annotation in brackets, optionally followed by more type text (an
/// array's element type).
///
/// Only the *first* bracket group directly after the base type keyword is
/// read as a size annotation, which is where Stan puts an array's
/// dimensions and where laplace puts a size.
pub fn parse_type(text: &str) -> StanType {
    let text = text.trim();
    let Some(open) = text.find('[') else {
        return StanType {
            bare: normalize_whitespace(text),
            sizes: Vec::new(),
            category: TypeCategory::of(base_word(text)),
        };
    };
    let Some(close) = matching_bracket(text, open) else {
        // Unbalanced: not something to guess at. Hand it back whole and
        // let stanc be the one to complain.
        return StanType {
            bare: normalize_whitespace(text),
            sizes: Vec::new(),
            category: TypeCategory::of(base_word(text)),
        };
    };

    let before = &text[..open];
    let inside = &text[open + 1..close];
    let after = &text[close + 1..];

    let sizes: Vec<String> = split_top_level_args(inside)
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // `array[] real` and `array[,] real` keep their brackets: the commas
    // are the array's *rank*, not a size, and Stan needs them.
    let category = TypeCategory::of(base_word(text));
    if category == TypeCategory::Array {
        return StanType {
            bare: normalize_whitespace(text),
            sizes,
            category,
        };
    }

    let bare = normalize_whitespace(&format!("{before}{after}"));
    StanType {
        bare,
        sizes,
        category,
    }
}

/// The first identifier-ish word of a type, which decides its category.
/// A `data` qualifier is skipped: Stan allows it on function arguments
/// and it says nothing about the type.
fn base_word(text: &str) -> &str {
    let mut words = text
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty());
    match words.next() {
        Some("data") => words.next().unwrap_or(""),
        Some(word) => word,
        None => "",
    }
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Index of the `]` matching the `[` at `open`.
pub fn matching_bracket(text: &str, open: usize) -> Option<usize> {
    matching_delimiter(text, open, b'[', b']')
}

/// Index of the `)` matching the `(` at `open`.
pub fn matching_paren(text: &str, open: usize) -> Option<usize> {
    matching_delimiter(text, open, b'(', b')')
}

/// Index of the closer matching the opener at `open`, counting nesting.
fn matching_delimiter(text: &str, open: usize, opener: u8, closer: u8) -> Option<usize> {
    let mut depth = 0i32;
    for (i, byte) in text.bytes().enumerate().skip(open) {
        if byte == opener {
            depth += 1;
        } else if byte == closer {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Split an argument list on its top-level commas, ignoring commas nested
/// in brackets, parentheses or braces.
///
/// Parentheses matter now that a parameter's type can be
/// `func(real, int) -> real`, and braces matter for an array literal
/// argument like `{1, 2}`. Always returns at least one element, so an
/// empty input yields one empty string -- callers filter.
pub fn split_top_level_args(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    let mut in_string = false;

    for c in text.chars() {
        if in_string {
            current.push(c);
            if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                current.push(c);
            }
            '[' | '(' | '{' => {
                depth += 1;
                current.push(c);
            }
            ']' | ')' | '}' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => out.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    out.push(current);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unsized_type_parses_as_itself() {
        let t = parse_type("vector");
        assert_eq!(t.bare, "vector");
        assert!(t.sizes.is_empty());
        assert!(!t.is_sized());
        assert_eq!(t.category, TypeCategory::Vector);
        assert_eq!(t.sized(), "vector");
    }

    #[test]
    fn a_sized_vector_splits_into_base_and_size() {
        let t = parse_type("vector[2]");
        assert_eq!(t.bare, "vector");
        assert_eq!(t.sizes, vec!["2"]);
        assert_eq!(t.sized(), "vector[2]");
    }

    #[test]
    fn a_sized_matrix_keeps_both_sizes_in_order() {
        let t = parse_type("matrix[K, N]");
        assert_eq!(t.bare, "matrix");
        assert_eq!(t.sizes, vec!["K", "N"]);
        assert_eq!(t.category, TypeCategory::Matrix);
        assert_eq!(t.sized(), "matrix[K, N]");
    }

    #[test]
    fn a_size_may_be_an_expression_over_parameters() {
        let t = parse_type("vector[2 * K + 1]");
        assert_eq!(t.sizes, vec!["2 * K + 1"]);
        assert_eq!(t.sized(), "vector[2 * K + 1]");
    }

    #[test]
    fn a_size_expression_may_contain_a_call_with_commas() {
        let t = parse_type("matrix[num_elements(x), min(a, b)]");
        assert_eq!(t.sizes, vec!["num_elements(x)", "min(a, b)"]);
    }

    #[test]
    fn scalar_categories_are_recognised() {
        assert_eq!(parse_type("real").category, TypeCategory::Real);
        assert_eq!(parse_type("int").category, TypeCategory::Int);
        assert_eq!(parse_type("complex").category, TypeCategory::Complex);
        assert_eq!(parse_type("row_vector").category, TypeCategory::RowVector);
    }

    #[test]
    fn a_data_qualifier_does_not_change_the_category() {
        assert_eq!(parse_type("data vector").category, TypeCategory::Vector);
        assert_eq!(parse_type("data real").category, TypeCategory::Real);
    }

    #[test]
    fn an_array_type_keeps_its_brackets_because_stan_needs_them() {
        let t = parse_type("array[] real");
        assert_eq!(t.category, TypeCategory::Array);
        assert_eq!(t.bare, "array[] real");
        assert_eq!(t.sizes, Vec::<String>::new());
    }

    #[test]
    fn a_multidimensional_array_type_is_passed_through() {
        let t = parse_type("array[,] vector");
        assert_eq!(t.category, TypeCategory::Array);
        assert_eq!(t.bare, "array[,] vector");
    }

    #[test]
    fn an_unknown_type_is_passed_through_rather_than_rejected() {
        let t = parse_type("tuple(real, real)");
        assert_eq!(t.category, TypeCategory::Other);
        assert_eq!(t.bare, "tuple(real, real)");
    }

    #[test]
    fn an_unbalanced_bracket_is_handed_back_whole() {
        let t = parse_type("vector[2");
        assert_eq!(t.bare, "vector[2");
        assert!(t.sizes.is_empty());
    }

    #[test]
    fn size_arity_matches_what_a_declaration_needs() {
        assert_eq!(TypeCategory::Real.size_arity(), Some(0));
        assert_eq!(TypeCategory::Vector.size_arity(), Some(1));
        assert_eq!(TypeCategory::Matrix.size_arity(), Some(2));
        assert_eq!(TypeCategory::Array.size_arity(), None);
    }

    #[test]
    fn accessors_belong_to_the_categories_that_have_them() {
        assert_eq!(TypeCategory::Vector.accessors(), &["size"]);
        assert_eq!(TypeCategory::RowVector.accessors(), &["size"]);
        assert_eq!(TypeCategory::Matrix.accessors(), &["rows", "cols"]);
        assert!(TypeCategory::Real.accessors().is_empty());
    }

    #[test]
    fn matching_ignores_sizes_because_stan_signatures_have_none() {
        assert!(parse_type("vector[2]").matches(&parse_type("vector")));
        assert!(parse_type("vector").matches(&parse_type("vector[K]")));
        assert!(!parse_type("vector").matches(&parse_type("row_vector")));
    }

    #[test]
    fn matching_normalizes_whitespace() {
        assert!(parse_type("array[]  real").matches(&parse_type("array[] real")));
    }

    #[test]
    fn splitting_args_respects_every_kind_of_nesting() {
        assert_eq!(
            split_top_level_args("real x, func(real, int) -> real f, array[N, M] real z"),
            vec![
                "real x",
                " func(real, int) -> real f",
                " array[N, M] real z"
            ]
        );
    }

    #[test]
    fn splitting_args_ignores_commas_in_braces_and_strings() {
        assert_eq!(
            split_top_level_args("{1, 2}, \"a, b\""),
            vec!["{1, 2}", " \"a, b\""]
        );
    }

    #[test]
    fn splitting_an_empty_list_yields_one_empty_piece() {
        assert_eq!(split_top_level_args(""), vec![""]);
    }

    #[test]
    fn parsing_is_deterministic() {
        assert_eq!(parse_type("matrix[K, N]"), parse_type("matrix[K, N]"));
    }
}
