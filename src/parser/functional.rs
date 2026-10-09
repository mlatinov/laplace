//! Functional parameters (`func(real) -> real f`) and the `@wait(f)`
//! return-type placeholder.
//!
//! Stan has no function values, so laplace adds them at compile time: a
//! function that takes a `func(...) -> ...` parameter is a *higher-order
//! function* (HOF), and laplace emits one specialized copy of it per
//! distinct function actually passed in. Nothing generic survives into
//! the `.stan` output.
//!
//! The contract lives in the parameter list, not in a separate
//! annotation:
//!
//! ```stan
//! real apply_twice(real x, func(real) -> real f) {
//!   real a = f(x);
//!   return f(a);
//! }
//! ```
//!
//! # `@wait(f)`
//!
//! Stan needs *sized* local declarations, and the author of a HOF cannot
//! know how long the bound function's result will be. `@wait(f)` stands
//! for "whatever concrete sized type ends up bound to `f`" and is
//! resolved during monomorphization:
//!
//! ```stan
//! matrix[num_elements(x), @wait(f).size] out;
//! @wait(f) row = f(x[i]);
//! ```
//!
//! This module only *finds and describes* these constructs. Checking and
//! substituting them is [`crate::monomorphize`]'s job.

use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::types::{
    matching_paren, parse_type, split_top_level_args, StanType, TypeCategory,
};

/// The keyword that introduces a functional parameter type.
pub const FUNC_KEYWORD: &str = "func";

/// The placeholder for a bound function's concrete return type.
pub const WAIT_KEYWORD: &str = "@wait";

/// A parameter whose type is a function shape rather than a value type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionalParam {
    /// The parameter's name, as the HOF's body calls it.
    pub name: String,
    /// Its position in the function's parameter list, so the
    /// corresponding argument can be found at a call site and dropped
    /// from the specialized signature.
    pub index: usize,
    pub arg_types: Vec<StanType>,
    pub return_type: StanType,
}

impl FunctionalParam {
    /// The shape as it was written, for error messages.
    pub fn shape(&self) -> String {
        let args: Vec<String> = self.arg_types.iter().map(|t| t.bare.clone()).collect();
        format!("func({}) -> {}", args.join(", "), self.return_type.bare)
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum FunctionalError {
    #[error(
        "`{param}`'s type is not a valid functional parameter: expected `func(<types>) -> <type>`"
    )]
    Malformed { param: String },

    #[error(
        "`{param}`'s type `{shape}` has a `func` inside it -- a functional parameter may not \
         itself take or return a function in this version"
    )]
    Nested { param: String, shape: String },

    #[error(
        "`{param}` returns `{return_type}`, and array return types are not supported for \
         functional parameters in this version"
    )]
    ArrayReturn { param: String, return_type: String },
}

impl FunctionalError {
    /// The `help:` line to print under the error.
    pub fn help(&self) -> &'static str {
        match self {
            FunctionalError::Malformed { .. } => {
                "write the shape as `func(real) -> real f`: the argument types in \
                 parentheses, then `->`, then the return type, then the parameter name"
            }
            FunctionalError::Nested { .. } => {
                "pass the inner function in separately as its own functional parameter"
            }
            FunctionalError::ArrayReturn { .. } => {
                "return a vector, row_vector, matrix or scalar instead, or wrap the array in \
                 a function that returns one of those"
            }
        }
    }
}

/// Whether `type_text` is a functional parameter type at all.
///
/// Cheap and total, so a signature can be classified before anything is
/// validated: a type that merely mentions `func` somewhere is still worth
/// parsing, because that is how a nested `func` gets reported rather than
/// silently treated as a value type.
pub fn is_functional_type(type_text: &str) -> bool {
    let mut from = 0usize;
    while let Some(rel) = type_text[from..].find(FUNC_KEYWORD) {
        let at = from + rel;
        if func_keyword_at(type_text, at) {
            return true;
        }
        from = at + FUNC_KEYWORD.len();
    }
    false
}

/// Whether the word `func` starts at `at` in `text`.
fn func_keyword_at(text: &str, at: usize) -> bool {
    let bytes = text.as_bytes();
    if !text[at..].starts_with(FUNC_KEYWORD) {
        return false;
    }
    let end = at + FUNC_KEYWORD.len();
    let before_ok = at == 0 || !is_ident_char(bytes[at - 1]);
    // `func` must be followed by its argument list.
    let after_ok = text[end..].trim_start().starts_with('(');
    before_ok && after_ok
}

/// Parse a parameter whose type is a `func(...) -> ...` shape.
///
/// `index` is the parameter's position in the signature. Returns `None`
/// when the type is an ordinary value type and this parameter is not
/// functional at all.
pub fn parse_functional_param(
    name: &str,
    type_text: &str,
    index: usize,
) -> Option<Result<FunctionalParam, FunctionalError>> {
    if !is_functional_type(type_text) {
        return None;
    }
    Some(parse_shape(name, type_text, index))
}

fn parse_shape(
    name: &str,
    type_text: &str,
    index: usize,
) -> Result<FunctionalParam, FunctionalError> {
    let text = type_text.trim();
    let malformed = || FunctionalError::Malformed {
        param: name.to_string(),
    };

    if !func_keyword_at(text, 0) {
        return Err(malformed());
    }
    let open = text.find('(').ok_or_else(malformed)?;
    let close = matching_paren(text, open).ok_or_else(malformed)?;

    let args_text = &text[open + 1..close];
    let rest = text[close + 1..].trim_start();
    let return_text = rest.strip_prefix("->").ok_or_else(malformed)?.trim();
    if return_text.is_empty() {
        return Err(malformed());
    }

    let arg_types: Vec<StanType> = split_top_level_args(args_text)
        .into_iter()
        .map(|arg| arg.trim().to_string())
        .filter(|arg| !arg.is_empty())
        .map(|arg| parse_type(&arg))
        .collect();
    let return_type = parse_type(return_text);

    // One level only: a functional parameter's shape may not itself
    // contain a function. Checked after parsing so the error can quote
    // the shape the author wrote.
    let nested = arg_types
        .iter()
        .chain(std::iter::once(&return_type))
        .any(|t| is_functional_type(&t.bare));
    let param = FunctionalParam {
        name: name.to_string(),
        index,
        arg_types,
        return_type,
    };
    if nested {
        return Err(FunctionalError::Nested {
            param: name.to_string(),
            shape: param.shape(),
        });
    }
    if param.return_type.category == TypeCategory::Array {
        return Err(FunctionalError::ArrayReturn {
            param: name.to_string(),
            return_type: param.return_type.bare.clone(),
        });
    }

    Ok(param)
}

/// One `@wait(f)` or `@wait(f).size` in a HOF body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitUse {
    /// The functional parameter named inside the parentheses.
    pub param: String,
    /// The accessor after the closing paren, if any: `size`, `rows`,
    /// `cols`. `None` means the whole sized type was asked for.
    pub accessor: Option<String>,
    /// Byte range of the whole construct, accessor included, so it can
    /// be replaced wholesale.
    pub range: Range<usize>,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("`@wait` must name a functional parameter, as in `@wait(f)` or `@wait(f).size`")]
pub struct WaitSyntaxError {
    pub offset: usize,
}

impl WaitSyntaxError {
    pub fn help(&self) -> &'static str {
        "write `@wait(f)` where `f` is one of this function's `func(...) -> ...` parameters"
    }
}

/// Find every `@wait(...)` use in `source`, in order.
///
/// Comments and string literals are skipped. A malformed `@wait` is an
/// error rather than being ignored: it can only be a typo for the real
/// thing, and silently passing `@wait` through would put laplace syntax
/// into the generated `.stan`.
pub fn find_wait_uses(source: &str) -> Result<Vec<WaitUse>, WaitSyntaxError> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut uses = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = source[search_from..].find(WAIT_KEYWORD) {
        let start = search_from + rel;
        search_from = start + WAIT_KEYWORD.len();
        if !mask.is_real(start) {
            continue;
        }
        // `@` cannot be part of an identifier, so the only thing to rule
        // out is a longer word starting here, like `@waiting`.
        let after_keyword = start + WAIT_KEYWORD.len();
        if after_keyword < bytes.len() && is_ident_char(bytes[after_keyword]) {
            continue;
        }

        let open_rel = source[after_keyword..]
            .find(|c: char| !c.is_whitespace())
            .ok_or(WaitSyntaxError { offset: start })?;
        let open = after_keyword + open_rel;
        if bytes[open] != b'(' {
            return Err(WaitSyntaxError { offset: start });
        }
        let close = matching_paren(source, open).ok_or(WaitSyntaxError { offset: start })?;
        let param = source[open + 1..close].trim().to_string();
        if param.is_empty() || !param.chars().all(|c| is_ident_char(c as u8)) {
            return Err(WaitSyntaxError { offset: start });
        }

        let (accessor, end) = match parse_accessor(source, close + 1) {
            Some((name, end)) => (Some(name), end),
            None => (None, close + 1),
        };

        uses.push(WaitUse {
            param,
            accessor,
            range: start..end,
        });
        search_from = end;
    }

    Ok(uses)
}

/// A `.name` directly after `at`, with its end offset.
fn parse_accessor(source: &str, at: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    if bytes.get(at) != Some(&b'.') {
        return None;
    }
    let mut end = at + 1;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    if end == at + 1 {
        return None;
    }
    Some((source[at + 1..end].to_string(), end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func(name: &str, ty: &str) -> FunctionalParam {
        parse_functional_param(name, ty, 0)
            .expect("should look functional")
            .expect("should parse")
    }

    fn err(name: &str, ty: &str) -> FunctionalError {
        parse_functional_param(name, ty, 0)
            .expect("should look functional")
            .expect_err("should not parse")
    }

    #[test]
    fn an_ordinary_value_type_is_not_functional() {
        assert!(parse_functional_param("x", "real", 0).is_none());
        assert!(parse_functional_param("k", "matrix[N, N]", 0).is_none());
        assert!(parse_functional_param("xs", "array[] real", 0).is_none());
    }

    #[test]
    fn a_type_that_merely_contains_the_letters_func_is_not_functional() {
        // Nothing in Stan is spelled this way today, but misreading a
        // value type as a function shape would reject valid code.
        assert!(!is_functional_type("funcy"));
        assert!(!is_functional_type("my_func"));
        assert!(!is_functional_type("func"));
        assert!(is_functional_type("func(real) -> real"));
        assert!(is_functional_type("func (real) -> real"));
    }

    #[test]
    fn a_one_argument_shape_parses() {
        let f = func("f", "func(real) -> real");
        assert_eq!(f.name, "f");
        assert_eq!(f.arg_types.len(), 1);
        assert_eq!(f.arg_types[0].bare, "real");
        assert_eq!(f.return_type.bare, "real");
        assert_eq!(f.shape(), "func(real) -> real");
    }

    #[test]
    fn a_multi_argument_shape_splits_on_its_own_commas() {
        let f = func("g", "func(real, int, vector) -> matrix");
        assert_eq!(
            f.arg_types
                .iter()
                .map(|t| t.bare.as_str())
                .collect::<Vec<_>>(),
            vec!["real", "int", "vector"]
        );
        assert_eq!(f.return_type.category, TypeCategory::Matrix);
    }

    #[test]
    fn a_zero_argument_shape_parses() {
        let f = func("f", "func() -> real");
        assert!(f.arg_types.is_empty());
        assert_eq!(f.shape(), "func() -> real");
    }

    #[test]
    fn an_array_argument_type_is_allowed() {
        let f = func("f", "func(array[] real) -> real");
        assert_eq!(f.arg_types[0].bare, "array[] real");
    }

    #[test]
    fn extra_whitespace_does_not_matter() {
        let f = func("f", "func ( real ,  int )  ->  vector");
        assert_eq!(f.arg_types.len(), 2);
        assert_eq!(f.return_type.bare, "vector");
    }

    #[test]
    fn the_index_is_kept_so_the_argument_can_be_found_at_a_call_site() {
        let f = parse_functional_param("f", "func(real) -> real", 3)
            .unwrap()
            .unwrap();
        assert_eq!(f.index, 3);
    }

    #[test]
    fn a_missing_arrow_is_malformed() {
        assert_eq!(
            err("f", "func(real) real"),
            FunctionalError::Malformed {
                param: "f".to_string()
            }
        );
    }

    #[test]
    fn a_missing_return_type_is_malformed() {
        assert!(matches!(
            err("f", "func(real) ->"),
            FunctionalError::Malformed { .. }
        ));
    }

    #[test]
    fn an_unclosed_argument_list_is_malformed() {
        assert!(matches!(
            err("f", "func(real -> real"),
            FunctionalError::Malformed { .. }
        ));
    }

    #[test]
    fn a_func_returning_a_func_is_rejected() {
        let e = err("f", "func(real) -> func(real) -> real");
        assert!(matches!(e, FunctionalError::Nested { .. }), "{e:?}");
        assert!(e.help().contains("separately"), "{}", e.help());
    }

    #[test]
    fn a_func_taking_a_func_is_rejected() {
        assert!(matches!(
            err("f", "func(func(real) -> real) -> real"),
            FunctionalError::Nested { .. }
        ));
    }

    #[test]
    fn an_array_return_type_is_rejected_with_a_clear_message() {
        let e = err("f", "func(real) -> array[] real");
        assert!(matches!(e, FunctionalError::ArrayReturn { .. }), "{e:?}");
        assert!(e.to_string().contains("array return types"), "{e}");
    }

    // ---- `@wait` -----------------------------------------------------

    #[test]
    fn a_bare_wait_is_found_with_no_accessor() {
        let source = "  @wait(f) row = f(x);\n";
        let uses = find_wait_uses(source).unwrap();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].param, "f");
        assert_eq!(uses[0].accessor, None);
        assert_eq!(&source[uses[0].range.clone()], "@wait(f)");
    }

    #[test]
    fn every_accessor_spelling_is_found() {
        let source = "matrix[@wait(f).rows, @wait(f).cols] a;\nvector[@wait(g).size] b;\n";
        let uses = find_wait_uses(source).unwrap();
        assert_eq!(
            uses.iter()
                .map(|u| (u.param.as_str(), u.accessor.as_deref()))
                .collect::<Vec<_>>(),
            vec![
                ("f", Some("rows")),
                ("f", Some("cols")),
                ("g", Some("size"))
            ]
        );
        assert_eq!(&source[uses[0].range.clone()], "@wait(f).rows");
    }

    #[test]
    fn a_wait_in_a_comment_or_string_is_ignored() {
        let source = "// @wait(f) is the placeholder\nprint(\"@wait(f)\");\n";
        assert!(find_wait_uses(source).unwrap().is_empty());
    }

    #[test]
    fn a_source_with_no_wait_yields_nothing() {
        assert!(find_wait_uses("real f(real x) {\n  return x;\n}\n")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn whitespace_between_wait_and_its_parentheses_is_allowed() {
        let uses = find_wait_uses("@wait (f) r = f(1);").unwrap();
        assert_eq!(uses[0].param, "f");
    }

    #[test]
    fn a_wait_without_parentheses_is_an_error() {
        let err = find_wait_uses("@wait f r = f(1);").unwrap_err();
        assert!(err.help().contains("@wait(f)"));
    }

    #[test]
    fn a_wait_with_an_empty_or_non_identifier_argument_is_an_error() {
        assert!(find_wait_uses("@wait() r;").is_err());
        assert!(find_wait_uses("@wait(a + b) r;").is_err());
    }

    #[test]
    fn a_longer_word_starting_with_wait_is_not_a_placeholder() {
        assert!(find_wait_uses("real waiting_room = 1;").unwrap().is_empty());
        assert!(find_wait_uses("// @waiting\nreal x = 1;")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn scanning_is_deterministic() {
        let source = "@wait(f) a = f(1);\nvector[@wait(g).size] b;\n";
        assert_eq!(
            find_wait_uses(source).unwrap(),
            find_wait_uses(source).unwrap()
        );
    }
}
