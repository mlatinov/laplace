//! Resolving `@wait(f)` against the function actually bound to `f`.
//!
//! A higher-order function's author cannot write a sized local
//! declaration for `f`'s result, because the size depends on what gets
//! bound. `@wait(f)` stands in for that type until monomorphization
//! knows the answer:
//!
//! ```stan
//! matrix[num_elements(x), @wait(f).size] out;   // -> matrix[num_elements(x), 2]
//! @wait(f) row = f(x[i]);                       // -> vector[2] row = to_pair(x[i]);
//! ```
//!
//! The size comes from the bound function's *return size annotation*
//! (`vector[2] to_pair(real x)`), which is laplace's own extension --
//! a Stan signature carries no sizes at all.
//!
//! # Parameter-dependent sizes
//!
//! `vector[K] basis(real t, int K)` has a size that is only known per
//! call. Such a function can still be bound, but `@wait(f)` may then
//! only appear in a declaration whose initializer is a *direct* call to
//! `f`, so the actual arguments can be substituted into the size:
//!
//! ```stan
//! @wait(f) r = f(t, K);      // -> vector[(K)] r = basis(t, K);
//! ```
//!
//! Reading `.size`/`.rows`/`.cols` off such a function is an error in
//! this version: there is no call site to take the arguments from.

use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::functional::{find_wait_uses, WaitUse};
use crate::parser::types::TypeCategory;

use super::calls::find_calls;

/// What a bound function returns, in enough detail to resolve `@wait`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnShape {
    /// The name the bound function has in the compiled output, for
    /// error messages.
    pub bound: String,
    pub category: TypeCategory,
    /// The bare return type Stan sees: `vector`, `real`, ...
    pub bare: String,
    /// The size expressions from the return annotation, if any.
    pub sizes: Vec<String>,
    /// The bound function's own parameter names, in order -- what a
    /// parameter-dependent size expression refers to.
    pub param_names: Vec<String>,
}

impl ReturnShape {
    /// How many sizes a declaration of this return type needs.
    fn required_sizes(&self) -> usize {
        self.category.size_arity().unwrap_or(0)
    }

    /// Whether the size expressions mention the bound function's own
    /// parameters, so they are only meaningful where actual arguments
    /// are known.
    pub fn sizes_depend_on_parameters(&self) -> bool {
        self.sizes
            .iter()
            .any(|size| self.param_names.iter().any(|p| mentions_word(size, p)))
    }

    /// The sized type as a declaration, given the actual arguments of
    /// the call the declaration is initialized with (empty when the
    /// sizes do not depend on parameters).
    fn sized_type(&self, actuals: &[String]) -> String {
        if self.sizes.is_empty() {
            return self.bare.clone();
        }
        let sizes: Vec<String> = self
            .sizes
            .iter()
            .map(|size| substitute_params(size, &self.param_names, actuals))
            .collect();
        format!("{}[{}]", self.bare, sizes.join(", "))
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WaitError {
    #[error("`@wait({param})` does not name a functional parameter of `{function}`")]
    UnknownParam {
        function: String,
        param: String,
        offset: usize,
    },

    #[error(
        "`@wait({param}).{accessor}` does not fit `{param}`'s declared return type \
         (`{category}`)"
    )]
    AccessorMismatch {
        param: String,
        accessor: String,
        category: &'static str,
        offset: usize,
    },

    #[error("`{function}` needs the return size of `{bound}` (used via `@wait({param})`)")]
    MissingReturnSize {
        function: String,
        param: String,
        bound: String,
        /// How the annotation should look, e.g. `vector[2]`.
        example: String,
        offset: usize,
    },

    #[error(
        "`{bound}`'s return size depends on its own parameters, so \
         `@wait({param}).{accessor}` cannot be resolved"
    )]
    AccessorOnParameterDependentSize {
        param: String,
        accessor: String,
        bound: String,
        offset: usize,
    },

    #[error(
        "`{bound}`'s return size depends on its own parameters, so `@wait({param})` is only \
         allowed in a declaration initialized by a direct call to `{param}`"
    )]
    NeedsDirectCall {
        param: String,
        bound: String,
        offset: usize,
    },
}

impl WaitError {
    pub fn offset(&self) -> usize {
        match self {
            WaitError::UnknownParam { offset, .. }
            | WaitError::AccessorMismatch { offset, .. }
            | WaitError::MissingReturnSize { offset, .. }
            | WaitError::AccessorOnParameterDependentSize { offset, .. }
            | WaitError::NeedsDirectCall { offset, .. } => *offset,
        }
    }

    pub fn help(&self) -> String {
        match self {
            WaitError::UnknownParam { .. } => {
                "name one of the function's `func(...) -> ...` parameters".to_string()
            }
            WaitError::AccessorMismatch { category, .. } => format!(
                "a `{category}` return has no such size; use `.size` for a vector or \
                 row_vector, `.rows`/`.cols` for a matrix, or `@wait(f)` for the whole type"
            ),
            WaitError::MissingReturnSize { bound, example, .. } => {
                format!("annotate the return type, e.g. `{example} {bound}(...)`")
            }
            WaitError::AccessorOnParameterDependentSize { param, .. } => format!(
                "declare the value with `@wait({param}) r = {param}(...);` instead, so the \
                 arguments are known, or give the bound function a fixed return size"
            ),
            WaitError::NeedsDirectCall { param, .. } => format!(
                "write `@wait({param}) r = {param}(...);` so the size arguments can be read \
                 from the call, or give the bound function a fixed return size"
            ),
        }
    }
}

/// Check every `@wait` in a HOF body against the *declared* shapes of
/// its functional parameters, before any binding is known.
///
/// Catches the mistakes that are the author's rather than the caller's:
/// naming a parameter that does not exist, and asking for a size the
/// declared return category does not have.
pub fn check_declared(
    function: &str,
    body: &str,
    declared: &[(String, TypeCategory)],
) -> Result<(), WaitError> {
    for use_ in find_wait_uses(body).unwrap_or_default() {
        let Some((_, category)) = declared.iter().find(|(name, _)| *name == use_.param) else {
            return Err(WaitError::UnknownParam {
                function: function.to_string(),
                param: use_.param.clone(),
                offset: use_.range.start,
            });
        };
        if let Some(accessor) = &use_.accessor {
            if !category.accessors().contains(&accessor.as_str()) {
                return Err(WaitError::AccessorMismatch {
                    param: use_.param.clone(),
                    accessor: accessor.clone(),
                    category: category.describe(),
                    offset: use_.range.start,
                });
            }
        }
    }
    Ok(())
}

/// Replace every `@wait(...)` in `body` with the concrete type or size
/// it stands for.
///
/// `shapes` maps each functional parameter name to what the function
/// bound to it returns. Returns the rewritten body.
pub fn substitute(
    function: &str,
    body: &str,
    shapes: &[(String, ReturnShape)],
) -> Result<String, WaitError> {
    let uses = find_wait_uses(body).unwrap_or_default();
    if uses.is_empty() {
        return Ok(body.to_string());
    }
    let mask = CodeMask::new(body);

    let mut out = String::with_capacity(body.len());
    let mut cursor = 0usize;
    for use_ in &uses {
        let shape = shapes
            .iter()
            .find(|(name, _)| *name == use_.param)
            .map(|(_, shape)| shape)
            .ok_or_else(|| WaitError::UnknownParam {
                function: function.to_string(),
                param: use_.param.clone(),
                offset: use_.range.start,
            })?;

        let replacement = resolve_one(function, body, &mask, use_, shape)?;
        out.push_str(&body[cursor..use_.range.start]);
        out.push_str(&replacement);
        cursor = use_.range.end;
    }
    out.push_str(&body[cursor..]);
    Ok(out)
}

fn resolve_one(
    function: &str,
    body: &str,
    mask: &CodeMask,
    use_: &WaitUse,
    shape: &ReturnShape,
) -> Result<String, WaitError> {
    let needs = shape.required_sizes();
    if needs > 0 && shape.sizes.len() < needs {
        return Err(WaitError::MissingReturnSize {
            function: function.to_string(),
            param: use_.param.clone(),
            bound: shape.bound.clone(),
            example: example_annotation(shape),
            offset: use_.range.start,
        });
    }

    match &use_.accessor {
        Some(accessor) => {
            if shape.sizes_depend_on_parameters() {
                return Err(WaitError::AccessorOnParameterDependentSize {
                    param: use_.param.clone(),
                    accessor: accessor.clone(),
                    bound: shape.bound.clone(),
                    offset: use_.range.start,
                });
            }
            let index = shape
                .category
                .accessors()
                .iter()
                .position(|a| a == accessor)
                .ok_or_else(|| WaitError::AccessorMismatch {
                    param: use_.param.clone(),
                    accessor: accessor.clone(),
                    category: shape.category.describe(),
                    offset: use_.range.start,
                })?;
            Ok(shape.sizes[index].clone())
        }
        None => {
            if !shape.sizes_depend_on_parameters() {
                return Ok(shape.sized_type(&[]));
            }
            // The size needs this call's arguments, so the declaration
            // has to be initialized by a direct call to the parameter.
            let actuals =
                direct_call_args(body, mask, use_.range.end, &use_.param).ok_or_else(|| {
                    WaitError::NeedsDirectCall {
                        param: use_.param.clone(),
                        bound: shape.bound.clone(),
                        offset: use_.range.start,
                    }
                })?;
            Ok(shape.sized_type(&actuals))
        }
    }
}

/// What a correct return annotation would look like for this shape,
/// for the `help:` line.
fn example_annotation(shape: &ReturnShape) -> String {
    match shape.category.size_arity() {
        Some(1) => format!("{}[2]", shape.bare),
        Some(2) => format!("{}[2, 2]", shape.bare),
        _ => shape.bare.clone(),
    }
}

/// The argument texts of a direct call to `param` that initializes the
/// declaration starting at `from`.
///
/// Recognises exactly `@wait(f) <name> = f(<args>);` -- an `=` followed
/// by a call to `f`, before the statement's `;`.
fn direct_call_args(body: &str, mask: &CodeMask, from: usize, param: &str) -> Option<Vec<String>> {
    let statement_end = body[from..].find(';').map(|i| from + i)?;
    let statement = &body[from..statement_end];
    let equals = statement.find('=')?;

    // `==` is a comparison, not an initializer.
    if statement[equals + 1..].starts_with('=') {
        return None;
    }

    let call = find_calls(body, mask, param)
        .into_iter()
        .find(|c| c.name_range.start > from + equals && c.full_range.end <= statement_end + 1)?;
    Some(
        call.arg_ranges
            .iter()
            .map(|range| body[range.clone()].trim().to_string())
            .collect(),
    )
}

/// Replace each whole-word parameter name in a size expression with the
/// actual argument, parenthesized so the surrounding arithmetic keeps
/// its meaning.
fn substitute_params(size: &str, param_names: &[String], actuals: &[String]) -> String {
    let bytes = size.as_bytes();
    let mut out = String::with_capacity(size.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if !is_ident_char(bytes[i]) || (i > 0 && is_ident_char(bytes[i - 1])) {
            out.push(size[i..].chars().next().expect("in bounds"));
            i += size[i..].chars().next().map(char::len_utf8).unwrap_or(1);
            continue;
        }
        let mut end = i;
        while end < bytes.len() && is_ident_char(bytes[end]) {
            end += 1;
        }
        let word = &size[i..end];
        match param_names.iter().position(|p| p == word) {
            Some(index) if index < actuals.len() => {
                out.push('(');
                out.push_str(&actuals[index]);
                out.push(')');
            }
            _ => out.push_str(word),
        }
        i = end;
    }
    out
}

/// Whether `text` uses `word` as a whole identifier.
fn mentions_word(text: &str, word: &str) -> bool {
    let bytes = text.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(word) {
        let start = from + rel;
        let end = start + word.len();
        let before_ok = start == 0 || !is_ident_char(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_ident_char(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// `body` with every `@wait(...)` construct replaced by spaces.
///
/// `@wait(f)` mentions `f` without calling it, which would otherwise
/// look exactly like storing the parameter. Blanking the placeholders --
/// with the same number of bytes, so every offset still lines up -- lets
/// the "only ever called" check see the body the way Stan will.
pub fn blank_uses(body: &str) -> String {
    let mut out = body.to_string();
    for range in uses(body) {
        out.replace_range(range.clone(), &" ".repeat(range.len()));
    }
    out
}

/// The byte range of each `@wait` use, for callers that only need to
/// know whether a body has any.
pub fn uses(body: &str) -> Vec<Range<usize>> {
    find_wait_uses(body)
        .unwrap_or_default()
        .into_iter()
        .map(|u| u.range)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_shape(sizes: &[&str], params: &[&str]) -> ReturnShape {
        ReturnShape {
            bound: "to_pair".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: sizes.iter().map(|s| s.to_string()).collect(),
            param_names: params.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn matrix_shape(sizes: &[&str]) -> ReturnShape {
        ReturnShape {
            bound: "grid".to_string(),
            category: TypeCategory::Matrix,
            bare: "matrix".to_string(),
            sizes: sizes.iter().map(|s| s.to_string()).collect(),
            param_names: Vec::new(),
        }
    }

    fn real_shape() -> ReturnShape {
        ReturnShape {
            bound: "add_one".to_string(),
            category: TypeCategory::Real,
            bare: "real".to_string(),
            sizes: Vec::new(),
            param_names: vec!["x".to_string()],
        }
    }

    fn sub(body: &str, shape: ReturnShape) -> Result<String, WaitError> {
        substitute("hof", body, &[("f".to_string(), shape)])
    }

    #[test]
    fn a_body_with_no_wait_is_returned_unchanged() {
        let body = "{\n  return f(x);\n}\n";
        assert_eq!(sub(body, real_shape()).unwrap(), body);
    }

    #[test]
    fn a_bare_wait_expands_to_the_sized_type() {
        let body = "{\n  @wait(f) row = f(x);\n  return row;\n}\n";
        assert_eq!(
            sub(body, vector_shape(&["2"], &["x"])).unwrap(),
            "{\n  vector[2] row = f(x);\n  return row;\n}\n"
        );
    }

    #[test]
    fn a_scalar_return_expands_to_the_bare_type() {
        let body = "{\n  @wait(f) a = f(x);\n  return a;\n}\n";
        assert_eq!(
            sub(body, real_shape()).unwrap(),
            "{\n  real a = f(x);\n  return a;\n}\n"
        );
    }

    #[test]
    fn the_size_accessor_expands_to_the_size_expression() {
        let body = "{\n  matrix[num_elements(x), @wait(f).size] out;\n}\n";
        assert_eq!(
            sub(body, vector_shape(&["2"], &["x"])).unwrap(),
            "{\n  matrix[num_elements(x), 2] out;\n}\n"
        );
    }

    #[test]
    fn matrix_accessors_expand_to_their_own_dimension() {
        let body = "{\n  matrix[@wait(f).rows, @wait(f).cols] out;\n}\n";
        assert_eq!(
            sub(body, matrix_shape(&["3", "4"])).unwrap(),
            "{\n  matrix[3, 4] out;\n}\n"
        );
    }

    #[test]
    fn several_uses_in_one_body_are_all_replaced() {
        let body = "{\n  vector[@wait(f).size] a;\n  @wait(f) b = f(1);\n}\n";
        assert_eq!(
            sub(body, vector_shape(&["2"], &["x"])).unwrap(),
            "{\n  vector[2] a;\n  vector[2] b = f(1);\n}\n"
        );
    }

    #[test]
    fn a_missing_return_size_is_an_error_naming_the_bound_function() {
        let body = "{\n  @wait(f) r = f(x);\n}\n";
        let err = sub(body, vector_shape(&[], &["x"])).unwrap_err();
        assert!(
            matches!(err, WaitError::MissingReturnSize { .. }),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .contains("needs the return size of `to_pair`"),
            "{err}"
        );
        assert!(
            err.help().contains("vector[2] to_pair(...)"),
            "{}",
            err.help()
        );
    }

    #[test]
    fn a_scalar_return_needs_no_annotation() {
        let body = "{\n  @wait(f) r = f(x);\n}\n";
        assert!(sub(body, real_shape()).is_ok());
    }

    // ---- parameter-dependent sizes -----------------------------------

    #[test]
    fn a_parameter_dependent_size_substitutes_the_calls_arguments() {
        // `vector[K] basis(real t, int K)` bound to `f`.
        let shape = ReturnShape {
            bound: "basis".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: vec!["K".to_string()],
            param_names: vec!["t".to_string(), "K".to_string()],
        };
        let body = "{\n  @wait(f) r = f(x[i], n_basis);\n  return r;\n}\n";
        assert_eq!(
            sub(body, shape).unwrap(),
            "{\n  vector[(n_basis)] r = f(x[i], n_basis);\n  return r;\n}\n"
        );
    }

    #[test]
    fn a_substituted_size_expression_keeps_its_arithmetic() {
        let shape = ReturnShape {
            bound: "basis".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: vec!["2 * K + 1".to_string()],
            param_names: vec!["K".to_string()],
        };
        let body = "{\n  @wait(f) r = f(a + b);\n}\n";
        assert_eq!(
            sub(body, shape).unwrap(),
            "{\n  vector[2 * (a + b) + 1] r = f(a + b);\n}\n"
        );
    }

    #[test]
    fn a_parameter_dependent_size_outside_a_direct_call_is_an_error() {
        let shape = ReturnShape {
            bound: "basis".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: vec!["K".to_string()],
            param_names: vec!["K".to_string()],
        };
        let body = "{\n  @wait(f) r;\n  r = f(3);\n}\n";
        let err = sub(body, shape).unwrap_err();
        assert!(matches!(err, WaitError::NeedsDirectCall { .. }), "{err:?}");
        assert!(err.help().contains("@wait(f) r = f(...)"), "{}", err.help());
    }

    #[test]
    fn an_accessor_on_a_parameter_dependent_size_is_an_error() {
        let shape = ReturnShape {
            bound: "basis".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: vec!["K".to_string()],
            param_names: vec!["K".to_string()],
        };
        let body = "{\n  vector[@wait(f).size] r;\n}\n";
        let err = sub(body, shape).unwrap_err();
        assert!(
            matches!(err, WaitError::AccessorOnParameterDependentSize { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_size_that_merely_looks_like_a_parameter_name_is_not_substituted() {
        // `KK` is not `K`.
        let shape = ReturnShape {
            bound: "b".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: vec!["KK".to_string()],
            param_names: vec!["K".to_string()],
        };
        assert!(!shape.sizes_depend_on_parameters());
        let body = "{\n  @wait(f) r = f(1);\n}\n";
        assert_eq!(sub(body, shape).unwrap(), "{\n  vector[KK] r = f(1);\n}\n");
    }

    #[test]
    fn an_equality_test_is_not_an_initializer() {
        let shape = ReturnShape {
            bound: "basis".to_string(),
            category: TypeCategory::Vector,
            bare: "vector".to_string(),
            sizes: vec!["K".to_string()],
            param_names: vec!["K".to_string()],
        };
        let body = "{\n  @wait(f) r == f(3);\n}\n";
        assert!(matches!(
            sub(body, shape).unwrap_err(),
            WaitError::NeedsDirectCall { .. }
        ));
    }

    // ---- definition-time checks --------------------------------------

    #[test]
    fn a_declared_check_accepts_matching_accessors() {
        let declared = vec![
            ("f".to_string(), TypeCategory::Vector),
            ("g".to_string(), TypeCategory::Matrix),
        ];
        let body = "{\n  vector[@wait(f).size] a;\n  matrix[@wait(g).rows, @wait(g).cols] b;\n}\n";
        assert!(check_declared("hof", body, &declared).is_ok());
    }

    #[test]
    fn a_declared_check_rejects_an_unknown_parameter() {
        let declared = vec![("f".to_string(), TypeCategory::Vector)];
        let err = check_declared("hof", "{\n  @wait(nope) r;\n}\n", &declared).unwrap_err();
        assert!(matches!(err, WaitError::UnknownParam { .. }), "{err:?}");
    }

    #[test]
    fn a_declared_check_rejects_an_accessor_the_category_lacks() {
        let declared = vec![("f".to_string(), TypeCategory::Vector)];
        let err =
            check_declared("hof", "{\n  matrix[@wait(f).rows, 1] m;\n}\n", &declared).unwrap_err();
        assert!(matches!(err, WaitError::AccessorMismatch { .. }), "{err:?}");
        assert!(err.help().contains(".size"), "{}", err.help());
    }

    #[test]
    fn a_declared_check_rejects_an_accessor_on_a_scalar_return() {
        let declared = vec![("f".to_string(), TypeCategory::Real)];
        assert!(check_declared("hof", "{\n  int n = @wait(f).size;\n}\n", &declared).is_err());
    }

    #[test]
    fn substitution_is_deterministic() {
        let body = "{\n  vector[@wait(f).size] a;\n  @wait(f) b = f(1);\n}\n";
        let once = sub(body, vector_shape(&["2"], &["x"])).unwrap();
        let twice = sub(body, vector_shape(&["2"], &["x"])).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn blanking_keeps_every_other_byte_where_it_was() {
        let body = "{\n  @wait(f) r = f(x);\n  vector[@wait(f).size] v;\n}\n";
        let blanked = blank_uses(body);
        assert_eq!(blanked.len(), body.len());
        assert!(!blanked.contains("@wait"), "{blanked}");
        assert!(blanked.contains("r = f(x);"), "{blanked}");
        assert!(
            blanked.contains("vector[") && blanked.contains("] v;"),
            "{blanked}"
        );
    }

    #[test]
    fn blanking_a_body_without_placeholders_changes_nothing() {
        let body = "{\n  return f(x);\n}\n";
        assert_eq!(blank_uses(body), body);
    }

    #[test]
    fn uses_reports_every_placeholder_range() {
        assert_eq!(uses("{\n  @wait(f) a = f(1);\n}\n").len(), 1);
        assert!(uses("{\n  return f(1);\n}\n").is_empty());
    }
}
