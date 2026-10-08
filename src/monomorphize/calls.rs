//! Finding call sites, and the references that can appear in one.
//!
//! Shallow by design: laplace never parses Stan expressions. To find
//! `apply_twice(5, add_one)` it looks for the whole word `apply_twice`
//! followed by `(`, matches the parenthesis, and splits the argument
//! list on its top-level commas. That is enough to pull out an argument
//! that is a bare function name, which is the only thing a functional
//! parameter accepts.

use std::ops::Range;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::types::{matching_paren, split_top_level_args};

/// How a function is named at a point in the source.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reference {
    /// An unqualified name: a function of this same unit.
    Plain(String),
    /// `pkg::func`: a function of another package.
    Qualified { package: String, func: String },
}

impl Reference {
    /// Parse an argument that is supposed to be a bare function name.
    /// `None` when it is an expression, a literal, or anything else that
    /// cannot be a function reference.
    pub fn parse(text: &str) -> Option<Reference> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        let is_name = |s: &str| {
            !s.is_empty()
                && s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                && s.bytes().all(is_ident_char)
        };

        match text.split_once("::") {
            Some((package, func)) => {
                let (package, func) = (package.trim(), func.trim());
                (is_name(package) && is_name(func)).then(|| Reference::Qualified {
                    package: package.to_string(),
                    func: func.to_string(),
                })
            }
            None => is_name(text).then(|| Reference::Plain(text.to_string())),
        }
    }

    /// How the reference was written, for error messages.
    pub fn as_written(&self) -> String {
        match self {
            Reference::Plain(name) => name.clone(),
            Reference::Qualified { package, func } => format!("{package}::{func}"),
        }
    }
}

/// One call to a known function name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSite {
    /// The called name itself.
    pub name_range: Range<usize>,
    /// The whole call, from the first byte of the name through the
    /// closing `)`.
    pub full_range: Range<usize>,
    /// Each argument's byte range, in order. Empty for `f()`.
    pub arg_ranges: Vec<Range<usize>>,
}

impl CallSite {
    pub fn arity(&self) -> usize {
        self.arg_ranges.len()
    }

    /// One argument's text, trimmed.
    pub fn arg<'a>(&self, text: &'a str, index: usize) -> Option<&'a str> {
        self.arg_ranges
            .get(index)
            .map(|range| text[range.clone()].trim())
    }
}

/// Find every call to `name` in `text`, in source order.
///
/// `name` may be qualified (`pkg::func`); only the byte *before* the
/// match has to be a non-identifier, since `:` cannot start one. Matches
/// inside comments and string literals are skipped.
pub fn find_calls(text: &str, mask: &CodeMask, name: &str) -> Vec<CallSite> {
    if name.is_empty() {
        return Vec::new();
    }
    let bytes = text.as_bytes();
    let mut calls = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = text[search_from..].find(name) {
        let start = search_from + rel;
        let end = start + name.len();
        search_from = end;

        if !mask.is_real(start) {
            continue;
        }
        let before_ok = start == 0 || !is_ident_char(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_ident_char(bytes[end]);
        if !before_ok || !after_ok {
            continue;
        }
        // The `(` may be separated from the name by whitespace.
        let rest = &text[end..];
        let open_rel = rest.len() - rest.trim_start().len();
        let open = end + open_rel;
        if bytes.get(open) != Some(&b'(') {
            continue;
        }
        let Some(close) = matching_paren(text, open) else {
            continue;
        };

        calls.push(CallSite {
            name_range: start..end,
            full_range: start..close + 1,
            arg_ranges: argument_ranges(text, open + 1..close),
        });
        search_from = close + 1;
    }

    calls
}

/// Byte ranges of the top-level arguments inside an argument list.
fn argument_ranges(text: &str, inside: Range<usize>) -> Vec<Range<usize>> {
    let args = split_top_level_args(&text[inside.clone()]);
    if args.len() == 1 && args[0].trim().is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::with_capacity(args.len());
    let mut at = inside.start;
    for arg in &args {
        ranges.push(at..at + arg.len());
        // +1 for the comma that separated this argument from the next.
        at += arg.len() + 1;
    }
    ranges
}

/// Whether `text` uses the whole word `name` anywhere *other* than
/// directly before a `(`.
///
/// This is how "a functional parameter may only be called, never stored,
/// returned or passed on" is enforced without a Stan parser: if every
/// occurrence is a call, none of those things can be happening.
pub fn uses_outside_call_position(text: &str, mask: &CodeMask, name: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut search_from = 0usize;

    while let Some(rel) = text[search_from..].find(name) {
        let start = search_from + rel;
        let end = start + name.len();
        search_from = end;

        if !mask.is_real(start) {
            continue;
        }
        let before_ok = start == 0 || !is_ident_char(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_ident_char(bytes[end]);
        if !before_ok || !after_ok {
            continue;
        }
        if !text[end..].trim_start().starts_with('(') {
            return Some(start);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calls(text: &str, name: &str) -> Vec<CallSite> {
        find_calls(text, &CodeMask::new(text), name)
    }

    #[test]
    fn a_plain_reference_parses() {
        assert_eq!(
            Reference::parse("add_one"),
            Some(Reference::Plain("add_one".to_string()))
        );
    }

    #[test]
    fn a_qualified_reference_parses() {
        assert_eq!(
            Reference::parse(" stats::mean_ "),
            Some(Reference::Qualified {
                package: "stats".to_string(),
                func: "mean_".to_string(),
            })
        );
    }

    #[test]
    fn an_expression_is_not_a_reference() {
        for text in ["x + 1", "5", "f(x)", "", "2.0", "a::b::c", "a-b"] {
            assert_eq!(Reference::parse(text), None, "{text:?} should not parse");
        }
    }

    #[test]
    fn a_name_starting_with_a_digit_is_not_a_reference() {
        assert_eq!(Reference::parse("2x"), None);
    }

    #[test]
    fn a_call_is_found_with_its_arguments() {
        let text = "  real r = apply_twice(5, add_one);\n";
        let found = calls(text, "apply_twice");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].arity(), 2);
        assert_eq!(&text[found[0].full_range.clone()], "apply_twice(5, add_one)");
        assert_eq!(found[0].arg(text, 0), Some("5"));
        assert_eq!(found[0].arg(text, 1), Some("add_one"));
    }

    #[test]
    fn a_zero_argument_call_has_no_arguments() {
        let text = "real r = thing();\n";
        let found = calls(text, "thing");
        assert_eq!(found[0].arity(), 0);
    }

    #[test]
    fn nested_calls_and_commas_do_not_confuse_argument_splitting() {
        let text = "real r = h(f(1, 2), {3, 4}, g(a));\n";
        let found = calls(text, "h");
        assert_eq!(found[0].arity(), 3);
        assert_eq!(found[0].arg(text, 0), Some("f(1, 2)"));
        assert_eq!(found[0].arg(text, 1), Some("{3, 4}"));
        assert_eq!(found[0].arg(text, 2), Some("g(a)"));
    }

    #[test]
    fn a_qualified_call_is_found() {
        let text = "real r = stats::apply(x, mean_);\n";
        let found = calls(text, "stats::apply");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].arg(text, 1), Some("mean_"));
    }

    #[test]
    fn a_substring_match_is_not_a_call() {
        assert!(calls("real r = apply_twice_more(1);\n", "apply_twice").is_empty());
        assert!(calls("real r = my_apply(1);\n", "apply").is_empty());
    }

    #[test]
    fn a_name_not_followed_by_a_paren_is_not_a_call() {
        assert!(calls("real r = apply + 1;\n", "apply").is_empty());
    }

    #[test]
    fn whitespace_before_the_paren_is_allowed() {
        let text = "real r = apply (1);\n";
        assert_eq!(calls(text, "apply").len(), 1);
    }

    #[test]
    fn calls_in_comments_and_strings_are_ignored() {
        let text = "// apply(1)\nprint(\"apply(1)\");\n";
        assert!(calls(text, "apply").is_empty());
    }

    #[test]
    fn several_calls_are_found_in_order() {
        let text = "apply(1); apply(2); apply(3);\n";
        let found = calls(text, "apply");
        assert_eq!(found.len(), 3);
        assert!(found[0].full_range.start < found[1].full_range.start);
    }

    #[test]
    fn argument_ranges_point_at_the_real_text() {
        let text = "h(alpha, beta + 1)";
        let found = calls(text, "h");
        assert_eq!(&text[found[0].arg_ranges[0].clone()], "alpha");
        assert_eq!(&text[found[0].arg_ranges[1].clone()], " beta + 1");
    }

    // ---- use outside call position -----------------------------------

    #[test]
    fn a_name_only_ever_called_reports_nothing() {
        let text = "{\n  real a = f(x);\n  return f(a);\n}\n";
        assert_eq!(uses_outside_call_position(text, &CodeMask::new(text), "f"), None);
    }

    #[test]
    fn storing_a_functional_parameter_is_reported() {
        let text = "{\n  real g = f;\n  return g(1);\n}\n";
        let at = uses_outside_call_position(text, &CodeMask::new(text), "f").unwrap();
        assert_eq!(&text[at..at + 1], "f");
    }

    #[test]
    fn passing_a_functional_parameter_on_is_reported() {
        let text = "{\n  return other(x, f);\n}\n";
        assert!(uses_outside_call_position(text, &CodeMask::new(text), "f").is_some());
    }

    #[test]
    fn returning_a_functional_parameter_is_reported() {
        let text = "{\n  return f;\n}\n";
        assert!(uses_outside_call_position(text, &CodeMask::new(text), "f").is_some());
    }

    #[test]
    fn a_mention_in_a_comment_is_not_a_use() {
        let text = "{\n  // f is the callback\n  return f(1);\n}\n";
        assert_eq!(uses_outside_call_position(text, &CodeMask::new(text), "f"), None);
    }

    #[test]
    fn a_longer_identifier_containing_the_name_is_not_a_use() {
        let text = "{\n  real fx = 1;\n  return f(fx);\n}\n";
        assert_eq!(uses_outside_call_position(text, &CodeMask::new(text), "f"), None);
    }
}
