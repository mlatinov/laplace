//! `pkg::func` -> `pkg__func` mangling and call-site rewriting.

use std::ops::Range;

use crate::parser::brace_match::{is_ident_char, CodeMask};

/// Mangle a package-qualified function name into its Stan-safe form.
pub fn mangle(package: &str, func: &str) -> String {
    format!("{package}__{func}")
}

/// Replace every whole-word occurrence of `name` immediately followed by
/// (optional whitespace then) `(` with `replacement`, leaving everything
/// else -- including the `(` itself and any whitespace before it -- byte-for-
/// byte untouched. Used to mangle a package's own exported function, both at
/// its definition and at any call-sites within the package's own source.
///
/// Ignores occurrences inside `//` comments or `"..."` string literals.
pub fn rename_identifier_calls(source: &str, name: &str, replacement: &str) -> String {
    if name.is_empty() {
        return source.to_string();
    }

    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let name_bytes = name.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut cursor = 0usize;
    let mut i = 0usize;

    while i + name_bytes.len() <= bytes.len() {
        if mask.is_real(i) && &bytes[i..i + name_bytes.len()] == name_bytes {
            let start = i;
            let end = i + name_bytes.len();
            let before_ok = start == 0 || !is_ident_char(bytes[start - 1]);
            let after_ok = end == bytes.len() || !is_ident_char(bytes[end]);
            if before_ok && after_ok {
                let rest = &source[end..];
                if rest.trim_start().starts_with('(') {
                    out.push_str(&source[cursor..start]);
                    out.push_str(replacement);
                    cursor = end;
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }

    out.push_str(&source[cursor..]);
    out
}

/// One `package::func(` call site found in a source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedCall {
    pub package: String,
    pub func: String,
    /// Byte range of the `package::func` text itself (not including the `(`).
    pub range: Range<usize>,
}

/// Find every `identifier::identifier(` occurrence in `source`, ignoring
/// matches inside `//` comments or `"..."` string literals.
pub fn find_qualified_calls(source: &str) -> Vec<QualifiedCall> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i + 1 < bytes.len() {
        if mask.is_real(i) && bytes[i] == b':' && bytes.get(i + 1) == Some(&b':') {
            let pkg_end = i;
            let mut pkg_start = i;
            while pkg_start > 0
                && is_ident_char(bytes[pkg_start - 1])
                && mask.is_real(pkg_start - 1)
            {
                pkg_start -= 1;
            }

            let func_start = i + 2;
            let mut func_end = func_start;
            while func_end < bytes.len() && is_ident_char(bytes[func_end]) && mask.is_real(func_end)
            {
                func_end += 1;
            }

            if pkg_start < pkg_end && func_start < func_end {
                let after = &source[func_end..];
                if after.trim_start().starts_with('(') {
                    out.push(QualifiedCall {
                        package: source[pkg_start..pkg_end].to_string(),
                        func: source[func_start..func_end].to_string(),
                        range: pkg_start..func_end,
                    });
                    i = func_end;
                    continue;
                }
            }
        }
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mangle_joins_with_double_underscore() {
        assert_eq!(mangle("gps", "rbf_cov"), "gps__rbf_cov");
    }

    #[test]
    fn renames_definition_and_internal_call_site() {
        let source = "matrix rbf_cov(vector x) {\n  return rbf_cov(x);\n}\n";
        let renamed = rename_identifier_calls(source, "rbf_cov", "gps__rbf_cov");
        assert_eq!(
            renamed,
            "matrix gps__rbf_cov(vector x) {\n  return gps__rbf_cov(x);\n}\n"
        );
    }

    #[test]
    fn does_not_rename_substring_matches() {
        let source = "real rbf_cov_jittered(real x) {\n  return rbf_cov_jittered_helper(x);\n}\n";
        let renamed = rename_identifier_calls(source, "rbf_cov", "gps__rbf_cov");
        assert_eq!(renamed, source);
    }

    #[test]
    fn does_not_rename_without_trailing_paren() {
        let source = "real x = rbf_cov + 1;\n";
        let renamed = rename_identifier_calls(source, "rbf_cov", "gps__rbf_cov");
        assert_eq!(renamed, source);
    }

    #[test]
    fn ignores_occurrences_in_comments_and_strings() {
        let source = "// calls rbf_cov(x) in a comment\nreal y = 1; // print(\"rbf_cov(x)\")\n";
        let renamed = rename_identifier_calls(source, "rbf_cov", "gps__rbf_cov");
        assert_eq!(renamed, source);
    }

    #[test]
    fn find_qualified_calls_extracts_package_and_func() {
        let source = "matrix K = gps::rbf_cov(x, alpha, rho);\n";
        let calls = find_qualified_calls(source);
        assert_eq!(
            calls,
            vec![QualifiedCall {
                package: "gps".to_string(),
                func: "rbf_cov".to_string(),
                range: 11..23,
            }]
        );
        assert_eq!(&source[calls[0].range.clone()], "gps::rbf_cov");
    }

    #[test]
    fn find_qualified_calls_ignores_non_call_double_colons() {
        // No trailing `(` -- not a call, shouldn't match.
        let source = "gps::rbf_cov;\n";
        assert_eq!(find_qualified_calls(source), vec![]);
    }

    #[test]
    fn find_qualified_calls_ignores_comments_and_strings() {
        let source = "// gps::rbf_cov(x)\nreal y = 1; // \"gps::rbf_cov(x)\"\n";
        assert_eq!(find_qualified_calls(source), vec![]);
    }

    #[test]
    fn find_qualified_calls_finds_multiple_in_order() {
        let source = "gps::rbf_cov(x); splines::basis(x);\n";
        let calls = find_qualified_calls(source);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].package, "gps");
        assert_eq!(calls[1].package, "splines");
    }
}
