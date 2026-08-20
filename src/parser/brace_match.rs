//! Shared comment/string-aware brace matching, used anywhere a shallow scan
//! needs to find a top-level `{ ... }` block without being confused by a
//! stray brace inside a `//` comment or a `"..."` string literal.
//!
//! Used by both `signatures.rs` (finding function bodies) and `codegen`
//! (finding the `functions { }` block and `pkg::func(` call sites).

/// Marks which byte offsets in a source string are "real code" as opposed to
/// inside a `//` line comment or a `"..."` string literal.
///
/// Block comments (`/* ... */`) are not handled -- the `// @laplace` doc
/// comment convention this tool cares about is line-comment-only, and Stan
/// source in the wild overwhelmingly uses `//` as well.
pub(crate) struct CodeMask(Vec<bool>);

impl CodeMask {
    pub(crate) fn new(source: &str) -> Self {
        let bytes = source.as_bytes();
        let mut mask = vec![true; bytes.len()];
        let mut in_line_comment = false;
        let mut in_string = false;
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if in_line_comment {
                mask[i] = false;
                if b == b'\n' {
                    in_line_comment = false;
                }
                i += 1;
            } else if in_string {
                mask[i] = false;
                if b == b'\\' && i + 1 < bytes.len() {
                    mask[i + 1] = false;
                    i += 2;
                    continue;
                }
                if b == b'"' {
                    in_string = false;
                }
                i += 1;
            } else if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
                in_line_comment = true;
                mask[i] = false;
                i += 1;
            } else if b == b'"' {
                in_string = true;
                mask[i] = false;
                i += 1;
            } else {
                i += 1;
            }
        }
        CodeMask(mask)
    }

    pub(crate) fn is_real(&self, index: usize) -> bool {
        self.0[index]
    }

    pub(crate) fn find_real(&self, source: &str, from: usize, target: u8) -> Option<usize> {
        let bytes = source.as_bytes();
        (from..bytes.len()).find(|&i| bytes[i] == target && self.0[i])
    }

    pub(crate) fn match_closing_brace(&self, source: &str, open_brace: usize) -> Option<usize> {
        let bytes = source.as_bytes();
        let mut depth = 0i32;
        for (i, &b) in bytes.iter().enumerate().skip(open_brace) {
            if !self.0[i] {
                continue;
            }
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }
}

pub(crate) fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_real_skips_comments_and_strings() {
        let source = "\"{\" // { \n{}";
        let mask = CodeMask::new(source);
        let idx = mask.find_real(source, 0, b'{').unwrap();
        assert_eq!(idx, source.rfind('{').unwrap());
    }

    #[test]
    fn match_closing_brace_ignores_braces_in_comments_and_strings() {
        let source = "{ \"}\" // }\n}";
        let mask = CodeMask::new(source);
        let close = mask.match_closing_brace(source, 0).unwrap();
        assert_eq!(close, source.len() - 1);
    }
}
