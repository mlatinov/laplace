//! `__` is reserved for laplace's generated names.
//!
//! laplace mangles every library item into `pkg__item`, and later patches
//! generate further names the same way (`apply_twice__add_one`). A
//! hand-written identifier containing `__` could collide with one of
//! those, so `.laplace` and `.laplacelib` sources may not use it anywhere
//! -- which makes "generated" and "hand-written" name spaces provably
//! disjoint rather than merely unlikely to overlap.
//!
//! Plain `.stan` files inside a package are *not* checked: they are
//! ordinary Stan, passed through verbatim, and were legal before this rule
//! existed.

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};

/// The byte sequence reserved for generated names.
pub const RESERVED_SEPARATOR: &str = "__";

#[derive(Debug, Error, PartialEq, Eq)]
#[error("`{identifier}` contains `__`, which laplace reserves for generated names")]
pub struct ReservedIdentifier {
    pub identifier: String,
    /// Byte offset of the identifier's first byte.
    pub offset: usize,
}

impl ReservedIdentifier {
    /// The `help:` line to print under the error.
    pub fn help(&self) -> String {
        format!(
            "rename it with a single underscore (`{}`) -- laplace builds its own names with \
             `__` (`pkg::func` compiles to `pkg__func`), so hand-written names may not use it",
            self.identifier.replace(RESERVED_SEPARATOR, "_"),
        )
    }
}

/// Reject the first identifier in `source` that contains `__`.
///
/// Comments and string literals are skipped: prose and `print()` output
/// are not identifiers. Only whole identifier tokens are considered, so
/// this never fires on punctuation that happens to sit next to an
/// underscore.
pub fn check_identifiers(source: &str) -> Result<(), ReservedIdentifier> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        if !mask.is_real(i) || !is_ident_char(bytes[i]) {
            i += 1;
            continue;
        }
        // Walk the whole token, so `a__b` is reported once and a token
        // like `datastore` can't re-match at an inner offset.
        let start = i;
        let mut end = i;
        while end < bytes.len() && is_ident_char(bytes[end]) && mask.is_real(end) {
            end += 1;
        }
        i = end;

        let token = &source[start..end];
        // A leading digit means this is a numeric literal, not a name.
        if token.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        if token.contains(RESERVED_SEPARATOR) {
            return Err(ReservedIdentifier {
                identifier: token.to_string(),
                offset: start,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_source_file_passes() {
        let source = "data {\n  int<lower=1> N;\n  vector[N] y_obs;\n}\nmodel {\n  y_obs ~ normal(0, 1);\n}\n";
        assert_eq!(check_identifiers(source), Ok(()));
    }

    #[test]
    fn single_underscores_are_fine_anywhere() {
        let source = "real _leading(real trailing_, real in_the_middle) {\n  return trailing_;\n}\n";
        assert_eq!(check_identifiers(source), Ok(()));
    }

    #[test]
    fn a_double_underscore_in_a_declaration_is_rejected() {
        let source = "parameters {\n  real my__theta;\n}\n";
        let err = check_identifiers(source).unwrap_err();
        assert_eq!(err.identifier, "my__theta");
        assert_eq!(&source[err.offset..err.offset + 9], "my__theta");
    }

    #[test]
    fn a_double_underscore_in_a_function_name_is_rejected() {
        let err = check_identifiers("real a__b(real x) {\n  return x;\n}\n").unwrap_err();
        assert_eq!(err.identifier, "a__b");
    }

    #[test]
    fn a_trailing_or_leading_double_underscore_is_rejected() {
        assert_eq!(check_identifiers("real x__;\n").unwrap_err().identifier, "x__");
        assert_eq!(check_identifiers("real __x;\n").unwrap_err().identifier, "__x");
    }

    #[test]
    fn an_import_name_in_a_library_block_is_checked_too() {
        let err = check_identifiers("library {\n  import my__pkg\n}\n").unwrap_err();
        assert_eq!(err.identifier, "my__pkg");
    }

    #[test]
    fn double_underscores_in_comments_are_ignored() {
        let source = "// pkg__func is what this compiles to\nreal f() {\n  return 1;\n}\n";
        assert_eq!(check_identifiers(source), Ok(()));
    }

    #[test]
    fn double_underscores_in_string_literals_are_ignored() {
        let source = "model {\n  print(\"calling pkg__func\");\n}\n";
        assert_eq!(check_identifiers(source), Ok(()));
    }

    #[test]
    fn the_first_offender_is_the_one_reported() {
        let source = "real a__b() {\n  return c__d();\n}\n";
        assert_eq!(check_identifiers(source).unwrap_err().identifier, "a__b");
    }

    #[test]
    fn the_help_text_suggests_a_single_underscore() {
        let err = check_identifiers("real my__theta;\n").unwrap_err();
        assert!(err.help().contains("my_theta"), "{}", err.help());
    }

    #[test]
    fn a_qualified_call_is_still_two_identifiers() {
        // `stats::mean_` is fine; the `__` only appears after mangling.
        assert_eq!(check_identifiers("real x = stats::mean_(y);\n"), Ok(()));
    }
}
