//! Optional `stanc` type-checking pass, gated behind `--validate` on
//! `laplace build`. Off by default so a normal build never requires `stanc`
//! to be installed.
//!
//! On failure, best-effort-annotates `stanc`'s raw error output with which
//! imported package(s) contributed code near the failing line(s), using the
//! splice boundaries codegen tracked (see `codegen::GeneratedStan`). Exact
//! source-mapping back to the original `.laplace` file is out of scope --
//! this only says "this range of the compiled output came from package X".

use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

use crate::codegen::GeneratedStan;

#[derive(Debug, Error)]
pub enum ValidateError {
    #[error("`{command}` was not found on PATH -- install cmdstan/stanc, or omit --validate: {source}")]
    CommandNotFound {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("could not create a scratch directory for stanc's generated C++: {0}")]
    Scratch(#[source] std::io::Error),

    #[error("stanc reported errors in {path}:\n{annotated}")]
    TypeCheckFailed { path: PathBuf, annotated: String },
}

/// Run `stanc` (or whatever `command` names) against the just-written file
/// at `path`. `generated` is the codegen result that produced it, used to
/// annotate any error output with likely-culprit packages.
///
/// This is a pure type-check: `stanc`'s generated C++ goes to a scratch
/// directory that is deleted afterwards, never next to the `.stan` file the
/// user commits. The output file's own directory is passed as an include
/// path, because `stanc` does not resolve `#include` relative to the
/// including file -- without it a `--split-functions` build could never
/// validate.
pub fn validate(command: &str, path: &Path, generated: &GeneratedStan) -> Result<(), ValidateError> {
    let include_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let scratch = tempfile::tempdir().map_err(ValidateError::Scratch)?;
    let output = Command::new(command)
        .arg(format!("--include-paths={}", include_dir.display()))
        .arg(format!("--o={}", scratch.path().join("model.hpp").display()))
        .arg(path)
        .output()
        .map_err(|source| ValidateError::CommandNotFound {
            command: command.to_string(),
            source,
        })?;

    if output.status.success() {
        return Ok(());
    }

    let mut raw = String::from_utf8_lossy(&output.stderr).into_owned();
    if raw.trim().is_empty() {
        raw = String::from_utf8_lossy(&output.stdout).into_owned();
    }

    Err(ValidateError::TypeCheckFailed {
        path: path.to_path_buf(),
        annotated: annotate_with_packages(&raw, &generated.package_line_ranges),
    })
}

fn annotate_with_packages(
    raw: &str,
    package_line_ranges: &[crate::codegen::PackageLineRange],
) -> String {
    let error_lines = extract_line_numbers(raw);
    let notes: Vec<String> = package_line_ranges
        .iter()
        .filter(|range| error_lines.iter().any(|n| range.lines.contains(n)))
        .map(|range| {
            format!(
                "note: lines {}-{} of the compiled output came from package `{}`",
                range.lines.start(),
                range.lines.end(),
                range.package
            )
        })
        .collect();

    if notes.is_empty() {
        raw.to_string()
    } else {
        format!("{}\n{raw}", notes.join("\n"))
    }
}

/// One error or warning parsed out of `stanc`'s human-readable output --
/// used by the LSP's live `stanc` pass to turn `stanc`'s text into LSP
/// diagnostics (`--validate`'s own CLI usage above just prints the raw,
/// package-annotated text and doesn't need this).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StancDiagnostic {
    pub severity: StancSeverity,
    /// 1-indexed line number in the `.stan` text `stanc` was run against.
    pub line: usize,
    /// 0-indexed, half-open `[column_start, column_end)` column range on
    /// that line -- this is `stanc`'s own convention, which happens to
    /// match the LSP spec's.
    pub column_start: usize,
    pub column_end: usize,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StancSeverity {
    Error,
    Warning,
}

/// Parse `stanc`'s output into structured diagnostics. There's no
/// machine-readable output format, so this matches the pretty-printed
/// `<Kind> error in '<file>', line L, column C1 to column C2[, ...]:`
/// header `stanc` prints before each error/warning, followed by a
/// dash-bordered source snippet and then the message body. Best-effort: a
/// header that doesn't parse cleanly is skipped rather than panicking, and
/// an unrecognized severity keyword defaults to `Error`.
pub fn parse_stanc_output(raw: &str) -> Vec<StancDiagnostic> {
    // Every header line contains this marker right after the quoted
    // filename, regardless of whether it's a "Syntax error", "Semantic
    // error", or "Warning" (and regardless of any `, parsing error`/`,
    // lexing error` suffix) -- so it's what block boundaries are found by.
    const MARKER: &str = "', line ";
    const DASHES_MIN_LEN: usize = 5;

    let header_starts: Vec<usize> = {
        let mut starts = Vec::new();
        let mut search_from = 0;
        while let Some(rel) = raw[search_from..].find(MARKER) {
            let marker_at = search_from + rel;
            let line_start = raw[..marker_at].rfind('\n').map(|i| i + 1).unwrap_or(0);
            starts.push(line_start);
            search_from = marker_at + MARKER.len();
        }
        starts
    };

    let mut out = Vec::new();
    for (i, &block_start) in header_starts.iter().enumerate() {
        let block_end = header_starts.get(i + 1).copied().unwrap_or(raw.len());
        let block = &raw[block_start..block_end];

        let header_end = block.find('\n').unwrap_or(block.len());
        let header = &block[..header_end];

        let severity = if header.to_ascii_lowercase().contains("warning") {
            StancSeverity::Warning
        } else {
            StancSeverity::Error
        };

        let Some(line) = number_after(header, "line ") else {
            continue;
        };
        let Some(column_start) = number_after(header, "column ") else {
            continue;
        };
        let column_end = match number_after_last(header, "column ") {
            Some(end) if end > column_start => end,
            _ => column_start + 1,
        };

        let message = message_body(&block[header_end..], DASHES_MIN_LEN);

        out.push(StancDiagnostic {
            severity,
            line,
            column_start,
            column_end,
            message,
        });
    }

    out
}

/// The message body of one diagnostic block: everything after the
/// dash-bordered source snippet (two lines of five or more `-` characters,
/// possibly indented), trimmed. Falls back to the whole block, trimmed, if
/// the snippet's borders aren't found (a future `stanc` output-format
/// change, say) so a diagnostic is still produced rather than dropped.
fn message_body(after_header: &str, dashes_min_len: usize) -> String {
    let is_dashes_line = |line: &str| {
        let t = line.trim();
        t.len() >= dashes_min_len && t.bytes().all(|b| b == b'-')
    };
    let mut dashes_seen = 0;
    for (i, line) in after_header.split('\n').enumerate() {
        if is_dashes_line(line) {
            dashes_seen += 1;
            if dashes_seen == 2 {
                let rest: String = after_header
                    .split('\n')
                    .skip(i + 1)
                    .collect::<Vec<_>>()
                    .join("\n");
                return rest.trim().to_string();
            }
        }
    }
    after_header.trim().to_string()
}

/// The value of the first run of ASCII digits immediately following the
/// first occurrence of `needle` in `text`, if any.
fn number_after(text: &str, needle: &str) -> Option<usize> {
    let start = text.find(needle)? + needle.len();
    parse_digits_at(text, start)
}

/// Same as [`number_after`], but for the *last* occurrence of `needle` --
/// used to get a header's end column when it reads `column C1 to column
/// C2`, distinct from the start column at the first occurrence.
fn number_after_last(text: &str, needle: &str) -> Option<usize> {
    let start = text.rfind(needle)? + needle.len();
    parse_digits_at(text, start)
}

fn parse_digits_at(text: &str, start: usize) -> Option<usize> {
    let end = text[start..]
        .find(|c: char| !c.is_ascii_digit())
        .map(|i| start + i)
        .unwrap_or(text.len());
    if end == start {
        return None;
    }
    text[start..end].parse().ok()
}

/// Best-effort scan for `line <N>` occurrences in stanc's (format-unstable)
/// error text.
fn extract_line_numbers(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(rel) = text[i..].find("line ") {
        let start = i + rel + "line ".len();
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start {
            if let Ok(n) = text[start..end].parse::<usize>() {
                out.push(n);
            }
        }
        i = end.max(start + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::PackageLineRange;

    #[test]
    fn extract_line_numbers_finds_all_occurrences() {
        let text = "Syntax error in 'model.stan', line 12, column 4\nsomething at line 30 too";
        assert_eq!(extract_line_numbers(text), vec![12, 30]);
    }

    #[test]
    fn extract_line_numbers_ignores_non_numeric_trailing_text() {
        assert_eq!(
            extract_line_numbers("no numbers here, just line noise"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn annotate_adds_a_note_when_an_error_line_falls_in_a_package_range() {
        let raw = "error: something wrong, line 5, column 1";
        let ranges = vec![PackageLineRange {
            package: "gps".to_string(),
            lines: 3..=7,
        }];
        let annotated = annotate_with_packages(raw, &ranges);
        assert!(annotated.starts_with("note: lines 3-7 of the compiled output came from package `gps`"));
        assert!(annotated.ends_with(raw));
    }

    #[test]
    fn annotate_is_a_no_op_when_no_range_matches() {
        let raw = "error: something wrong, line 100";
        let ranges = vec![PackageLineRange {
            package: "gps".to_string(),
            lines: 3..=7,
        }];
        assert_eq!(annotate_with_packages(raw, &ranges), raw);
    }

    #[test]
    fn annotate_with_no_package_ranges_is_a_no_op() {
        let raw = "error: line 5";
        assert_eq!(annotate_with_packages(raw, &[]), raw);
    }

    #[test]
    fn validate_reports_a_clear_error_when_the_command_is_missing() {
        let generated = GeneratedStan {
            source: String::new(),
            package_line_ranges: vec![],
            function_files: vec![],
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let err = validate(
            "laplace-test-definitely-not-a-real-command",
            tmp.path(),
            &generated,
        )
        .unwrap_err();
        assert!(matches!(err, ValidateError::CommandNotFound { .. }));
    }

    // The following `parse_stanc_output` fixtures are verbatim captures of
    // real `stanc3 v2.39.0` output (`stanc --filename-in-msg=model.stan
    // <file>`), not hand-written approximations -- the exact formatting
    // (indentation, dash-border width, trailing blank line) is what the
    // parser has to cope with.

    const MISSING_SEMICOLON: &str = "Syntax error in 'model.stan', line 7, column 2 to column 3, parsing error:
   -------------------------------------------------
     5:  model {
     6:    real y = x[1]
     7:    y ~ normal(0, 1);
           ^
     8:  }
   -------------------------------------------------

Ill-formed expression. Unexpected input after the conclusion of a valid expression.
You may be missing a \",\" between expressions, an operator, or a terminating \"}\", \")\", \"]\", or \";\".
";

    const TYPE_MISMATCH: &str = "Semantic error in 'model.stan', line 7, column 11 to column 16:
   -------------------------------------------------
     5:  model {
     6:    matrix[N,N] K;
     7:    real y = x + K;
                    ^
     8:    y ~ normal(0, 1);
     9:  }
   -------------------------------------------------

Ill-typed arguments supplied to infix operator +. Available signatures:
(int, int) => int
Instead supplied arguments of incompatible type: vector, matrix.
";

    const INVALID_TYPE: &str = "Syntax error in 'model.stan', line 2, column 2 to column 9, parsing error:
   -------------------------------------------------
     1:  data {
     2:    integer N;
           ^
     3:  }
   -------------------------------------------------

Invalid type in declaration. Valid types:
  int, real, vector, row_vector, matrix,
optionally preceded by a single array[...]
";

    #[test]
    fn parses_a_missing_semicolon_syntax_error() {
        let diags = parse_stanc_output(MISSING_SEMICOLON);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.severity, StancSeverity::Error);
        assert_eq!(d.line, 7);
        assert_eq!((d.column_start, d.column_end), (2, 3));
        assert!(d.message.starts_with("Ill-formed expression."));
        assert!(!d.message.contains("------"), "dash border leaked into message: {}", d.message);
    }

    #[test]
    fn parses_a_semantic_type_mismatch() {
        let diags = parse_stanc_output(TYPE_MISMATCH);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.severity, StancSeverity::Error);
        assert_eq!(d.line, 7);
        assert_eq!((d.column_start, d.column_end), (11, 16));
        assert!(d.message.contains("Ill-typed arguments"));
        assert!(d.message.contains("vector, matrix"));
    }

    #[test]
    fn parses_an_invalid_type_declaration() {
        let diags = parse_stanc_output(INVALID_TYPE);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.line, 2);
        assert_eq!((d.column_start, d.column_end), (2, 9));
        assert!(d.message.starts_with("Invalid type in declaration."));
    }

    #[test]
    fn parses_multiple_blocks_in_one_run() {
        let combined = format!("{MISSING_SEMICOLON}{TYPE_MISMATCH}");
        let diags = parse_stanc_output(&combined);
        assert_eq!(diags.len(), 2);
        assert_eq!(diags[0].line, 7);
        assert!(diags[0].message.starts_with("Ill-formed expression."));
        assert_eq!(diags[1].line, 7);
        assert!(diags[1].message.contains("Ill-typed arguments"));
    }

    #[test]
    fn parses_a_warning_with_no_dash_bordered_snippet() {
        // Captured from `stanc --warn-uninitialized`: warnings, unlike
        // errors, don't include the dash-bordered source snippet at all.
        let raw = "Warning in 'model.stan', line 6, column 2 to column 3:
    The variable y may not have been assigned a value before its first use.
";
        let diags = parse_stanc_output(raw);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(d.severity, StancSeverity::Warning);
        assert_eq!(d.line, 6);
        assert_eq!((d.column_start, d.column_end), (2, 3));
        assert_eq!(
            d.message,
            "The variable y may not have been assigned a value before its first use."
        );
    }

    #[test]
    fn empty_output_parses_to_no_diagnostics() {
        assert_eq!(parse_stanc_output(""), Vec::new());
        assert_eq!(parse_stanc_output("some unrelated text\n"), Vec::new());
    }
}
