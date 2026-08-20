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

    #[error("stanc reported errors in {path}:\n{annotated}")]
    TypeCheckFailed { path: PathBuf, annotated: String },
}

/// Run `stanc` (or whatever `command` names) against the just-written file
/// at `path`. `generated` is the codegen result that produced it, used to
/// annotate any error output with likely-culprit packages.
pub fn validate(command: &str, path: &Path, generated: &GeneratedStan) -> Result<(), ValidateError> {
    let output = Command::new(command)
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
}
