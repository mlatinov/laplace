//! Finds the `library { }` block in a `.laplace` source file and extracts its
//! `import` statements. Everything outside the block is left untouched here;
//! codegen (Task 4) is responsible for actually deleting the block from the
//! final output, using the byte range this module reports.

use std::ops::Range;

use thiserror::Error;

/// One `import` statement inside a `library { }` block, e.g. `import gps` or
/// `import gps@1.0.0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportStatement {
    pub name: String,
    pub version: Option<String>,
}

/// A parsed `library { }` block: its imports, and the byte range it occupies
/// in the source (including the `library` keyword and both braces).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryBlock {
    pub imports: Vec<ImportStatement>,
    pub byte_range: Range<usize>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LibraryBlockError {
    #[error("`library {{` block opened at byte {start} is never closed with a matching `}}`")]
    UnclosedBlock { start: usize },

    #[error(
        "malformed import statement `{statement}`: expected `import <name>` or `import <name>@<version>`"
    )]
    MalformedImport { statement: String },

    #[error("import statement `{statement}` has no package name")]
    EmptyPackageName { statement: String },
}

/// Find the `library { }` block in `source`, if present, and parse its imports.
///
/// Returns `Ok(None)` if the source contains no `library { ... }` block at all
/// (e.g. the bare word `library` appears in a comment but is never followed by
/// a brace). Does not inspect anything outside the block.
pub fn parse_library_block(source: &str) -> Result<Option<LibraryBlock>, LibraryBlockError> {
    let Some((keyword_start, open_brace)) = find_library_block_header(source) else {
        return Ok(None);
    };

    let close_brace = find_matching_brace(source, open_brace).ok_or(
        LibraryBlockError::UnclosedBlock {
            start: keyword_start,
        },
    )?;

    let body = &source[open_brace + 1..close_brace];
    let imports = parse_imports(body)?;

    Ok(Some(LibraryBlock {
        imports,
        byte_range: keyword_start..close_brace + 1,
    }))
}

/// Locate a `library` keyword (as a whole word) that is followed, possibly
/// after whitespace, by `{`. Returns `(keyword_start, open_brace_index)`.
fn find_library_block_header(source: &str) -> Option<(usize, usize)> {
    let bytes = source.as_bytes();
    let mut search_from = 0;
    while let Some(rel) = source[search_from..].find("library") {
        let start = search_from + rel;
        let end = start + "library".len();
        let is_whole_word = (start == 0 || !is_ident_char(bytes[start - 1]))
            && (end == bytes.len() || !is_ident_char(bytes[end]));

        if is_whole_word {
            let after = &source[end..];
            let trimmed = after.trim_start();
            if trimmed.starts_with('{') {
                let open_brace = end + (after.len() - trimmed.len());
                return Some((start, open_brace));
            }
        }

        search_from = end;
    }
    None
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Given the index of an opening `{`, find the index of its matching `}`.
fn find_matching_brace(source: &str, open_brace: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in source[open_brace..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_brace + i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_imports(body: &str) -> Result<Vec<ImportStatement>, LibraryBlockError> {
    let mut imports = Vec::new();
    for statement in split_statements(body) {
        let statement = statement.trim();
        if statement.is_empty() {
            continue;
        }
        imports.push(parse_one_import(statement)?);
    }
    Ok(imports)
}

/// Split the block body into individual statements on line comments,
/// newlines, and semicolons (imports may be one-per-line or `;`-separated).
fn split_statements(body: &str) -> Vec<String> {
    body.lines()
        .map(strip_line_comment)
        .flat_map(|line| line.split(';').map(str::to_string).collect::<Vec<_>>())
        .collect()
}

fn strip_line_comment(line: &str) -> String {
    match line.find("//") {
        Some(idx) => line[..idx].to_string(),
        None => line.to_string(),
    }
}

fn parse_one_import(statement: &str) -> Result<ImportStatement, LibraryBlockError> {
    let Some(rest) = statement.strip_prefix("import") else {
        return Err(LibraryBlockError::MalformedImport {
            statement: statement.to_string(),
        });
    };

    // Reject e.g. `importgps`, where "import" only matched as a prefix of a
    // longer identifier rather than as the keyword.
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return Err(LibraryBlockError::MalformedImport {
            statement: statement.to_string(),
        });
    }

    let rest = rest.trim();
    if rest.is_empty() {
        return Err(LibraryBlockError::EmptyPackageName {
            statement: statement.to_string(),
        });
    }

    let (name, version) = match rest.split_once('@') {
        Some((name, version)) => (name.trim(), Some(version.trim())),
        None => (rest, None),
    };

    if name.is_empty() {
        return Err(LibraryBlockError::EmptyPackageName {
            statement: statement.to_string(),
        });
    }
    if !is_valid_package_name(name) {
        return Err(LibraryBlockError::MalformedImport {
            statement: statement.to_string(),
        });
    }
    if let Some(version) = version {
        if version.is_empty() || !is_valid_version(version) {
            return Err(LibraryBlockError::MalformedImport {
                statement: statement.to_string(),
            });
        }
    }

    Ok(ImportStatement {
        name: name.to_string(),
        version: version.map(str::to_string),
    })
}

fn is_valid_package_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn is_valid_version(version: &str) -> bool {
    !version.is_empty()
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_library_block_returns_none() {
        let source = "data {\n  int n;\n}\nmodel {\n}\n";
        assert_eq!(parse_library_block(source).unwrap(), None);
    }

    #[test]
    fn library_word_in_comment_without_brace_is_ignored() {
        let source = "// library note: nothing here\ndata {}\n";
        assert_eq!(parse_library_block(source).unwrap(), None);
    }

    #[test]
    fn identifier_containing_library_is_not_matched() {
        let source = "mylibrary { import gps }\n";
        assert_eq!(parse_library_block(source).unwrap(), None);
    }

    #[test]
    fn empty_library_block() {
        let source = "library {}\ndata {}\n";
        let block = parse_library_block(source).unwrap().unwrap();
        assert!(block.imports.is_empty());
        assert_eq!(&source[block.byte_range.clone()], "library {}");
    }

    #[test]
    fn multiple_imports() {
        let source = "library {\n  import gps\n  import splines\n  import utils\n}\n";
        let block = parse_library_block(source).unwrap().unwrap();
        assert_eq!(
            block.imports,
            vec![
                ImportStatement {
                    name: "gps".into(),
                    version: None
                },
                ImportStatement {
                    name: "splines".into(),
                    version: None
                },
                ImportStatement {
                    name: "utils".into(),
                    version: None
                },
            ]
        );
    }

    #[test]
    fn pinned_and_unpinned_mixed() {
        let source =
            "library {\n  import gps@1.0.0\n  import utils\n  import splines@0.3.1\n}\n";
        let block = parse_library_block(source).unwrap().unwrap();
        assert_eq!(
            block.imports,
            vec![
                ImportStatement {
                    name: "gps".into(),
                    version: Some("1.0.0".into())
                },
                ImportStatement {
                    name: "utils".into(),
                    version: None
                },
                ImportStatement {
                    name: "splines".into(),
                    version: Some("0.3.1".into())
                },
            ]
        );
    }

    #[test]
    fn semicolon_separated_imports_on_one_line() {
        let source = "library { import gps@1.0.0; import utils; }\n";
        let block = parse_library_block(source).unwrap().unwrap();
        assert_eq!(
            block.imports,
            vec![
                ImportStatement {
                    name: "gps".into(),
                    version: Some("1.0.0".into())
                },
                ImportStatement {
                    name: "utils".into(),
                    version: None
                },
            ]
        );
    }

    #[test]
    fn comment_inside_library_block_is_stripped() {
        let source = "library {\n  import gps // pinned later\n  import utils\n}\n";
        let block = parse_library_block(source).unwrap().unwrap();
        assert_eq!(
            block.imports,
            vec![
                ImportStatement {
                    name: "gps".into(),
                    version: None
                },
                ImportStatement {
                    name: "utils".into(),
                    version: None
                },
            ]
        );
    }

    #[test]
    fn byte_range_covers_full_block_for_deletion() {
        let source = "functions {}\nlibrary {\n  import gps\n}\ndata {}\n";
        let block = parse_library_block(source).unwrap().unwrap();
        assert_eq!(
            &source[block.byte_range.clone()],
            "library {\n  import gps\n}"
        );
    }

    #[test]
    fn missing_closing_brace_is_an_error() {
        let source = "library {\n  import gps\ndata {\n  int n;\n}\n";
        let err = parse_library_block(source).unwrap_err();
        assert_eq!(err, LibraryBlockError::UnclosedBlock { start: 0 });
    }

    #[test]
    fn malformed_import_missing_import_keyword() {
        let source = "library {\n  gps\n}\n";
        let err = parse_library_block(source).unwrap_err();
        assert_eq!(
            err,
            LibraryBlockError::MalformedImport {
                statement: "gps".to_string()
            }
        );
    }

    #[test]
    fn malformed_import_concatenated_keyword() {
        let source = "library {\n  importgps\n}\n";
        let err = parse_library_block(source).unwrap_err();
        assert_eq!(
            err,
            LibraryBlockError::MalformedImport {
                statement: "importgps".to_string()
            }
        );
    }

    #[test]
    fn malformed_import_empty_package_name() {
        let source = "library {\n  import\n}\n";
        let err = parse_library_block(source).unwrap_err();
        assert_eq!(
            err,
            LibraryBlockError::EmptyPackageName {
                statement: "import".to_string()
            }
        );
    }

    #[test]
    fn malformed_import_empty_version() {
        let source = "library {\n  import gps@\n}\n";
        let err = parse_library_block(source).unwrap_err();
        assert_eq!(
            err,
            LibraryBlockError::MalformedImport {
                statement: "import gps@".to_string()
            }
        );
    }
}
