//! Renaming pass + splicing into the `functions{}` block: the core
//! "compiler" step that turns a parsed `.laplace` file plus its resolved
//! packages into final `.stan` text.

pub mod rename;

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::CodeMask;
use crate::parser::library_block::{ImportStatement, LibraryBlock};
use crate::parser::signatures::FunctionSig;
use rename::{find_qualified_calls, mangle, rename_identifier_calls};

/// A resolved, installed package ready for codegen: its raw `.stan` source
/// (every function it defines, exported or private), the signatures Task 2
/// extracted from that source, and which of those are actually exported
/// (from the package's own `laplace.toml`).
#[derive(Debug, Clone)]
pub struct InstalledPackage {
    pub name: String,
    pub source: String,
    pub signatures: Vec<FunctionSig>,
    pub exported: Vec<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CodegenError {
    #[error(
        "`library` imports `{package}`, but no installed package by that name was provided \
         (run `laplace install`?)"
    )]
    MissingImport { package: String },

    #[error("`{package}::{func}(` is called, but `{func}` is not in `{package}`'s exports")]
    FunctionNotExported { package: String, func: String },

    #[error(
        "`{package}::{func}(` is called, but `{package}` was not imported in the `library {{ }}` block"
    )]
    UnknownPackageReference { package: String, func: String },

    #[error(
        "package `{package}` declares export `{func}`, but no such function was found in its source"
    )]
    ExportedFunctionMissing { package: String, func: String },

    #[error(
        "packages `{first_package}` and `{second_package}` both mangle to `{mangled}` -- this \
         should be impossible given the package-name prefix and indicates a naming collision \
         between the two packages"
    )]
    DuplicateMangledName {
        mangled: String,
        first_package: String,
        second_package: String,
    },
}

/// Compile a `.laplace` source file's text into final `.stan` text.
///
/// `library_block` is the parsed `library { }` block, if the source has one
/// (its import list and byte range, from Task 1's `parse_library_block`).
/// `installed` are the resolved packages available for those imports (from
/// the lockfile plus Task 2's signature extraction over each package's
/// source) -- codegen does not itself consult the lockfile or filesystem,
/// and does not re-parse the library block.
///
/// The renamed imported functions are spliced in the order they appear in
/// `installed` (expected to be lockfile order), not the order they're
/// written in the `library { }` block.
pub fn generate(
    source: &str,
    library_block: Option<&LibraryBlock>,
    installed: &[InstalledPackage],
) -> Result<String, CodegenError> {
    let imports: &[ImportStatement] = library_block.map(|b| b.imports.as_slice()).unwrap_or(&[]);

    let mut by_name: BTreeMap<&str, &InstalledPackage> = BTreeMap::new();
    for pkg in installed {
        by_name.insert(pkg.name.as_str(), pkg);
    }

    for import in imports {
        if !by_name.contains_key(import.name.as_str()) {
            return Err(CodegenError::MissingImport {
                package: import.name.clone(),
            });
        }
    }

    for pkg in installed {
        let sig_names: HashSet<&str> = pkg.signatures.iter().map(|s| s.name.as_str()).collect();
        for export in &pkg.exported {
            if !sig_names.contains(export.as_str()) {
                return Err(CodegenError::ExportedFunctionMissing {
                    package: pkg.name.clone(),
                    func: export.clone(),
                });
            }
        }
    }

    // Only the packages this file actually imports, in the order `installed`
    // gives them (the caller's responsibility to supply lockfile order).
    let imported_names: HashSet<&str> = imports.iter().map(|i| i.name.as_str()).collect();
    let ordered_installed: Vec<&InstalledPackage> = installed
        .iter()
        .filter(|p| imported_names.contains(p.name.as_str()))
        .collect();
    let imported_by_name: BTreeMap<&str, &InstalledPackage> = ordered_installed
        .iter()
        .map(|p| (p.name.as_str(), *p))
        .collect();

    let mut mangled_sources: Vec<String> = Vec::new();
    let mut mangled_owner: BTreeMap<String, String> = BTreeMap::new();
    for pkg in &ordered_installed {
        let mut exported_sorted = pkg.exported.clone();
        exported_sorted.sort();

        let mut renamed = pkg.source.clone();
        for export in &exported_sorted {
            let mangled_name = mangle(&pkg.name, export);
            if let Some(owner) = mangled_owner.get(&mangled_name) {
                if owner != &pkg.name {
                    return Err(CodegenError::DuplicateMangledName {
                        mangled: mangled_name,
                        first_package: owner.clone(),
                        second_package: pkg.name.clone(),
                    });
                }
            }
            mangled_owner.insert(mangled_name.clone(), pkg.name.clone());
            renamed = rename_identifier_calls(&renamed, export, &mangled_name);
        }
        mangled_sources.push(renamed);
    }

    let mut edits: Vec<Edit> = Vec::new();

    if let Some(block) = library_block {
        edits.push(Edit {
            range: block.byte_range.clone(),
            replacement: String::new(),
        });
    }

    for call in find_qualified_calls(source) {
        if let Some(block) = library_block {
            if block.byte_range.start <= call.range.start && call.range.end <= block.byte_range.end
            {
                continue;
            }
        }

        let Some(target) = imported_by_name.get(call.package.as_str()) else {
            return Err(CodegenError::UnknownPackageReference {
                package: call.package,
                func: call.func,
            });
        };
        if !target.exported.iter().any(|e| e == &call.func) {
            return Err(CodegenError::FunctionNotExported {
                package: call.package,
                func: call.func,
            });
        }

        edits.push(Edit {
            range: call.range.clone(),
            replacement: mangle(&call.package, &call.func),
        });
    }

    if !mangled_sources.is_empty() {
        let assembled = mangled_sources.join("\n");
        match find_functions_block(source) {
            Some(fb) => edits.push(Edit {
                range: fb.open_brace + 1..fb.open_brace + 1,
                replacement: format!("\n{assembled}\n"),
            }),
            None => edits.push(Edit {
                range: 0..0,
                replacement: format!("functions {{\n{assembled}\n}}\n"),
            }),
        }
    }

    Ok(apply_edits(source, edits))
}

struct Edit {
    range: Range<usize>,
    replacement: String,
}

/// Apply a set of non-overlapping edits (in original-source byte offsets) in
/// one linear pass, so offsets computed up front never need adjusting for
/// earlier edits.
fn apply_edits(source: &str, mut edits: Vec<Edit>) -> String {
    edits.sort_by_key(|e| (e.range.start, e.range.end));

    let mut out = String::with_capacity(source.len());
    let mut cursor = 0usize;
    for edit in edits {
        if edit.range.start < cursor {
            // Overlapping edits shouldn't occur given how callers build
            // them; skip defensively rather than corrupt the output.
            continue;
        }
        out.push_str(&source[cursor..edit.range.start]);
        out.push_str(&edit.replacement);
        cursor = edit.range.end;
    }
    out.push_str(&source[cursor..]);
    out
}

struct FunctionsBlock {
    open_brace: usize,
}

/// Find the user's own `functions { }` block, if present, using the same
/// comment/string-aware brace matching as Task 2's signature scanner.
fn find_functions_block(source: &str) -> Option<FunctionsBlock> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut search_from = 0;

    while let Some(rel) = source[search_from..].find("functions") {
        let start = search_from + rel;
        let end = start + "functions".len();
        let word_ok = mask.is_real(start)
            && (start == 0 || !crate::parser::brace_match::is_ident_char(bytes[start - 1]))
            && (end == bytes.len() || !crate::parser::brace_match::is_ident_char(bytes[end]));

        if word_ok {
            let after = &source[end..];
            let trimmed = after.trim_start();
            if trimmed.starts_with('{') {
                let open_brace = end + (after.len() - trimmed.len());
                if mask.is_real(open_brace) && mask.match_closing_brace(source, open_brace).is_some()
                {
                    return Some(FunctionsBlock { open_brace });
                }
            }
        }

        search_from = end;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::library_block::parse_library_block;
    use crate::parser::signatures::extract_signatures;

    const GPS_SOURCE: &str = r#"// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @param x Vector of input locations.
// @param alpha Marginal standard deviation of the GP.
// @param rho Length-scale of the GP.
// @return An N x N positive semi-definite covariance matrix.
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}
"#;

    fn gps_package() -> InstalledPackage {
        InstalledPackage {
            name: "gps".to_string(),
            source: GPS_SOURCE.to_string(),
            signatures: extract_signatures(GPS_SOURCE),
            exported: vec!["rbf_cov".to_string()],
        }
    }

    #[test]
    fn end_to_end_gps_rbf_cov_example_synthesizes_functions_block() {
        let laplace_source = r#"library {
  import gps@1.0.0
}

data {
  int<lower=1> N;
  vector[N] x;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
}

model {
  matrix[N, N] K = gps::rbf_cov(x, alpha, rho);
  x ~ multi_normal(rep_vector(0, N), K);
}
"#;

        let block = parse_library_block(laplace_source).unwrap();
        let installed = vec![gps_package()];

        let output_1 = generate(laplace_source, block.as_ref(), &installed).unwrap();
        let output_2 = generate(laplace_source, block.as_ref(), &installed).unwrap();
        assert_eq!(output_1, output_2, "codegen must be deterministic");

        assert!(!output_1.contains("library"));
        assert!(!output_1.contains("gps::rbf_cov"));
        assert!(output_1.contains("gps__rbf_cov"));

        // The library block is deleted and replaced by a synthesized
        // `functions { }` block (this .laplace file defines none of its
        // own); everything else is untouched byte-for-byte. Build the
        // expected text from the same byte range Task 1 reported, so this
        // assertion doesn't depend on hand-counting newlines.
        let byte_range = block.unwrap().byte_range;
        let expected_functions_block = format!(
            "functions {{\n{}\n}}\n",
            GPS_SOURCE.replace("rbf_cov", "gps__rbf_cov")
        );
        let expected_tail = format!(
            "{}{}",
            &laplace_source[..byte_range.start],
            &laplace_source[byte_range.end..]
        )
        .replace("gps::rbf_cov", "gps__rbf_cov");
        assert_eq!(output_1, format!("{expected_functions_block}{expected_tail}"));
    }

    #[test]
    fn prepends_into_an_existing_functions_block_in_installed_order() {
        let laplace_source = r#"library {
  import beta
  import alpha
}

functions {
  real user_fn(real x) {
    return x;
  }
}

model {
}
"#;

        let alpha = InstalledPackage {
            name: "alpha".to_string(),
            source: "real a_fn(real x) {\n  return x;\n}\n".to_string(),
            signatures: extract_signatures("real a_fn(real x) {\n  return x;\n}\n"),
            exported: vec!["a_fn".to_string()],
        };
        let beta = InstalledPackage {
            name: "beta".to_string(),
            source: "real b_fn(real x) {\n  return x;\n}\n".to_string(),
            signatures: extract_signatures("real b_fn(real x) {\n  return x;\n}\n"),
            exported: vec!["b_fn".to_string()],
        };

        // `installed` is given in lockfile (alphabetical) order: alpha, beta
        // -- even though the library block imports beta first.
        let block = parse_library_block(laplace_source).unwrap();
        let output = generate(laplace_source, block.as_ref(), &[alpha, beta]).unwrap();

        let alpha_pos = output.find("alpha__a_fn").unwrap();
        let beta_pos = output.find("beta__b_fn").unwrap();
        let user_pos = output.find("user_fn").unwrap();
        assert!(
            alpha_pos < beta_pos && beta_pos < user_pos,
            "expected alpha before beta before the user's own function, got:\n{output}"
        );
    }

    #[test]
    fn no_library_block_and_no_imports_is_a_pure_passthrough() {
        let source = "data {\n  int n;\n}\nmodel {\n}\n";
        let output = generate(source, None, &[]).unwrap();
        assert_eq!(output, source);
    }

    #[test]
    fn empty_library_block_just_gets_deleted() {
        let source = "library {}\ndata {\n  int n;\n}\n";
        let block = parse_library_block(source).unwrap();
        let output = generate(source, block.as_ref(), &[]).unwrap();
        assert_eq!(output, "\ndata {\n  int n;\n}\n");
    }

    #[test]
    fn missing_import_is_an_error() {
        let source = "library {\n  import gps\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::MissingImport {
                package: "gps".to_string()
            }
        );
    }

    #[test]
    fn calling_a_non_exported_function_is_an_error() {
        let source = r#"library {
  import gps
}
model {
  real y = gps::secret_helper(1.0);
}
"#;
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[gps_package()]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::FunctionNotExported {
                package: "gps".to_string(),
                func: "secret_helper".to_string(),
            }
        );
    }

    #[test]
    fn calling_an_unimported_package_is_an_error() {
        let source = "model {\n  real y = gps::rbf_cov(1.0);\n}\n";
        let err = generate(source, None, &[gps_package()]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::UnknownPackageReference {
                package: "gps".to_string(),
                func: "rbf_cov".to_string(),
            }
        );
    }

    #[test]
    fn declared_export_missing_from_source_is_an_error() {
        let pkg = InstalledPackage {
            name: "gps".to_string(),
            source: "real rbf_cov(real x) {\n  return x;\n}\n".to_string(),
            signatures: extract_signatures("real rbf_cov(real x) {\n  return x;\n}\n"),
            exported: vec!["rbf_cov".to_string(), "matern_cov".to_string()],
        };
        let source = "library {\n  import gps\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[pkg]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::ExportedFunctionMissing {
                package: "gps".to_string(),
                func: "matern_cov".to_string(),
            }
        );
    }

    #[test]
    fn duplicate_mangled_name_across_packages_is_an_error() {
        // "a" exporting "b__c" and "a__b" exporting "c" both mangle to
        // "a__b__c" -- a contrived but real collision the sanity check must
        // catch.
        let pkg_a = InstalledPackage {
            name: "a".to_string(),
            source: "real b__c(real x) {\n  return x;\n}\n".to_string(),
            signatures: extract_signatures("real b__c(real x) {\n  return x;\n}\n"),
            exported: vec!["b__c".to_string()],
        };
        let pkg_a_b = InstalledPackage {
            name: "a__b".to_string(),
            source: "real c(real x) {\n  return x;\n}\n".to_string(),
            signatures: extract_signatures("real c(real x) {\n  return x;\n}\n"),
            exported: vec!["c".to_string()],
        };

        let source = "library {\n  import a\n  import a__b\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[pkg_a, pkg_a_b]).unwrap_err();
        assert!(matches!(err, CodegenError::DuplicateMangledName { .. }));
    }

    #[test]
    fn qualified_calls_inside_the_users_functions_block_are_rewritten_too() {
        let source = r#"library {
  import gps
}
functions {
  real wrapper(vector x, real alpha, real rho) {
    return gps::rbf_cov(x, alpha, rho)[1, 1];
  }
}
model {
}
"#;
        let block = parse_library_block(source).unwrap();
        let output = generate(source, block.as_ref(), &[gps_package()]).unwrap();
        assert!(output.contains("return gps__rbf_cov(x, alpha, rho)[1, 1];"));
    }
}
