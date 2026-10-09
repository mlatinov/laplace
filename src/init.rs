//! `laplace init`: scan a package repo's `.stan` and `.laplacelib` files and
//! generate a starter `laplace.toml`, guessing `exports` from
//! `// @laplace`-documented functions -- undocumented functions are assumed
//! private and left out. `laplace init --update` syncs an existing manifest
//! with the sources instead, without disturbing anything already in it.
//!
//! `exports` only ever names functions from plain `.stan` files. A
//! `.laplacelib` item's visibility comes from `pub` in the source, and
//! listing one in `exports` is an error, so `init` reports those items
//! instead of writing a manifest that would not load -- and a package with
//! only `.laplacelib` sources gets no `exports` key at all.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;
use toml_edit::{Array, DocumentMut, Item, Value};

use crate::package::{self, PackageError};
use crate::parser::signatures::extract_signatures;

#[derive(Debug, Error)]
pub enum InitError {
    // The bare file name: `init` only ever runs on the current directory.
    #[error("laplace.toml already exists -- run `laplace init --update` to sync it")]
    AlreadyExists(PathBuf),

    #[error("laplace.toml does not exist -- run `laplace init` to create it")]
    NothingToUpdate(PathBuf),

    #[error("failed to parse {path}: {message}")]
    ParseExisting { path: PathBuf, message: String },

    #[error("{path}: `{key}` must be {expected}")]
    WrongShape {
        path: PathBuf,
        key: &'static str,
        expected: &'static str,
    },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Package(Box<PackageError>),
}

/// What the sources in a package directory say, independent of any
/// manifest: the raw material for both `init` and `init --update`.
#[derive(Debug, Default)]
pub struct SourceScan {
    /// How many `.stan`/`.laplacelib` source files were scanned.
    pub source_files: usize,
    /// Every function name defined in a plain `.stan` file, in source order,
    /// one entry per name. Only these can ever appear in `exports`.
    pub stan_functions: Vec<String>,
    /// `// @laplace`-documented `.stan` functions -- the `exports` guess. A
    /// name with any documented overload counts as documented.
    pub included: Vec<String>,
    /// Undocumented `.stan` functions, assumed private and left out of
    /// `exports` -- reported so the user can add them by hand if that
    /// guess is wrong.
    pub excluded: Vec<String>,
    /// `.laplacelib` items already marked `pub`: public, and needing no
    /// manifest entry at all.
    pub already_pub: Vec<String>,
    /// The `pub` functions among [`Self::already_pub`] with no
    /// `// @laplace` doc comment, so `laplace doc` has nothing to show.
    pub pub_undocumented: Vec<String>,
    /// `.laplacelib` items *not* marked `pub`, and so private. Reported
    /// because `exports` cannot make them public -- only `pub` can.
    pub needs_pub: Vec<String>,
    /// Every `.laplacelib` item name, public or not.
    pub laplacelib_items: Vec<String>,
    /// Packages the `.laplacelib` sources import, which need entries under
    /// `[dependencies]` before the package can be installed.
    pub imports: Vec<String>,
}

impl SourceScan {
    /// Whether the package has plain `.stan` functions, and so has any use
    /// for an `exports` list.
    pub fn uses_exports(&self) -> bool {
        !self.stan_functions.is_empty() || self.laplacelib_items.is_empty()
    }
}

/// Scan every `.stan` and `.laplacelib` file directly inside `dir`.
pub fn scan(dir: &Path) -> Result<SourceScan, InitError> {
    let sources =
        package::read_package_sources(dir).map_err(|e| InitError::Package(Box::new(e)))?;
    let signatures = extract_signatures(&sources.body);
    let from_laplacelib = |name: &String| sources.laplacelib_items.contains(name);

    let mut stan_functions: Vec<String> = Vec::new();
    let mut included: Vec<String> = Vec::new();
    for sig in &signatures {
        if from_laplacelib(&sig.name) {
            continue;
        }
        if !stan_functions.contains(&sig.name) {
            stan_functions.push(sig.name.clone());
        }
        if sig.doc.is_some() && !included.contains(&sig.name) {
            included.push(sig.name.clone());
        }
    }
    let excluded: Vec<String> = stan_functions
        .iter()
        .filter(|name| !included.contains(name))
        .cloned()
        .collect();

    let needs_pub: Vec<String> = sources
        .laplacelib_items
        .iter()
        .filter(|name| !sources.public_items.contains(name))
        .cloned()
        .collect();
    // Only functions carry `// @laplace` docs that `laplace doc` shows; a
    // `pub` template or macro is not reported as undocumented.
    let pub_undocumented: Vec<String> = sources
        .public_items
        .iter()
        .filter(|name| {
            let mut overloads = signatures.iter().filter(|s| &s.name == *name).peekable();
            overloads.peek().is_some() && overloads.all(|s| s.doc.is_none())
        })
        .cloned()
        .collect();

    Ok(SourceScan {
        source_files: sources.files,
        stan_functions,
        included,
        excluded,
        already_pub: sources.public_items.clone(),
        pub_undocumented,
        needs_pub,
        laplacelib_items: sources.laplacelib_items.clone(),
        imports: sources.imports,
    })
}

/// What `init` did, for the CLI to report back to the user.
#[derive(Debug)]
pub struct InitSummary {
    pub name: String,
    /// The directory name `name` was derived from, when it had to be changed
    /// to be a valid package name (`laplace-splines` -> `laplace_splines`).
    pub renamed_from: Option<String>,
    /// Whether the written manifest has an `exports` key. `false` for a
    /// package whose sources are all `.laplacelib`.
    pub wrote_exports: bool,
    pub scan: SourceScan,
}

/// Scan every `.stan` and `.laplacelib` file directly inside `dir` and write
/// a starter `laplace.toml` there: `name` guessed from `dir`'s name,
/// `version = "0.1.0"`, and -- unless every source is `.laplacelib` --
/// `exports` pre-filled with every `// @laplace`-documented `.stan`
/// function. Errors rather than overwriting if `dir/laplace.toml` already
/// exists.
pub fn init(dir: &Path) -> Result<InitSummary, InitError> {
    let manifest_path = dir.join("laplace.toml");
    if manifest_path.exists() {
        return Err(InitError::AlreadyExists(manifest_path));
    }

    let dir_name = directory_name(dir);
    let name = sanitize_package_name(&dir_name);
    let renamed_from = (name != dir_name).then_some(dir_name);

    let scan = scan(dir)?;

    let mut doc = DocumentMut::new();
    doc["name"] = toml_edit::value(name.clone());
    doc["version"] = toml_edit::value("0.1.0");
    let wrote_exports = scan.uses_exports();
    if wrote_exports {
        doc["exports"] = Item::Value(Value::Array(string_array(&scan.included)));
    }
    fs::write(&manifest_path, doc.to_string())?;

    Ok(InitSummary {
        name,
        renamed_from,
        wrote_exports,
        scan,
    })
}

/// Why an `exports` entry no longer belongs there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleReason {
    /// No `.stan` file defines a function by this name any more.
    Missing,
    /// It is a `.laplacelib` item, whose visibility comes from `pub`.
    LaplacelibItem,
}

/// What `init --update` did, for the CLI to report back to the user.
#[derive(Debug)]
pub struct UpdateSummary {
    pub name: String,
    /// Top-level keys that were missing and got a default (`name`,
    /// `version`).
    pub added_keys: Vec<&'static str>,
    /// Documented `.stan` functions newly appended to `exports`.
    pub added_exports: Vec<String>,
    /// `exports` entries that do not name a `.stan` function, with why.
    pub stale_exports: Vec<(String, StaleReason)>,
    /// Whether [`Self::stale_exports`] were removed (`--prune`).
    pub pruned: bool,
    /// Whether `laplace.toml` was rewritten at all.
    pub changed: bool,
    pub scan: SourceScan,
}

/// `laplace init --update`: bring an existing `laplace.toml` in line with
/// the sources without disturbing it.
///
/// Only ever adds: a missing `name`/`version`, and documented `.stan`
/// functions not yet in `exports` (appended, in source order). Nothing is
/// reordered, comments and unknown keys survive, and an `exports` entry for
/// a function that no longer exists is reported, not removed -- unless
/// `prune` is set. Running it twice in a row changes nothing the second
/// time; the file is not even rewritten.
pub fn update(dir: &Path, prune: bool) -> Result<UpdateSummary, InitError> {
    let manifest_path = dir.join("laplace.toml");
    let original = match fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(InitError::NothingToUpdate(manifest_path))
        }
        Err(e) => return Err(e.into()),
    };
    let mut doc: DocumentMut =
        original
            .parse()
            .map_err(|e: toml_edit::TomlError| InitError::ParseExisting {
                path: manifest_path.clone(),
                message: e.to_string(),
            })?;

    let scan = scan(dir)?;

    let mut added_keys = Vec::new();
    if !doc.contains_key("name") {
        doc["name"] = toml_edit::value(sanitize_package_name(&directory_name(dir)));
        added_keys.push("name");
    }
    if !doc.contains_key("version") {
        doc["version"] = toml_edit::value("0.1.0");
        added_keys.push("version");
    }
    let name = doc["name"]
        .as_str()
        .ok_or_else(|| InitError::WrongShape {
            path: manifest_path.clone(),
            key: "name",
            expected: "a string",
        })?
        .to_string();

    let existing: Vec<String> = match doc.get("exports") {
        None => Vec::new(),
        Some(item) => item
            .as_array()
            .and_then(|array| {
                array
                    .iter()
                    .map(|v| v.as_str().map(str::to_string))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or_else(|| InitError::WrongShape {
                path: manifest_path.clone(),
                key: "exports",
                expected: "an array of function names",
            })?,
    };

    let stale_exports: Vec<(String, StaleReason)> = existing
        .iter()
        .filter(|name| !scan.stan_functions.contains(name))
        .map(|name| {
            let reason = if scan.laplacelib_items.contains(name) {
                StaleReason::LaplacelibItem
            } else {
                StaleReason::Missing
            };
            (name.clone(), reason)
        })
        .collect();
    let added_exports: Vec<String> = scan
        .included
        .iter()
        .filter(|name| !existing.contains(name))
        .cloned()
        .collect();

    if !added_exports.is_empty() || (prune && !stale_exports.is_empty()) {
        if !doc.contains_key("exports") {
            doc["exports"] = Item::Value(Value::Array(Array::new()));
        }
        let array = doc["exports"]
            .as_array_mut()
            .expect("checked to be an array above");
        if prune {
            array.retain(|v| {
                !stale_exports
                    .iter()
                    .any(|(stale, _)| v.as_str() == Some(stale.as_str()))
            });
        }
        for export in &added_exports {
            array.push(export.as_str());
        }
    }

    let updated = doc.to_string();
    let changed = updated != original;
    if changed {
        fs::write(&manifest_path, updated)?;
    }

    Ok(UpdateSummary {
        name,
        added_keys,
        added_exports,
        stale_exports,
        pruned: prune,
        changed,
        scan,
    })
}

fn string_array(items: &[String]) -> Array {
    items.iter().map(String::as_str).collect()
}

/// `dir`'s own name. Canonicalizes first so `.` and other relative paths
/// (which have no `file_name()` of their own) still resolve to the
/// directory's actual name.
fn directory_name(dir: &Path) -> String {
    dir.canonicalize()
        .ok()
        .as_deref()
        .unwrap_or(dir)
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| "package".to_string())
}

/// Turn a directory name into a valid package name (see
/// [`manifest::is_valid_package_name`]): every character that can't appear
/// in a Stan identifier becomes `_`, and a name that doesn't start with a
/// letter gets a `pkg_` prefix.
fn sanitize_package_name(raw: &str) -> String {
    let mut name = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() {
            name.push(c);
        } else if !name.ends_with('_') {
            // Runs collapse to one `_`: `__` is reserved for generated
            // names, so `kernels--2d` must not become `kernels__2d`.
            name.push('_');
        }
    }
    if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
        name.insert_str(0, "pkg_");
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest;

    fn write_stan(dir: &Path, filename: &str, contents: &str) {
        fs::write(dir.join(filename), contents).unwrap();
    }

    const DOCUMENTED_AND_UNDOCUMENTED: &str = r#"
// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @param x Vector of input locations.
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}

// Not a laplace doc comment, just a maintainer note.
real jitter(real epsilon) {
  return epsilon;
}
"#;

    #[test]
    fn generates_manifest_with_documented_exports_only() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("gps");
        fs::create_dir_all(&dir).unwrap();
        write_stan(&dir, "gps.stan", DOCUMENTED_AND_UNDOCUMENTED);

        let summary = init(&dir).unwrap();
        assert_eq!(summary.name, "gps");
        assert_eq!(summary.scan.included, vec!["rbf_cov".to_string()]);
        assert_eq!(summary.scan.excluded, vec!["jitter".to_string()]);

        let manifest = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert_eq!(manifest.name, "gps");
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.exports, vec!["rbf_cov".to_string()]);
    }

    #[test]
    fn scans_laplacelib_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("splines");
        fs::create_dir_all(&dir).unwrap();
        write_stan(
            &dir,
            "splines.laplacelib",
            "functions {\n// @laplace\n// @brief Knots.\nvector knots(int k) {\n  return rep_vector(0, k);\n}\n}\n",
        );

        let summary = init(&dir).unwrap();
        assert_eq!(summary.scan.source_files, 1);
        // `exports` never names a `.laplacelib` item: `pub` does that, and
        // `knots` has no `pub`, so it is reported as private instead.
        assert!(
            summary.scan.included.is_empty(),
            "{:?}",
            summary.scan.included
        );
        assert_eq!(summary.scan.needs_pub, vec!["knots".to_string()]);
        assert!(summary.scan.already_pub.is_empty());

        let written = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert!(written.exports.is_empty(), "{:?}", written.exports);
    }

    #[test]
    fn a_pub_laplacelib_item_is_reported_and_stays_out_of_exports() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("splines");
        fs::create_dir_all(&dir).unwrap();
        write_stan(
            &dir,
            "splines.laplacelib",
            "// @laplace
// @brief Knots.
pub vector knots(int k) {
  return rep_vector(0, k);
}

real helper() {
  return 1;
}
",
        );

        let summary = init(&dir).unwrap();
        assert_eq!(summary.scan.already_pub, vec!["knots".to_string()]);
        assert_eq!(summary.scan.needs_pub, vec!["helper".to_string()]);
        assert!(summary.scan.included.is_empty());

        // The generated manifest loads: nothing in `exports` contradicts
        // what the source says.
        let written = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert!(written.exports.is_empty());
        assert!(package::load_with_manifest(&dir, "splines", &written).is_ok());
    }

    #[test]
    fn a_mixed_package_exports_only_its_stan_functions() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("mixed");
        fs::create_dir_all(&dir).unwrap();
        write_stan(
            &dir,
            "a.stan",
            "// @laplace
// @brief From Stan.
real plain() {
  return 1;
}
",
        );
        write_stan(
            &dir,
            "b.laplacelib",
            "// @laplace
// @brief From laplacelib.
pub real fancy() {
  return 2;
}
",
        );

        let summary = init(&dir).unwrap();
        assert_eq!(summary.scan.included, vec!["plain".to_string()]);
        assert_eq!(summary.scan.already_pub, vec!["fancy".to_string()]);
    }

    #[test]
    fn a_sanitized_name_never_contains_a_double_underscore() {
        assert_eq!(sanitize_package_name("kernels--2d"), "kernels_2d");
        assert_eq!(sanitize_package_name("a..b__c"), "a_b_c");
        assert!(manifest::is_valid_package_name(&sanitize_package_name(
            "kernels--2d"
        )));
    }

    #[test]
    fn overloads_get_one_export_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("kinetics");
        fs::create_dir_all(&dir).unwrap();
        write_stan(
            &dir,
            "kinetics.stan",
            "// @laplace\n// @brief Hill.\nreal hill(real x) {\n  return x;\n}\n\
             vector hill(vector x) {\n  return x;\n}\n\
             real helper(real x) {\n  return x;\n}\nvector helper(vector x) {\n  return x;\n}\n",
        );

        let summary = init(&dir).unwrap();
        assert_eq!(summary.scan.included, vec!["hill".to_string()]);
        assert_eq!(summary.scan.excluded, vec!["helper".to_string()]);
        let manifest = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert_eq!(manifest.exports, vec!["hill".to_string()]);
    }

    #[test]
    fn hyphenated_directory_name_is_sanitized() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("laplace-splines");
        fs::create_dir_all(&dir).unwrap();

        let summary = init(&dir).unwrap();
        assert_eq!(summary.name, "laplace_splines");
        assert_eq!(summary.renamed_from.as_deref(), Some("laplace-splines"));
        assert!(manifest::is_valid_package_name(&summary.name));
        assert_eq!(sanitize_package_name("2d-kernels"), "pkg_2d_kernels");
    }

    #[test]
    fn errors_instead_of_overwriting_an_existing_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("gps");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("laplace.toml"),
            "name = \"gps\"\nversion = \"9.9.9\"\nexports = []\n",
        )
        .unwrap();

        let err = init(&dir).unwrap_err();
        assert!(matches!(err, InitError::AlreadyExists(_)));
        // The existing file must be untouched.
        assert!(fs::read_to_string(dir.join("laplace.toml"))
            .unwrap()
            .contains("9.9.9"));
    }

    #[test]
    fn no_stan_files_yields_an_empty_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("empty-pkg");
        fs::create_dir_all(&dir).unwrap();

        let summary = init(&dir).unwrap();
        assert_eq!(summary.scan.source_files, 0);
        assert!(summary.scan.included.is_empty());
        assert!(summary.scan.excluded.is_empty());

        let manifest = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert!(manifest.exports.is_empty());
    }

    #[test]
    fn multiple_stan_files_are_scanned_together_in_filename_order() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("gps");
        fs::create_dir_all(&dir).unwrap();
        write_stan(
            &dir,
            "a_first.stan",
            "// @laplace\n// @brief First.\nreal first_fn(real x) {\n  return x;\n}\n",
        );
        write_stan(
            &dir,
            "b_second.stan",
            "// @laplace\n// @brief Second.\nreal second_fn(real x) {\n  return x;\n}\n",
        );

        let summary = init(&dir).unwrap();
        assert_eq!(
            summary.scan.included,
            vec!["first_fn".to_string(), "second_fn".to_string()]
        );
    }

    // -- `init --update` -------------------------------------------------

    const ONE_DOCUMENTED: &str =
        "// @laplace\n// @brief RBF.\nmatrix rbf_cov(vector x) {\n  return x;\n}\n";
    const TWO_DOCUMENTED: &str = "// @laplace\n// @brief RBF.\nmatrix rbf_cov(vector x) {\n  \
         return x;\n}\n\n// @laplace\n// @brief Matern.\nmatrix matern_cov(vector x) {\n  return x;\n}\n";

    fn package_dir(tmp: &tempfile::TempDir) -> PathBuf {
        let dir = tmp.path().join("gps");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_laplacelib_only_package_gets_no_exports_key() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        write_stan(
            &dir,
            "gps.laplacelib",
            "// @laplace\n// @brief K.\npub real k() {\n  return 1;\n}\n",
        );
        let summary = init(&dir).unwrap();
        assert!(!summary.wrote_exports);
        let text = fs::read_to_string(dir.join("laplace.toml")).unwrap();
        assert!(!text.contains("exports"), "{text}");
        // ...and the manifest still loads.
        let written = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert!(package::load_with_manifest(&dir, "gps", &written).is_ok());
    }

    #[test]
    fn a_documented_pub_function_is_not_reported_as_undocumented() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        write_stan(
            &dir,
            "gps.laplacelib",
            "// @laplace\n// @brief K.\npub real k() {\n  return 1;\n}\n\
             pub real bare() {\n  return 2;\n}\n",
        );
        let scan = scan(&dir).unwrap();
        assert_eq!(scan.already_pub, vec!["bare".to_string(), "k".to_string()]);
        assert_eq!(scan.pub_undocumented, vec!["bare".to_string()]);
    }

    #[test]
    fn update_keeps_a_custom_name_version_comments_and_unknown_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        write_stan(&dir, "gps.stan", TWO_DOCUMENTED);
        let original = "# my package\nname = \"gaussian_processes\"\nversion = \"3.1.4\"\n\
                        laplace = \">=0.2\"\nexports = [\"rbf_cov\"] # hand-picked\n\n\
                        [dependencies]\nstats = \"^1.0\"\n";
        fs::write(dir.join("laplace.toml"), original).unwrap();

        let summary = update(&dir, false).unwrap();
        assert_eq!(summary.added_exports, vec!["matern_cov".to_string()]);
        assert!(summary.added_keys.is_empty());
        let text = fs::read_to_string(dir.join("laplace.toml")).unwrap();
        for kept in [
            "# my package",
            "name = \"gaussian_processes\"",
            "version = \"3.1.4\"",
            "laplace = \">=0.2\"",
            "# hand-picked",
            "stats = \"^1.0\"",
        ] {
            assert!(text.contains(kept), "lost `{kept}`:\n{text}");
        }
        let written = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert_eq!(written.exports, vec!["rbf_cov", "matern_cov"]);
    }

    #[test]
    fn a_newly_documented_function_is_added_once_and_a_second_run_is_a_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        write_stan(&dir, "gps.stan", ONE_DOCUMENTED);
        init(&dir).unwrap();
        write_stan(&dir, "gps.stan", TWO_DOCUMENTED);

        let first = update(&dir, false).unwrap();
        assert_eq!(first.added_exports, vec!["matern_cov".to_string()]);
        assert!(first.changed);
        let after_first = fs::read_to_string(dir.join("laplace.toml")).unwrap();

        let second = update(&dir, false).unwrap();
        assert!(second.added_exports.is_empty());
        assert!(!second.changed);
        assert_eq!(
            fs::read_to_string(dir.join("laplace.toml")).unwrap(),
            after_first
        );
        let written = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert_eq!(written.exports, vec!["rbf_cov", "matern_cov"]);
    }

    #[test]
    fn stale_exports_are_reported_and_only_removed_with_prune() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        write_stan(&dir, "gps.stan", ONE_DOCUMENTED);
        write_stan(
            &dir,
            "extra.laplacelib",
            "pub real fancy() {\n  return 1;\n}\n",
        );
        let original = "name = \"gps\"\nversion = \"1.0.0\"\n\
                        exports = [\"gone\", \"rbf_cov\", \"fancy\"]\n";
        fs::write(dir.join("laplace.toml"), original).unwrap();

        let summary = update(&dir, false).unwrap();
        assert_eq!(
            summary.stale_exports,
            vec![
                ("gone".to_string(), StaleReason::Missing),
                ("fancy".to_string(), StaleReason::LaplacelibItem),
            ]
        );
        assert!(!summary.changed);
        assert_eq!(
            fs::read_to_string(dir.join("laplace.toml")).unwrap(),
            original
        );

        let pruned = update(&dir, true).unwrap();
        assert!(pruned.changed);
        let text = fs::read_to_string(dir.join("laplace.toml")).unwrap();
        let written: PackageManifestExports = toml::from_str(&text).unwrap();
        assert_eq!(written.exports, vec!["rbf_cov"]);
    }

    #[derive(serde::Deserialize)]
    struct PackageManifestExports {
        exports: Vec<String>,
    }

    #[test]
    fn update_fills_in_a_missing_name_and_version() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        write_stan(&dir, "gps.stan", ONE_DOCUMENTED);
        fs::write(dir.join("laplace.toml"), "exports = []\n").unwrap();
        let summary = update(&dir, false).unwrap();
        assert_eq!(summary.added_keys, vec!["name", "version"]);
        let written = manifest::read_package_manifest(&dir.join("laplace.toml")).unwrap();
        assert_eq!(written.name, "gps");
        assert_eq!(written.version, "0.1.0");
        assert_eq!(written.exports, vec!["rbf_cov"]);
    }

    #[test]
    fn update_without_a_manifest_points_at_plain_init() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = package_dir(&tmp);
        assert!(matches!(
            update(&dir, false),
            Err(InitError::NothingToUpdate(_))
        ));
    }
}
