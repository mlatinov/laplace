//! `laplace init`: scan a package repo's `.stan` and `.laplacelib` files and
//! generate a starter `laplace.toml`, guessing `exports` from
//! `// @laplace`-documented functions -- undocumented functions are assumed
//! private and left out.
//!
//! `exports` only ever names functions from plain `.stan` files. A
//! `.laplacelib` item's visibility comes from `pub` in the source, and
//! listing one in `exports` is an error, so `init` reports those items
//! instead of writing a manifest that would not load.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::manifest::{self, ManifestError, PackageManifest};
use crate::package::{self, PackageError};
use crate::parser::signatures::extract_signatures;

#[derive(Debug, Error)]
pub enum InitError {
    #[error("{0} already exists -- remove it first if you want to regenerate it")]
    AlreadyExists(PathBuf),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Manifest(#[from] ManifestError),

    #[error(transparent)]
    Package(Box<PackageError>),
}

/// What `init` did, for the CLI to report back to the user.
#[derive(Debug)]
pub struct InitSummary {
    pub name: String,
    /// The directory name `name` was derived from, when it had to be changed
    /// to be a valid package name (`laplace-splines` -> `laplace_splines`).
    pub renamed_from: Option<String>,
    /// How many `.stan`/`.laplacelib` source files were scanned.
    pub source_files: usize,
    /// `// @laplace`-documented functions written into `exports`, in the
    /// order they were found. One entry per name: overloads share an export.
    pub included: Vec<String>,
    /// Undocumented functions left out of `exports` on the assumption
    /// they're private -- reported so the user can add them by hand if
    /// that guess is wrong. A name with any documented overload is exported,
    /// never listed here.
    pub excluded: Vec<String>,
    /// `.laplacelib` items already marked `pub`: public, and needing no
    /// manifest entry at all.
    pub already_pub: Vec<String>,
    /// `.laplacelib` items *not* marked `pub`, and so private. Reported
    /// because `exports` cannot make them public -- only `pub` can.
    pub needs_pub: Vec<String>,
    /// Packages the `.laplacelib` sources import, which need entries under
    /// `[dependencies]` before the package can be installed.
    pub imports: Vec<String>,
}

/// Scan every `.stan` and `.laplacelib` file directly inside `dir` and write
/// a starter `laplace.toml` there: `name` guessed from `dir`'s name,
/// `version = "0.1.0"`, and `exports` pre-filled with every
/// `// @laplace`-documented function. Errors rather than overwriting if
/// `dir/laplace.toml` already exists.
pub fn init(dir: &Path) -> Result<InitSummary, InitError> {
    let manifest_path = dir.join("laplace.toml");
    if manifest_path.exists() {
        return Err(InitError::AlreadyExists(manifest_path));
    }

    let dir_name = directory_name(dir);
    let name = sanitize_package_name(&dir_name);
    let renamed_from = (name != dir_name).then_some(dir_name);

    let sources =
        package::read_package_sources(dir).map_err(|e| InitError::Package(Box::new(e)))?;
    let signatures = extract_signatures(&sources.body);

    let from_laplacelib = |name: &String| sources.laplacelib_items.contains(name);

    let mut included: Vec<String> = Vec::new();
    for sig in signatures.iter().filter(|s| s.doc.is_some()) {
        if !from_laplacelib(&sig.name) && !included.contains(&sig.name) {
            included.push(sig.name.clone());
        }
    }
    let mut excluded: Vec<String> = Vec::new();
    for sig in &signatures {
        if !from_laplacelib(&sig.name)
            && !included.contains(&sig.name)
            && !excluded.contains(&sig.name)
        {
            excluded.push(sig.name.clone());
        }
    }
    let needs_pub: Vec<String> = sources
        .laplacelib_items
        .iter()
        .filter(|name| !sources.public_items.contains(name))
        .cloned()
        .collect();

    let package_manifest = PackageManifest::new(name.clone(), "0.1.0", included.clone());
    manifest::write_package_manifest(&manifest_path, &package_manifest)?;

    Ok(InitSummary {
        name,
        renamed_from,
        source_files: sources.files,
        included,
        excluded,
        already_pub: sources.public_items.clone(),
        needs_pub,
        imports: sources.imports,
    })
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
    use std::fs;

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
        assert_eq!(summary.included, vec!["rbf_cov".to_string()]);
        assert_eq!(summary.excluded, vec!["jitter".to_string()]);

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
        assert_eq!(summary.source_files, 1);
        // `exports` never names a `.laplacelib` item: `pub` does that, and
        // `knots` has no `pub`, so it is reported as private instead.
        assert!(summary.included.is_empty(), "{:?}", summary.included);
        assert_eq!(summary.needs_pub, vec!["knots".to_string()]);
        assert!(summary.already_pub.is_empty());

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
        assert_eq!(summary.already_pub, vec!["knots".to_string()]);
        assert_eq!(summary.needs_pub, vec!["helper".to_string()]);
        assert!(summary.included.is_empty());

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
        assert_eq!(summary.included, vec!["plain".to_string()]);
        assert_eq!(summary.already_pub, vec!["fancy".to_string()]);
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
        assert_eq!(summary.included, vec!["hill".to_string()]);
        assert_eq!(summary.excluded, vec!["helper".to_string()]);
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
        assert_eq!(summary.source_files, 0);
        assert!(summary.included.is_empty());
        assert!(summary.excluded.is_empty());

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
            summary.included,
            vec!["first_fn".to_string(), "second_fn".to_string()]
        );
    }
}
