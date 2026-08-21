//! `laplace init`: scan a package repo's `.stan` files and generate a
//! starter `laplace.toml`, guessing `exports` from `// @laplace`-documented
//! functions -- undocumented functions are assumed private and left out.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::manifest::{self, ManifestError, PackageManifest};
use crate::parser::signatures::extract_signatures;

#[derive(Debug, Error)]
pub enum InitError {
    #[error("{0} already exists -- remove it first if you want to regenerate it")]
    AlreadyExists(PathBuf),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Manifest(#[from] ManifestError),
}

/// What `init` did, for the CLI to report back to the user.
#[derive(Debug)]
pub struct InitSummary {
    pub name: String,
    /// `// @laplace`-documented functions written into `exports`, in the
    /// order they were found.
    pub included: Vec<String>,
    /// Undocumented functions left out of `exports` on the assumption
    /// they're private -- reported so the user can add them by hand if
    /// that guess is wrong.
    pub excluded: Vec<String>,
}

/// Scan every `.stan` file directly inside `dir` and write a starter
/// `laplace.toml` there: `name` guessed from `dir`'s name, `version =
/// "0.1.0"`, and `exports` pre-filled with every `// @laplace`-documented
/// function. Errors rather than overwriting if `dir/laplace.toml` already
/// exists.
pub fn init(dir: &Path) -> Result<InitSummary, InitError> {
    let manifest_path = dir.join("laplace.toml");
    if manifest_path.exists() {
        return Err(InitError::AlreadyExists(manifest_path));
    }

    let name = guess_package_name(dir);
    let source = manifest::read_package_stan_source(dir)?;
    let signatures = extract_signatures(&source);

    let mut included = Vec::new();
    let mut excluded = Vec::new();
    for sig in &signatures {
        if sig.doc.is_some() {
            included.push(sig.name.clone());
        } else {
            excluded.push(sig.name.clone());
        }
    }

    let package_manifest = PackageManifest {
        name: name.clone(),
        version: "0.1.0".to_string(),
        exports: included.clone(),
    };
    manifest::write_package_manifest(&manifest_path, &package_manifest)?;

    Ok(InitSummary {
        name,
        included,
        excluded,
    })
}

/// Guess a package name from `dir`'s own name. Canonicalizes first so `.`
/// and other relative paths (which have no `file_name()` of their own)
/// still resolve to the directory's actual name.
fn guess_package_name(dir: &Path) -> String {
    dir.canonicalize()
        .ok()
        .as_deref()
        .unwrap_or(dir)
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| "package".to_string())
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
