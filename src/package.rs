//! Loading a package directory (`laplace.toml` + source files) into the
//! shape codegen wants.
//!
//! A package's sources are its `.stan` files (plain Stan, passed through
//! verbatim) and its `.laplacelib` files (the library dialect: may import
//! other packages, see [`crate::parser::laplacelib`]). Both kinds are read
//! in filename order and concatenated into one body, so a multi-file
//! package behaves exactly like a single-file one -- unchanged from how
//! multi-file `.stan` packages have always worked.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::codegen::InstalledPackage;
use crate::manifest::{self, ManifestError, PackageManifest};
use crate::parser::laplacelib::{self, LaplaceLibError, LAPLACELIB_EXTENSION};
use crate::parser::origin::{FileOrigin, LineSegment, PackageOrigin};
use crate::parser::signatures::extract_signatures;
use crate::parser::macros::LocatedMacro;
use crate::parser::template::LocatedTemplate;
use crate::parser::visibility::{ItemKind, Visibility};

#[derive(Debug, Error)]
pub enum PackageError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Manifest(#[from] ManifestError),

    #[error(transparent)]
    LaplaceLib(#[from] LaplaceLibError),

    #[error(
        "{path} imports `{import}`, but `{package}`'s laplace.toml does not list it under \
         [dependencies] -- a library must declare every package it imports"
    )]
    UndeclaredImport {
        path: PathBuf,
        package: String,
        import: String,
    },

    #[error(
        "`{package}`'s laplace.toml lists `{item}` under `exports`, but `{item}` is defined in a \
         `.laplacelib` file, where visibility comes from the `pub` keyword and `exports` is \
         ignored\n  help: write `pub` in front of `{item}`'s definition and drop it from \
         `exports` (`exports` still applies to a package's plain `.stan` files)"
    )]
    ExportsListsLaplacelibItem { package: String, item: String },

    #[error(
        "`{package}` marks some definitions of the {kind} `{item}` `pub` and others not -- \
         visibility applies to the name, because Stan has one flat function namespace\n  help: \
         mark every definition of `{item}` `pub`, or none of them"
    )]
    InconsistentVisibility {
        package: String,
        item: String,
        kind: &'static str,
    },
}

/// A package's concatenated source body plus everything it imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageSources {
    /// Every `.stan` file verbatim and every `.laplacelib` file's body
    /// (`library { }` stripped, `functions { }` unwrapped, `pub` markers
    /// removed), concatenated in filename order.
    pub body: String,
    /// Package names imported by this package's `.laplacelib` files,
    /// sorted and deduplicated.
    pub imports: Vec<String>,
    /// How many source files went into `body`.
    pub files: usize,
    /// Which file and line each byte of `body` was written on.
    pub origin: PackageOrigin,
    /// Every item name defined in a `.laplacelib` file, sorted and
    /// deduplicated. These take their visibility from `pub`, never from
    /// the manifest's `exports`.
    pub laplacelib_items: Vec<String>,
    /// The `.laplacelib` item names marked `pub`, sorted and deduplicated.
    pub public_items: Vec<String>,
    /// Every `@template` the package defines, with where it was
    /// written. Kept apart from `public_items`, which is the function
    /// API: nothing calls a template, a `@use` expands it.
    pub templates: Vec<LocatedTemplate>,
    /// Every `@macro` the package defines, with where it was written.
    pub macros: Vec<LocatedMacro>,
    /// A `.laplacelib` name with both a `pub` and a non-`pub` definition,
    /// as `(name, item kind)`. Reported as an error by
    /// [`load_with_manifest`], which knows the package's name.
    pub inconsistent: Option<(String, &'static str)>,
}

/// Read and combine every source file in `package_dir`.
pub fn read_package_sources(package_dir: &Path) -> Result<PackageSources, PackageError> {
    let mut paths: Vec<PathBuf> = fs::read_dir(package_dir)
        .map_err(|source| PackageError::Io {
            path: package_dir.to_path_buf(),
            source,
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("stan") | Some(LAPLACELIB_EXTENSION)
            )
        })
        .collect();
    // Filename order, so a multi-file package concatenates deterministically.
    paths.sort();

    let mut body = String::new();
    let mut imports: Vec<String> = Vec::new();
    let mut origin = PackageOrigin::default();
    let mut laplacelib_items: Vec<String> = Vec::new();
    let mut public_items: Vec<String> = Vec::new();
    // Every `.laplacelib` item's visibility, to catch a name that is
    // `pub` in one definition and not in another.
    let mut seen: Vec<(String, ItemKind, Visibility)> = Vec::new();
    let mut templates: Vec<LocatedTemplate> = Vec::new();
    let mut macros: Vec<LocatedMacro> = Vec::new();

    for path in &paths {
        let text = fs::read_to_string(path).map_err(|source| PackageError::Io {
            path: path.clone(),
            source,
        })?;
        // The file name alone: provenance comments and error messages must
        // not leak the machine-specific cache path, or build output would
        // stop being reproducible across machines.
        let file = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let base = body.len();

        if path.extension().and_then(|ext| ext.to_str()) == Some(LAPLACELIB_EXTENSION) {
            let parsed = laplacelib::parse(path, &text)?;
            imports.extend(parsed.imports.iter().map(|i| i.name.clone()));
            for item in &parsed.items {
                laplacelib_items.push(item.name.clone());
                if item.visibility.is_public() {
                    public_items.push(item.name.clone());
                }
                seen.push((item.name.clone(), item.kind, item.visibility));
            }
            templates.extend(parsed.located_templates(&file));
            macros.extend(parsed.located_macros(&file));
            for template in &parsed.templates {
                seen.push((
                    template.name.clone(),
                    ItemKind::Template,
                    template.visibility,
                ));
            }
            for definition in &parsed.macros {
                seen.push((
                    definition.name.clone(),
                    ItemKind::Macro,
                    definition.visibility,
                ));
            }
            body.push_str(&parsed.body);
            origin.files.push(FileOrigin {
                file,
                body_range: base..body.len(),
                segments: parsed.segments,
            });
        } else {
            body.push_str(&text);
            origin.files.push(FileOrigin {
                file,
                body_range: base..body.len(),
                segments: vec![LineSegment {
                    range: 0..text.len(),
                    original_line: 1,
                    original_offset: 0,
                }],
            });
        }
        body.push('\n');
    }

    imports.sort();
    imports.dedup();
    laplacelib_items.sort();
    laplacelib_items.dedup();
    public_items.sort();
    public_items.dedup();

    Ok(PackageSources {
        body,
        imports,
        files: paths.len(),
        origin,
        laplacelib_items,
        public_items,
        templates,
        macros,
        inconsistent: first_inconsistent_visibility(&seen),
    })
}

/// A name with both a `pub` and a non-`pub` definition, if there is one.
/// Reported from [`load_with_manifest`], which knows the package name.
fn first_inconsistent_visibility(
    seen: &[(String, ItemKind, Visibility)],
) -> Option<(String, &'static str)> {
    let mut sorted: Vec<&(String, ItemKind, Visibility)> = seen.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted
        .windows(2)
        .find(|pair| pair[0].0 == pair[1].0 && pair[0].2 != pair[1].2)
        .map(|pair| (pair[0].0.clone(), pair[0].1.describe()))
}

/// Load `package_dir` into an [`InstalledPackage`] ready for codegen.
///
/// Cross-checks the package's `.laplacelib` `library { }` imports against
/// its manifest `[dependencies]`: importing something the manifest never
/// declares would leave the resolver with no version range to work from, so
/// it is rejected here rather than failing mysteriously later.
pub fn load(package_dir: &Path, name: &str) -> Result<InstalledPackage, PackageError> {
    let pkg_manifest = manifest::read_package_manifest(&package_dir.join("laplace.toml"))?;
    load_with_manifest(package_dir, name, &pkg_manifest)
}

pub fn load_with_manifest(
    package_dir: &Path,
    name: &str,
    pkg_manifest: &PackageManifest,
) -> Result<InstalledPackage, PackageError> {
    let sources = read_package_sources(package_dir)?;

    for import in &sources.imports {
        if !pkg_manifest.dependencies.contains_key(import) {
            return Err(PackageError::UndeclaredImport {
                path: package_dir.to_path_buf(),
                package: name.to_string(),
                import: import.clone(),
            });
        }
    }

    let exported = public_names(name, &sources, pkg_manifest)?;
    let signatures = extract_signatures(&sources.body);
    Ok(InstalledPackage {
        name: name.to_string(),
        version: pkg_manifest.version.clone(),
        source: sources.body,
        signatures,
        exported,
        dependencies: sources.imports,
        origin: sources.origin,
        templates: sources.templates,
        macros: sources.macros,
    })
}

/// A package's public API: the names a consumer may write as
/// `pkg::name`.
///
/// There are two dialects and one rule each, by design:
///
/// - a `.laplacelib` item is public exactly when it is marked `pub`
/// - a plain `.stan` file's function is public exactly when the manifest's
///   `exports` lists it, which is how every package worked before `pub`
///   existed and is why those packages keep compiling unchanged
///
/// Listing a `.laplacelib` item in `exports` is therefore a contradiction,
/// not a second way to say `pub`: it is rejected rather than honoured,
/// because honouring it would leave two places to look for one answer and
/// silently ignoring it would un-export the item without saying so.
fn public_names(
    package: &str,
    sources: &PackageSources,
    pkg_manifest: &PackageManifest,
) -> Result<Vec<String>, PackageError> {
    if let Some((item, kind)) = &sources.inconsistent {
        return Err(PackageError::InconsistentVisibility {
            package: package.to_string(),
            item: item.clone(),
            kind,
        });
    }

    let mut public = sources.public_items.clone();
    for export in &pkg_manifest.exports {
        if sources.laplacelib_items.contains(export) {
            return Err(PackageError::ExportsListsLaplacelibItem {
                package: package.to_string(),
                item: export.clone(),
            });
        }
        public.push(export.clone());
    }
    public.sort();
    public.dedup();
    Ok(public)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, contents: &str) {
        fs::write(dir.join(name), contents).unwrap();
    }

    fn manifest_toml(body: &str) -> String {
        body.to_string()
    }

    #[test]
    fn plain_stan_files_are_concatenated_verbatim_in_filename_order() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "b.stan", "real b() { return 2; }\n");
        write(tmp.path(), "a.stan", "real a() { return 1; }\n");

        let sources = read_package_sources(tmp.path()).unwrap();
        assert_eq!(
            sources.body,
            "real a() { return 1; }\n\nreal b() { return 2; }\n\n"
        );
        assert!(sources.imports.is_empty());
    }

    #[test]
    fn laplacelib_files_contribute_their_body_and_their_imports() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            tmp.path(),
            "regression.laplacelib",
            "library {\n  import stats\n}\n\nreal fit() {\n  return stats::mean_();\n}\n",
        );

        let sources = read_package_sources(tmp.path()).unwrap();
        assert_eq!(sources.imports, vec!["stats"]);
        assert!(!sources.body.contains("library"));
        assert!(sources.body.contains("stats::mean_()"));
    }

    #[test]
    fn a_package_can_mix_stan_and_laplacelib_files() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "helpers.stan", "real helper() { return 0; }\n");
        write(
            tmp.path(),
            "main.laplacelib",
            "library {\n  import stats\n}\nreal fit() { return helper(); }\n",
        );

        let sources = read_package_sources(tmp.path()).unwrap();
        assert!(sources.body.contains("real helper()"));
        assert!(sources.body.contains("real fit()"));
        assert_eq!(sources.imports, vec!["stats"]);
    }

    #[test]
    fn non_source_files_are_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "a.stan", "real a() { return 1; }\n");
        write(tmp.path(), "README.md", "# hi\n");
        write(tmp.path(), "docs.json", "{}\n");

        let sources = read_package_sources(tmp.path()).unwrap();
        assert_eq!(sources.body, "real a() { return 1; }\n\n");
    }

    #[test]
    fn load_fills_in_version_exports_and_dependencies() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            tmp.path(),
            "laplace.toml",
            &manifest_toml(concat!(
                "name = \"regression\"\n",
                "version = \"1.2.0\"\n",
                "[dependencies]\n",
                "stats = \"^1.0\"\n",
            )),
        );
        write(
            tmp.path(),
            "regression.laplacelib",
            "library {\n  import stats\n}\npub real fit() { return stats::mean_(); }\n",
        );

        let pkg = load(tmp.path(), "regression").unwrap();
        assert_eq!(pkg.version, "1.2.0");
        assert_eq!(pkg.exported, vec!["fit"]);
        assert_eq!(pkg.dependencies, vec!["stats"]);
        assert_eq!(pkg.signatures.len(), 1);
        assert_eq!(pkg.signatures[0].name, "fit");
    }

    #[test]
    fn importing_something_the_manifest_never_declares_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            tmp.path(),
            "laplace.toml",
            "name = \"regression\"\nversion = \"1.0.0\"\nexports = [\"fit\"]\n",
        );
        write(
            tmp.path(),
            "regression.laplacelib",
            "library {\n  import stats\n}\nreal fit() { return stats::mean_(); }\n",
        );

        let err = load(tmp.path(), "regression").unwrap_err();
        assert!(matches!(err, PackageError::UndeclaredImport { .. }), "{err:?}");
        let rendered = err.to_string();
        assert!(rendered.contains("stats"), "{rendered}");
        assert!(rendered.contains("[dependencies]"), "{rendered}");
    }

    #[test]
    fn a_model_block_in_a_laplacelib_is_rejected_at_load_time() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            tmp.path(),
            "laplace.toml",
            "name = \"bad\"\nversion = \"1.0.0\"\nexports = []\n",
        );
        write(tmp.path(), "bad.laplacelib", "model {\n  y ~ normal(0, 1);\n}\n");

        let err = load(tmp.path(), "bad").unwrap_err();
        assert!(matches!(err, PackageError::LaplaceLib(_)), "{err:?}");
        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn reading_sources_is_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        for name in ["c.stan", "a.stan", "b.laplacelib"] {
            write(tmp.path(), name, "real f() { return 1; }\n");
        }
        assert_eq!(
            read_package_sources(tmp.path()).unwrap(),
            read_package_sources(tmp.path()).unwrap()
        );
    }
}
