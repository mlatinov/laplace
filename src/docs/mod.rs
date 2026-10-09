//! Doc comment -> `docs.json` extraction, and `laplace doc <pkg>::<func>`
//! lookup/render.
//!
//! `docs.json` is written into a package's installed directory
//! (`cache_root/<name>/<version>/docs.json`) whenever that package lands in
//! the cache -- see `resolve::install_one`, which calls `write_sidecar`
//! after every copy, so `add`/`update`/`install` all keep it current.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::parser::signatures::{extract_signatures, Doc, FunctionSig};
use crate::resolve::lockfile::{self, LockfileError};

/// The full `docs.json` sidecar for one installed package version: every
/// top-level function Task 2 could extract, sorted by name for determinism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageDocs {
    pub package: String,
    pub version: String,
    /// Which doc extractor wrote this file ([`DOCS_FORMAT`]). A sidecar
    /// from an older extractor is rebuilt from source on lookup, so a fix to
    /// doc parsing reaches already-installed packages without a reinstall.
    /// Absent in files written before it existed, which read as 0.
    #[serde(default)]
    pub format: u32,
    pub functions: Vec<FunctionSig>,
    /// Names defined in a `.laplacelib` file without `pub`: private to
    /// the package, so `laplace doc` does not show them. Only
    /// `.laplacelib` items appear here -- a plain `.stan` package has no
    /// `pub` keyword, so its functions are all documented as before.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub private: Vec<String>,
}

/// Bump whenever doc extraction changes what it produces for the same
/// source. 1: a plain comment glued above `// @laplace` no longer hides the
/// doc block. 2: `private` records which `.laplacelib` items are not
/// `pub`, so a sidecar written before `pub` existed is rebuilt rather
/// than trusted to know what is public.
pub const DOCS_FORMAT: u32 = 2;

#[derive(Debug, Error)]
pub enum DocsError {
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to serialize docs for {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error(transparent)]
    Lockfile(#[from] LockfileError),

    #[error("`{package}` is not a dependency of this project (not found in laplace.lock)")]
    PackageNotInLockfile { package: String },

    #[error("`{package}` is locked but not installed at {path} -- run `laplace install`?")]
    PackageNotInstalled { package: String, path: PathBuf },

    #[error("package `{package}` has no function named `{func}`")]
    FunctionNotFound { package: String, func: String },

    #[error(
        "`{package}::{func}` is private to package `{package}`: the installed \
         `{package}@{version}` defines `{func}` without `pub`\n  help: only items marked `pub` \
         are part of `{package}`'s documented API. If you expected it to be public, the \
         installed copy may be older than you think -- laplace.lock pins {version} from \
         `{pkg_source}`; check that this version/tag contains the `pub`, then run \
         `laplace update {package}` (which also refreshes a cache whose source changed \
         without a version bump)"
    )]
    ItemIsPrivate {
        package: String,
        func: String,
        version: String,
        pkg_source: String,
    },

    #[error(transparent)]
    Package(Box<crate::package::PackageError>),

    #[error(transparent)]
    Manifest(#[from] crate::manifest::ManifestError),

    #[error(transparent)]
    Resolve(Box<crate::resolve::ResolveError>),
}

/// Extract every function signature from `package_dir`'s `.stan` file(s)
/// and write them to `package_dir/docs.json`. Called after a package is
/// copied into the cache, so `laplace doc` never has to re-parse Stan
/// source at lookup time.
pub fn write_sidecar(
    package_dir: &Path,
    name: &str,
    version: &str,
) -> Result<PackageDocs, DocsError> {
    let sources = crate::package::read_package_sources(package_dir)
        .map_err(|source| DocsError::Package(Box::new(source)))?;

    let mut functions = extract_signatures(&sources.body);
    functions.sort_by(|a, b| a.name.cmp(&b.name));

    let private: Vec<String> = sources
        .laplacelib_items
        .iter()
        .filter(|name| !sources.public_items.contains(name))
        .cloned()
        .collect();

    let docs = PackageDocs {
        package: name.to_string(),
        version: version.to_string(),
        format: DOCS_FORMAT,
        functions,
        private,
    };
    write_docs_json(&package_dir.join("docs.json"), &docs)?;
    Ok(docs)
}

fn write_docs_json(path: &Path, docs: &PackageDocs) -> Result<(), DocsError> {
    let text = serde_json::to_string_pretty(docs).map_err(|source| DocsError::Serialize {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, text).map_err(|source| DocsError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_docs_json(path: &Path) -> Result<PackageDocs, DocsError> {
    let text = fs::read_to_string(path).map_err(|source| DocsError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| DocsError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

/// `laplace doc <pkg>::<func>`: resolve `pkg`'s installed version via the
/// project's lockfile, load its `docs.json`, and find every overload of
/// `func`, in source order. Never empty on success.
pub fn lookup(
    lockfile_path: &Path,
    cache_root: &Path,
    package: &str,
    func: &str,
) -> Result<Vec<FunctionSig>, DocsError> {
    let lock = lockfile::read_lockfile(lockfile_path)?;
    let Some(locked) = lock.packages.iter().find(|p| p.name == package) else {
        return Err(DocsError::PackageNotInLockfile {
            package: package.to_string(),
        });
    };

    // A path dependency is re-synced from its directory first, so the docs
    // shown are the ones in the source being edited.
    let (package_dir, _) = crate::resolve::installed_package_dir(lockfile_path, cache_root, locked)
        .map_err(|e| DocsError::Resolve(Box::new(e)))?;
    let docs_path = package_dir.join("docs.json");
    if !docs_path.is_file() {
        return Err(DocsError::PackageNotInstalled {
            package: package.to_string(),
            path: package_dir,
        });
    }

    let manifest_path = package_dir.join("laplace.toml");
    if manifest_path.is_file() {
        crate::manifest::read_package_manifest(&manifest_path)?;
    }

    let mut docs = read_docs_json(&docs_path)?;
    if docs.format < DOCS_FORMAT {
        docs = write_sidecar(&package_dir, &locked.name, &locked.version)?;
    }
    if docs.private.iter().any(|name| name == func) {
        return Err(DocsError::ItemIsPrivate {
            package: package.to_string(),
            func: func.to_string(),
            version: locked.version.clone(),
            pkg_source: locked.source.clone(),
        });
    }
    let overloads: Vec<FunctionSig> = docs
        .functions
        .into_iter()
        .filter(|f| f.name == func)
        .collect();
    if overloads.is_empty() {
        return Err(DocsError::FunctionNotFound {
            package: package.to_string(),
            func: func.to_string(),
        });
    }
    Ok(overloads)
}

/// Overloads that share one doc comment, for rendering: consecutive
/// signatures with an identical doc, plus any undocumented overload directly
/// after a documented one (a library typically writes one `@laplace` block
/// above the first of several overloads).
fn group_overloads(sigs: &[FunctionSig]) -> Vec<(Vec<&FunctionSig>, Option<&Doc>)> {
    let mut groups: Vec<(Vec<&FunctionSig>, Option<&Doc>)> = Vec::new();
    for sig in sigs {
        let joins_previous = match groups.last() {
            Some((_, group_doc)) => sig.doc.is_none() || sig.doc.as_ref() == *group_doc,
            None => false,
        };
        if joins_previous {
            groups.last_mut().expect("checked above").0.push(sig);
        } else {
            groups.push((vec![sig], sig.doc.as_ref()));
        }
    }
    groups
}

/// Render every overload of one name for the terminal: each group of
/// overloads sharing a doc comment lists its signatures, then the doc once.
pub fn render_overloads(package: &str, sigs: &[FunctionSig]) -> String {
    let mut out = String::new();
    if sigs.len() > 1 {
        out.push_str(&format!(
            "{package}::{} has {} overloads\n\n",
            sigs[0].name,
            sigs.len()
        ));
    }
    for (i, (members, doc)) in group_overloads(sigs).into_iter().enumerate() {
        if i > 0 {
            out.push_str("\n---\n\n");
        }
        for sig in members {
            out.push_str(&signature_line(package, sig));
            out.push('\n');
        }
        out.push_str(&doc_body(doc));
    }
    out
}

fn signature_line(package: &str, sig: &FunctionSig) -> String {
    let params_sig = sig
        .params
        .iter()
        .map(|(name, ty)| format!("{name}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{package}::{}({params_sig}) -> {}",
        sig.name, sig.return_type
    )
}

/// Pretty-print a function's signature and doc comment (if any) for
/// terminal display. A function with no `// @laplace` doc comment still
/// renders its signature, with a note that no docs are available.
pub fn render(package: &str, sig: &FunctionSig) -> String {
    format!(
        "{}\n{}",
        signature_line(package, sig),
        doc_body(sig.doc.as_ref())
    )
}

/// The part of [`render`] below the signature line.
fn doc_body(doc: Option<&Doc>) -> String {
    let mut out = String::new();
    let Some(doc) = doc else {
        out.push_str("\n(no @laplace documentation available for this function)\n");
        return out;
    };

    if let Some(brief) = &doc.brief {
        out.push('\n');
        out.push_str(brief);
        out.push('\n');
    }

    if let Some(math) = &doc.math {
        out.push_str("\nMath:\n");
        for line in math.lines() {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }

    if !doc.params.is_empty() {
        out.push_str("\nParameters:\n");
        let width = doc
            .params
            .iter()
            .map(|(name, _)| name.len())
            .max()
            .unwrap_or(0);
        for (name, desc) in &doc.params {
            out.push_str(&format!("  {name:width$}  {desc}\n"));
        }
    }

    if let Some(ret) = &doc.return_doc {
        out.push_str("\nReturns:\n  ");
        out.push_str(ret);
        out.push('\n');
    }

    if let Some(example) = &doc.example {
        out.push_str("\nExample:\n");
        for line in example.lines() {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }

    out
}

/// Render a function's signature and doc comment as a minimal standalone
/// HTML file: brief/params/return as plain text, the example in a
/// `<pre><code>` block (preserving line breaks and monospacing, since it's
/// Stan code, not prose), and -- if present -- the math field in a KaTeX
/// auto-render span loaded from a CDN script tag, so opening the file in
/// any browser renders both the formula and a correctly formatted example.
pub fn render_html(package: &str, sig: &FunctionSig) -> String {
    render_html_overloads(package, std::slice::from_ref(sig))
}

/// [`render_html`] for every overload of one name, grouped the same way as
/// [`render_overloads`].
pub fn render_html_overloads(package: &str, sigs: &[FunctionSig]) -> String {
    let title = match sigs {
        [only] => signature_line(package, only),
        _ => format!("{package}::{}", sigs[0].name),
    };
    let mut body = String::new();
    for (members, doc) in group_overloads(sigs) {
        for sig in members {
            body.push_str(&format!(
                "<h1>{}</h1>\n",
                html_escape(&signature_line(package, sig))
            ));
        }
        body.push_str(&html_doc_body(doc));
    }
    wrap_html(&title, &body)
}

fn html_doc_body(doc: Option<&Doc>) -> String {
    let mut body = String::new();
    let Some(doc) = doc else {
        body.push_str("<p><em>no @laplace documentation available for this function</em></p>\n");
        return body;
    };

    if let Some(brief) = &doc.brief {
        body.push_str(&format!("<p>{}</p>\n", html_escape(brief)));
    }

    if let Some(math) = &doc.math {
        body.push_str("<span class=\"math\">\\[");
        body.push_str(&html_escape(math));
        body.push_str("\\]</span>\n");
    }

    if !doc.params.is_empty() {
        body.push_str("<h2>Parameters</h2>\n<ul>\n");
        for (name, desc) in &doc.params {
            body.push_str(&format!(
                "<li><strong>{}</strong>: {}</li>\n",
                html_escape(name),
                html_escape(desc)
            ));
        }
        body.push_str("</ul>\n");
    }

    if let Some(ret) = &doc.return_doc {
        body.push_str(&format!("<h2>Returns</h2>\n<p>{}</p>\n", html_escape(ret)));
    }

    if let Some(example) = &doc.example {
        body.push_str(&format!(
            "<h2>Example</h2>\n<pre><code>{}</code></pre>\n",
            html_escape(example)
        ));
    }

    body
}

fn wrap_html(title: &str, body: &str) -> String {
    let title = html_escape(title);
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
</head>
<body>
{body}<script>
document.addEventListener("DOMContentLoaded", function () {{
  renderMathInElement(document.body, {{
    delimiters: [
      {{left: "\\[", right: "\\]", display: true}},
      {{left: "\\(", right: "\\)", display: false}}
    ]
  }});
}});
</script>
</body>
</html>
"#
    )
}

fn html_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::signatures::Doc;
    use crate::resolve::lockfile::{LockedPackage, Lockfile};

    const RBF_COV_SOURCE: &str = r#"// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @param x Vector of input locations.
// @param alpha Marginal standard deviation of the GP.
// @param rho Length-scale of the GP.
// @return An N x N positive semi-definite covariance matrix.
// @example rbf_cov(x, 1.0, 0.5)
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}

real jitter(real epsilon) {
  return epsilon;
}
"#;

    fn write_gps_package(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(
            dir.join("laplace.toml"),
            "name = \"gps\"\nversion = \"1.0.0\"\nexports = [\"rbf_cov\"]\n",
        )
        .unwrap();
        fs::write(dir.join("gps.stan"), RBF_COV_SOURCE).unwrap();
    }

    #[test]
    fn write_sidecar_then_read_docs_json_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path().join("gps").join("1.0.0");
        write_gps_package(&pkg_dir);

        let docs = write_sidecar(&pkg_dir, "gps", "1.0.0").unwrap();
        assert_eq!(docs.package, "gps");
        assert_eq!(docs.version, "1.0.0");
        // sorted by name: jitter before rbf_cov
        assert_eq!(
            docs.functions
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["jitter", "rbf_cov"]
        );

        let reloaded = read_docs_json(&pkg_dir.join("docs.json")).unwrap();
        assert_eq!(reloaded.functions.len(), 2);
        let rbf = reloaded
            .functions
            .iter()
            .find(|f| f.name == "rbf_cov")
            .unwrap();
        assert_eq!(
            rbf.doc.as_ref().unwrap().brief.as_deref(),
            Some("Squared exponential (RBF) covariance matrix.")
        );
    }

    fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let lockfile_path = tmp.path().join("laplace.lock");
        let cache_root = tmp.path().join("cache");
        (tmp, lockfile_path, cache_root)
    }

    #[test]
    fn lookup_finds_a_documented_function() {
        let (_tmp, lockfile_path, cache_root) = fixture();
        let pkg_dir = cache_root.join("gps").join("1.0.0");
        write_gps_package(&pkg_dir);
        write_sidecar(&pkg_dir, "gps", "1.0.0").unwrap();

        lockfile::write_lockfile(
            &lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    "sha256:whatever",
                    "registry",
                )],
            },
        )
        .unwrap();

        let sigs = lookup(&lockfile_path, &cache_root, "gps", "rbf_cov").unwrap();
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "rbf_cov");
        assert!(sigs[0].doc.is_some());
    }

    #[test]
    fn lookup_rebuilds_a_sidecar_from_an_older_extractor() {
        let (_tmp, lockfile_path, cache_root) = fixture();
        let pkg_dir = cache_root.join("gps").join("1.0.0");
        write_gps_package(&pkg_dir);
        // What an older laplace wrote: no `format`, and the doc lost.
        fs::write(
            pkg_dir.join("docs.json"),
            r#"{"package":"gps","version":"1.0.0","functions":[{"name":"rbf_cov","params":[],"return_type":"matrix","doc":null}]}"#,
        )
        .unwrap();
        lockfile::write_lockfile(
            &lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    "sha256:whatever",
                    "registry",
                )],
            },
        )
        .unwrap();

        let sigs = lookup(&lockfile_path, &cache_root, "gps", "rbf_cov").unwrap();
        assert!(sigs[0].doc.is_some());
        let rewritten = read_docs_json(&pkg_dir.join("docs.json")).unwrap();
        assert_eq!(rewritten.format, DOCS_FORMAT);
    }

    #[test]
    fn lookup_package_not_in_lockfile_errors() {
        let (_tmp, lockfile_path, cache_root) = fixture();
        let err = lookup(&lockfile_path, &cache_root, "gps", "rbf_cov").unwrap_err();
        assert!(matches!(err, DocsError::PackageNotInLockfile { .. }));
    }

    #[test]
    fn lookup_locked_but_not_installed_errors() {
        let (_tmp, lockfile_path, cache_root) = fixture();
        lockfile::write_lockfile(
            &lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    "sha256:whatever",
                    "registry",
                )],
            },
        )
        .unwrap();

        let err = lookup(&lockfile_path, &cache_root, "gps", "rbf_cov").unwrap_err();
        assert!(matches!(err, DocsError::PackageNotInstalled { .. }));
    }

    #[test]
    fn lookup_function_not_found_errors() {
        let (_tmp, lockfile_path, cache_root) = fixture();
        let pkg_dir = cache_root.join("gps").join("1.0.0");
        write_gps_package(&pkg_dir);
        write_sidecar(&pkg_dir, "gps", "1.0.0").unwrap();
        lockfile::write_lockfile(
            &lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    "sha256:whatever",
                    "registry",
                )],
            },
        )
        .unwrap();

        let err = lookup(&lockfile_path, &cache_root, "gps", "matern_cov").unwrap_err();
        assert!(matches!(err, DocsError::FunctionNotFound { .. }));
    }

    #[test]
    fn render_includes_brief_params_return_and_example() {
        let sig = FunctionSig {
            name: "rbf_cov".to_string(),
            params: vec![
                ("x".to_string(), "vector".to_string()),
                ("alpha".to_string(), "real".to_string()),
            ],
            return_type: "matrix".to_string(),
            doc: Some(Doc {
                brief: Some("Squared exponential covariance.".to_string()),
                params: vec![
                    ("x".to_string(), "Input locations.".to_string()),
                    ("alpha".to_string(), "Marginal std dev.".to_string()),
                ],
                return_doc: Some("An N x N matrix.".to_string()),
                example: Some("rbf_cov(x, 1.0)".to_string()),
                math: None,
            }),
            ..Default::default()
        };

        let rendered = render("gps", &sig);
        assert!(rendered.starts_with("gps::rbf_cov(x: vector, alpha: real) -> matrix\n"));
        assert!(rendered.contains("Squared exponential covariance."));
        assert!(rendered.contains("Parameters:"));
        assert!(rendered.contains("x      Input locations."));
        assert!(rendered.contains("alpha  Marginal std dev."));
        assert!(rendered.contains("Returns:\n  An N x N matrix."));
        assert!(rendered.contains("Example:\n  rbf_cov(x, 1.0)"));
    }

    #[test]
    fn render_overloads_shows_every_signature_and_shares_the_doc() {
        let source = "// @laplace\n// @brief Hill curve.\nreal hill(real x) {\n  return x;\n}\n\
                      vector hill(vector x) {\n  return x;\n}\n\n\
                      // @laplace\n// @brief Matrix form.\nmatrix hill(matrix x) {\n  return x;\n}\n";
        let sigs = extract_signatures(source);
        let rendered = render_overloads("kinetics", &sigs);

        assert!(
            rendered.starts_with("kinetics::hill has 3 overloads\n"),
            "{rendered}"
        );
        assert!(rendered.contains(
            "kinetics::hill(x: real) -> real\nkinetics::hill(x: vector) -> vector\n\nHill curve.\n"
        ), "{rendered}");
        assert!(
            rendered.contains("kinetics::hill(x: matrix) -> matrix\n\nMatrix form.\n"),
            "{rendered}"
        );
        assert_eq!(rendered.matches("Hill curve.").count(), 1);
        assert!(!rendered.contains("no @laplace documentation"));

        let html = render_html_overloads("kinetics", &sigs);
        assert_eq!(html.matches("<h1>").count(), 3);
    }

    #[test]
    fn render_overloads_of_a_single_function_matches_render() {
        let sigs =
            extract_signatures("// @laplace\n// @brief One.\nreal f(real x) {\n  return x;\n}\n");
        assert_eq!(render_overloads("p", &sigs), render("p", &sigs[0]));
    }

    #[test]
    fn render_without_doc_notes_no_documentation_available() {
        let sig = FunctionSig {
            name: "jitter".to_string(),
            params: vec![("epsilon".to_string(), "real".to_string())],
            return_type: "real".to_string(),
            doc: None,
            ..Default::default()
        };

        let rendered = render("gps", &sig);
        assert!(rendered.starts_with("gps::jitter(epsilon: real) -> real\n"));
        assert!(rendered.contains("no @laplace documentation available"));
    }

    const RBF_COV_WITH_MATH_SOURCE: &str = r#"// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @math k(x, x') = \alpha^2 \exp\left(
//   -\frac{(x - x')^2}{2 \rho^2}
// \right)
// @param x Vector of input locations.
// @return An N x N positive semi-definite covariance matrix.
// @example matrix k = rbf_cov(x, 1.0, 0.5);
//   print(k);
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}

real jitter(real epsilon) {
  return epsilon;
}
"#;

    fn write_gps_package_with_math(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(
            dir.join("laplace.toml"),
            "name = \"gps\"\nversion = \"1.0.0\"\nexports = [\"rbf_cov\"]\n",
        )
        .unwrap();
        fs::write(dir.join("gps.stan"), RBF_COV_WITH_MATH_SOURCE).unwrap();
    }

    #[test]
    fn math_round_trips_through_docs_json_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path().join("gps").join("1.0.0");
        write_gps_package_with_math(&pkg_dir);

        write_sidecar(&pkg_dir, "gps", "1.0.0").unwrap();
        let reloaded = read_docs_json(&pkg_dir.join("docs.json")).unwrap();
        let rbf = reloaded
            .functions
            .iter()
            .find(|f| f.name == "rbf_cov")
            .unwrap();

        let expected_math =
            "k(x, x') = \\alpha^2 \\exp\\left(\n-\\frac{(x - x')^2}{2 \\rho^2}\n\\right)";
        assert_eq!(
            rbf.doc.as_ref().unwrap().math.as_deref(),
            Some(expected_math)
        );

        let rendered = render("gps", rbf);
        assert!(rendered.contains("Math:\n"));
        for line in expected_math.lines() {
            assert!(
                rendered.contains(line),
                "rendered output missing math line: {line}"
            );
        }

        let html = render_html("gps", rbf);
        assert!(html.contains("<span class=\"math\">"));
        // `&` in the LaTeX must be HTML-escaped so the browser hands KaTeX
        // back the original raw string via textContent.
        assert!(html.contains(&html_escape(expected_math)));
        assert!(html.contains("katex"));
    }

    #[test]
    fn no_math_tag_means_no_math_section_anywhere() {
        // `write_gps_package` (defined above) uses RBF_COV_SOURCE, which is
        // documented (@brief/@param/@return/@example) but has no @math tag.
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path().join("gps").join("1.0.0");
        write_gps_package(&pkg_dir);

        write_sidecar(&pkg_dir, "gps", "1.0.0").unwrap();
        let reloaded = read_docs_json(&pkg_dir.join("docs.json")).unwrap();
        let rbf = reloaded
            .functions
            .iter()
            .find(|f| f.name == "rbf_cov")
            .unwrap();
        assert_eq!(rbf.doc.as_ref().unwrap().math, None);

        let rendered = render("gps", rbf);
        assert!(!rendered.contains("Math:"));

        let html = render_html("gps", rbf);
        assert!(!html.contains("class=\"math\""));

        // docs.json must omit the `math` key entirely (not serialize it as
        // `null`) for a documented function with no @math tag.
        let raw_json = fs::read_to_string(pkg_dir.join("docs.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw_json).unwrap();
        let rbf_json = parsed["functions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == "rbf_cov")
            .unwrap();
        assert!(rbf_json["doc"].get("math").is_none());
    }

    #[test]
    fn multiline_example_preserves_line_breaks_through_json_and_render() {
        let tmp = tempfile::tempdir().unwrap();
        let pkg_dir = tmp.path().join("gps").join("1.0.0");
        write_gps_package_with_math(&pkg_dir);

        write_sidecar(&pkg_dir, "gps", "1.0.0").unwrap();
        let reloaded = read_docs_json(&pkg_dir.join("docs.json")).unwrap();
        let rbf = reloaded
            .functions
            .iter()
            .find(|f| f.name == "rbf_cov")
            .unwrap();

        let expected_example = "matrix k = rbf_cov(x, 1.0, 0.5);\nprint(k);";
        assert_eq!(
            rbf.doc.as_ref().unwrap().example.as_deref(),
            Some(expected_example)
        );

        let rendered = render("gps", rbf);
        assert!(rendered.contains("Example:\n  matrix k = rbf_cov(x, 1.0, 0.5);\n  print(k);\n"));

        let html = render_html("gps", rbf);
        assert!(html.contains(&format!(
            "<pre><code>{}</code></pre>",
            html_escape(expected_example)
        )));
    }
}
