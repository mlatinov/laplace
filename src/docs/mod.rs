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

use crate::manifest;
use crate::parser::signatures::{extract_signatures, FunctionSig};
use crate::resolve::lockfile::{self, LockfileError};

/// The full `docs.json` sidecar for one installed package version: every
/// top-level function Task 2 could extract, sorted by name for determinism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageDocs {
    pub package: String,
    pub version: String,
    pub functions: Vec<FunctionSig>,
}

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
}

/// Extract every function signature from `package_dir`'s `.stan` file(s)
/// and write them to `package_dir/docs.json`. Called after a package is
/// copied into the cache, so `laplace doc` never has to re-parse Stan
/// source at lookup time.
pub fn write_sidecar(package_dir: &Path, name: &str, version: &str) -> Result<PackageDocs, DocsError> {
    let source = manifest::read_package_stan_source(package_dir).map_err(|source| DocsError::Io {
        path: package_dir.to_path_buf(),
        source,
    })?;

    let mut functions = extract_signatures(&source);
    functions.sort_by(|a, b| a.name.cmp(&b.name));

    let docs = PackageDocs {
        package: name.to_string(),
        version: version.to_string(),
        functions,
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
/// project's lockfile, load its `docs.json`, and find `func`.
pub fn lookup(
    lockfile_path: &Path,
    cache_root: &Path,
    package: &str,
    func: &str,
) -> Result<FunctionSig, DocsError> {
    let lock = lockfile::read_lockfile(lockfile_path)?;
    let Some(locked) = lock.packages.iter().find(|p| p.name == package) else {
        return Err(DocsError::PackageNotInLockfile {
            package: package.to_string(),
        });
    };

    let package_dir = cache_root.join(&locked.name).join(&locked.version);
    let docs_path = package_dir.join("docs.json");
    if !docs_path.is_file() {
        return Err(DocsError::PackageNotInstalled {
            package: package.to_string(),
            path: package_dir,
        });
    }

    let docs = read_docs_json(&docs_path)?;
    docs.functions
        .into_iter()
        .find(|f| f.name == func)
        .ok_or_else(|| DocsError::FunctionNotFound {
            package: package.to_string(),
            func: func.to_string(),
        })
}

/// Pretty-print a function's signature and doc comment (if any) for
/// terminal display. A function with no `// @laplace` doc comment still
/// renders its signature, with a note that no docs are available.
pub fn render(package: &str, sig: &FunctionSig) -> String {
    let params_sig = sig
        .params
        .iter()
        .map(|(name, ty)| format!("{name}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");

    let mut out = format!("{package}::{}({params_sig}) -> {}\n", sig.name, sig.return_type);

    let Some(doc) = &sig.doc else {
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
        let width = doc.params.iter().map(|(name, _)| name.len()).max().unwrap_or(0);
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
    let params_sig = sig
        .params
        .iter()
        .map(|(name, ty)| format!("{name}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    let title = format!("{package}::{}({params_sig}) -> {}", sig.name, sig.return_type);

    let mut body = format!("<h1>{}</h1>\n", html_escape(&title));

    let Some(doc) = &sig.doc else {
        body.push_str("<p><em>no @laplace documentation available for this function</em></p>\n");
        return wrap_html(&title, &body);
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

    wrap_html(&title, &body)
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
            docs.functions.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
            vec!["jitter", "rbf_cov"]
        );

        let reloaded = read_docs_json(&pkg_dir.join("docs.json")).unwrap();
        assert_eq!(reloaded.functions.len(), 2);
        let rbf = reloaded.functions.iter().find(|f| f.name == "rbf_cov").unwrap();
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
                packages: vec![LockedPackage {
                    name: "gps".to_string(),
                    version: "1.0.0".to_string(),
                    checksum: "sha256:whatever".to_string(),
                    source: "registry".to_string(),
                }],
            },
        )
        .unwrap();

        let sig = lookup(&lockfile_path, &cache_root, "gps", "rbf_cov").unwrap();
        assert_eq!(sig.name, "rbf_cov");
        assert!(sig.doc.is_some());
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
                packages: vec![LockedPackage {
                    name: "gps".to_string(),
                    version: "1.0.0".to_string(),
                    checksum: "sha256:whatever".to_string(),
                    source: "registry".to_string(),
                }],
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
                packages: vec![LockedPackage {
                    name: "gps".to_string(),
                    version: "1.0.0".to_string(),
                    checksum: "sha256:whatever".to_string(),
                    source: "registry".to_string(),
                }],
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
    fn render_without_doc_notes_no_documentation_available() {
        let sig = FunctionSig {
            name: "jitter".to_string(),
            params: vec![("epsilon".to_string(), "real".to_string())],
            return_type: "real".to_string(),
            doc: None,
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
        let rbf = reloaded.functions.iter().find(|f| f.name == "rbf_cov").unwrap();

        let expected_math = "k(x, x') = \\alpha^2 \\exp\\left(\n-\\frac{(x - x')^2}{2 \\rho^2}\n\\right)";
        assert_eq!(rbf.doc.as_ref().unwrap().math.as_deref(), Some(expected_math));

        let rendered = render("gps", rbf);
        assert!(rendered.contains("Math:\n"));
        for line in expected_math.lines() {
            assert!(rendered.contains(line), "rendered output missing math line: {line}");
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
        let rbf = reloaded.functions.iter().find(|f| f.name == "rbf_cov").unwrap();
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
        let rbf = reloaded.functions.iter().find(|f| f.name == "rbf_cov").unwrap();

        let expected_example = "matrix k = rbf_cov(x, 1.0, 0.5);\nprint(k);";
        assert_eq!(rbf.doc.as_ref().unwrap().example.as_deref(), Some(expected_example));

        let rendered = render("gps", rbf);
        assert!(rendered.contains("Example:\n  matrix k = rbf_cov(x, 1.0, 0.5);\n  print(k);\n"));

        let html = render_html("gps", rbf);
        assert!(html.contains(&format!(
            "<pre><code>{}</code></pre>",
            html_escape(expected_example)
        )));
    }
}
