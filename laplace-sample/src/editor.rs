//! Library API for editor support: parse and check `.lapkernel` and
//! `.lapsampler` source text and get diagnostics back, without BridgeStan,
//! a C++ toolchain or a compiled model.
//!
//! Both functions take the text as the editor holds it (it need not be
//! saved) and never touch a model. They run exactly the checks
//! `laplace sampler check` runs before it needs a model; the one check
//! left out is block coverage (D2: every model parameter in exactly one
//! block), which needs the model's parameters. See
//! `docs/sampler/lsp-integration.md`.
//!
//! ```
//! use laplace_sample::editor::{check_lapkernel, Severity};
//! let ds = check_lapkernel("k.lapkernel", "kernel k {\n  step {\n    move(nope);\n  }\n}\n", None);
//! assert_eq!(ds.len(), 1);
//! assert_eq!((ds[0].line, ds[0].column, ds[0].severity), (3, 10, Severity::Error));
//! assert_eq!(ds[0].message, "unknown name `nope`");
//! ```

use std::path::Path;

use crate::lapkernel::builtins::Registry;
use crate::lapkernel::{check, parse, LapError};
use crate::lapsampler::SamplerFile;
pub use crate::libraries::Project;

/// How bad a diagnostic is. Every diagnostic the checkers produce today is
/// an [`Severity::Error`] (the file would be refused by `laplace sample`);
/// `Warning` is part of the API so that adding warnings later does not
/// change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// One problem, at a position.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    /// The file the problem is in: the `path` passed in, or another file it
    /// refers to (a library kernel is reported as
    /// `<package>/kernels/<name>.lapkernel`, a local one by its path).
    pub file: String,
    /// 1-based line and column (columns count characters) of where the
    /// problem starts. A position is a point, not a range. `0, 0` means
    /// the whole file (it could not be read).
    pub line: usize,
    pub column: usize,
    pub severity: Severity,
    /// One line, except when a `.lapsampler` refers to a kernel file that
    /// does not check: then the message is `<kernel> does not check:`
    /// followed by that file's own diagnostics, one per indented line, each
    /// with its own `file:line:column`.
    pub message: String,
}

impl From<LapError> for Diagnostic {
    fn from(e: LapError) -> Self {
        Self {
            file: e.file,
            line: e.span.line,
            column: e.span.col,
            severity: Severity::Error,
            message: e.message,
        }
    }
}

/// Parse and check one `.lapkernel`. `dim` is the size of the block the
/// kernel will update, if known (it only sharpens size checks); pass
/// `None` in an editor. Sorted by position; empty means the kernel checks.
pub fn check_lapkernel(path: &str, source: &str, dim: Option<usize>) -> Vec<Diagnostic> {
    let parsed = match parse::parse(path, source) {
        Ok(p) => p,
        // A syntax error stops parsing: there is exactly one.
        Err(e) => return vec![e.into()],
    };
    match check::check(path, &parsed, dim, &Registry::standard()) {
        Ok(_) => vec![],
        Err(es) => es.into_iter().map(Diagnostic::from).collect(),
    }
}

/// Parse and check one `.lapsampler`: syntax, structure, run settings,
/// built-in kernels and their settings, local `.lapkernel` files (read
/// from disk, relative to `dir`, and checked), library kernels (`import`
/// resolved in `project`'s laplace.lock, as `laplace sample` does), the
/// `adapt()` contracts. Block coverage needs a model and is not checked.
/// Sorted by file and position; empty means the file checks.
pub fn check_lapsampler(
    path: &str,
    source: &str,
    dir: &Path,
    project: &Project,
) -> Vec<Diagnostic> {
    match SamplerFile::from_source_in(path, source.to_string(), dir, project) {
        Ok(_) => vec![],
        Err(es) => es.into_iter().map(Diagnostic::from).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kernel_that_checks_has_no_diagnostics() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/mala.lapkernel"
        ))
        .unwrap();
        assert_eq!(check_lapkernel("mala.lapkernel", &src, None), vec![]);
    }

    #[test]
    fn kernel_syntax_and_check_errors_carry_positions() {
        let ds = check_lapkernel(
            "k.lapkernel",
            "kernel k {\n  step {\n    move(q)\n  }\n}\n",
            None,
        );
        assert_eq!(ds.len(), 1);
        assert_eq!(
            (ds[0].file.as_str(), ds[0].line, ds[0].column),
            ("k.lapkernel", 4, 3)
        );
        assert!(
            ds[0].message.starts_with("syntax error"),
            "{}",
            ds[0].message
        );

        let ds = check_lapkernel(
            "k.lapkernel",
            "kernel k {\n  tuning {\n    real eps = 1 @adapt(step_size);\n  }\n  step {\n    move(x);\n  }\n}\n",
            None,
        );
        let at: Vec<(usize, usize)> = ds.iter().map(|d| (d.line, d.column)).collect();
        assert_eq!(at.len(), 2, "{ds:#?}");
        assert!(at.windows(2).all(|w| w[0] <= w[1]), "sorted: {at:?}");
        assert!(ds.iter().all(|d| d.severity == Severity::Error));
        assert!(
            ds.iter().any(|d| d.message.contains("target_accept")),
            "{ds:#?}"
        );
    }

    #[test]
    fn sampler_diagnostics_need_no_model() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let project = Project {
            dir: dir.to_path_buf(),
            cache_root: dir.join("no-cache"),
        };
        let ok = "sampler s {\n  kernel {\n    block(beta) : adapt(./kernels/mala.lapkernel)\n    rest : nuts()\n  }\n}\n";
        assert_eq!(check_lapsampler("s.lapsampler", ok, dir, &project), vec![]);
        let bad = "sampler s {\n  kernel {\n    block(beta) : nuts(step = 1)\n    rest : frob()\n  }\n}\n";
        let ds = check_lapsampler("s.lapsampler", bad, dir, &project);
        let at: Vec<(usize, usize)> = ds.iter().map(|d| (d.line, d.column)).collect();
        assert_eq!(at, vec![(3, 24), (4, 12)], "{ds:#?}");
        assert!(
            ds[0].message.contains("nuts() has no setting `step`"),
            "{}",
            ds[0].message
        );
        assert!(
            ds[1].message.starts_with("unknown kernel `frob`"),
            "{}",
            ds[1].message
        );
    }
}
