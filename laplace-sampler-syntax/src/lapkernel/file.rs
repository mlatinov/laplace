//! A `.lapkernel` file, parsed and checked (Phase 2.4). laplace-sample runs
//! one as an `InterpretedKernel`.

use std::path::Path;

use super::ast::KernelFile;
use super::builtins::Registry;
use super::check::{check, AdaptKind, Checked};
use super::parse::parse;

/// A parsed `.lapkernel` file, checked without a dimension (types and
/// structure); [`LapKernelFile::check`] re-checks it with the block size.
#[derive(Clone, Debug, PartialEq)]
pub struct LapKernelFile {
    /// The path as given (messages, provenance).
    pub path: String,
    pub source: String,
    pub parsed: KernelFile,
}

fn errors(es: Vec<super::LapError>) -> anyhow::Error {
    let lines: Vec<String> = es.iter().map(|e| e.to_string()).collect();
    anyhow::anyhow!("{}", lines.join("\n"))
}

impl LapKernelFile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        Self::from_source(&path.display().to_string(), source)
    }

    pub fn from_source(path: &str, source: String) -> anyhow::Result<Self> {
        let parsed = parse(path, &source).map_err(|e| anyhow::anyhow!("{e}"))?;
        let me = Self {
            path: path.to_string(),
            source,
            parsed,
        };
        me.check(None)?;
        Ok(me)
    }

    pub fn check(&self, dim: Option<usize>) -> anyhow::Result<Checked> {
        check(&self.path, &self.parsed, dim, &Registry::standard()).map_err(errors)
    }

    pub fn name(&self) -> &str {
        &self.parsed.name
    }

    /// Names of the tuning values `--set` may change (scalars only).
    pub fn settable(&self) -> Vec<String> {
        self.parsed
            .tuning
            .iter()
            .filter(|d| matches!(d.ty, super::ast::TypeExpr::Int | super::ast::TypeExpr::Real))
            .map(|d| d.name.clone())
            .collect()
    }

    pub fn has_adapt(&self, kind: AdaptKind) -> bool {
        self.check(None)
            .map(|c| match kind {
                AdaptKind::StepSize => c.step_size.is_some(),
                AdaptKind::Metric => c.metric.is_some(),
            })
            .unwrap_or(false)
    }
}
