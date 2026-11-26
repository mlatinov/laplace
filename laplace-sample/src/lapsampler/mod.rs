//! `.lapsampler` files: composing kernels over parameter blocks (Level 2,
//! Phase 3.5).
//!
//! Parsing and the model-free checks are in laplace-sampler-syntax
//! ([`SamplerFile`]), so the editor can use them without this crate's
//! runtime; they are re-exported here. This module adds what needs the
//! model: [`SamplerRuntime::blocks`] checks the blocks against the model's
//! parameters (coverage, D2), and [`SamplerRuntime::instantiate`] builds
//! the kernel tree. `laplace sampler check` runs both, the second when
//! given a model.
//!
//! Every error is `file:line:column`.

pub use laplace_sampler_syntax::lapsampler::*;

use std::sync::Arc;

use crate::adapt::Adapt;
use crate::blocks::{Block, Param};
use crate::combinators::{Mix, On, Repeat, Seq};
use crate::csv::KernelLine;
use crate::kernel::Kernel;
use crate::kernels::nuts::Nuts;
use crate::lapkernel::ast::Span;
use crate::lapkernel::kernel::{InterpretedKernel, LapKernelFile};
use crate::lapkernel::LapError;
use crate::runner::{BuildKernel, BuiltIn, SamplerSpec};
use crate::target::Target;
use parse::{ArgValue, KExpr, LineTarget};

/// The kernel tree for one chain, plus its provenance lines.
pub struct Built {
    pub kernel: Box<dyn Kernel>,
    pub lines: Vec<KernelLine>,
    /// Labels of the kernel lines, in order (CSV per-block columns).
    pub labels: Vec<String>,
    /// Each line's number of unconstrained coordinates.
    pub dims: Vec<usize>,
}

struct Ctx<'a> {
    file: &'a str,
    errors: Vec<LapError>,
}

impl Ctx<'_> {
    fn err(&mut self, span: Span, msg: impl Into<String>) {
        self.errors.push(LapError::new(self.file, span, msg));
    }
}

/// What a checked [`SamplerFile`] needs the model for.
pub trait SamplerRuntime {
    /// Blocks of each kernel line against the model's parameters: unknown
    /// names, overlaps, `rest`, and coverage (D2: every parameter in a
    /// block). Errors point at the line.
    fn blocks(&self, params: &[Param]) -> Result<Vec<Block>, Vec<LapError>>;

    /// The kernel tree for one chain: seq(on(block, kernel), ...).
    /// `full` is the whole-model target (nuts uses it for its block target).
    fn instantiate(
        &self,
        params: &[Param],
        full: Arc<dyn Target>,
        warmup: u64,
        draws: u64,
    ) -> anyhow::Result<Built>;
}

impl SamplerRuntime for SamplerFile {
    fn blocks(&self, params: &[Param]) -> Result<Vec<Block>, Vec<LapError>> {
        let mut c = Ctx {
            file: &self.path,
            errors: vec![],
        };
        let mut used: Vec<(String, Span)> = vec![];
        let mut out = vec![];
        for l in &self.ast.lines {
            match &l.target {
                LineTarget::Block {
                    params: names,
                    label,
                } => {
                    for n in names {
                        if !params.iter().any(|p| &p.name == n) {
                            c.err(
                                l.span,
                                format!(
                                    "the model has no parameter `{n}`; its parameters are: {}",
                                    params
                                        .iter()
                                        .map(|p| p.name.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                            );
                        } else if let Some((_, sp)) = used.iter().find(|(u, _)| u == n) {
                            let sp = *sp;
                            c.err(l.span, format!("`{n}` is already in the block at {sp}; a parameter belongs to one block"));
                        } else {
                            used.push((n.clone(), l.span));
                        }
                    }
                    if let Ok(b) = Block::new(params, names, label.as_deref()) {
                        out.push(b);
                    }
                }
                LineTarget::Rest => {
                    let left: Vec<String> = params
                        .iter()
                        .filter(|p| !used.iter().any(|(u, _)| *u == p.name))
                        .map(|p| p.name.clone())
                        .collect();
                    if left.is_empty() {
                        c.err(
                            l.span,
                            "`rest` covers no parameters: every parameter is already in a block",
                        );
                    } else {
                        out.push(Block::new(params, &left, Some("rest")).expect("own names"));
                        used.extend(left.into_iter().map(|n| (n, l.span)));
                    }
                }
            }
        }
        let uncovered: Vec<&str> = params
            .iter()
            .filter(|p| !used.iter().any(|(u, _)| *u == p.name))
            .map(|p| p.name.as_str())
            .collect();
        if !uncovered.is_empty() {
            let sp = self.ast.kernel_block.unwrap_or(self.ast.span);
            c.err(
                sp,
                format!(
                    "parameter(s) {} are in no block (D2); add them to a block or end with `rest : <kernel>`",
                    uncovered.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
                ),
            );
        }
        if c.errors.is_empty() {
            Ok(out)
        } else {
            Err(c.errors)
        }
    }

    fn instantiate(
        &self,
        params: &[Param],
        full: Arc<dyn Target>,
        warmup: u64,
        draws: u64,
    ) -> anyhow::Result<Built> {
        let blocks = self.blocks(params).map_err(errors)?;
        let mut items: Vec<Box<dyn Kernel>> = vec![];
        let mut lines = vec![];
        let mut labels = vec![];
        let mut dims = vec![];
        for (l, b) in self.ast.lines.iter().zip(blocks) {
            dims.push(b.dim());
            let inner = self.build(&l.kernel, &b, 1.0, &full, warmup, draws)?;
            let (version, sources) = self.provenance(&l.kernel);
            lines.push(KernelLine {
                label: b.label.clone(),
                name: render(&l.kernel),
                version,
                source: format!("{}\n{sources}", render(&l.kernel)),
            });
            labels.push(b.label.clone());
            items.push(Box::new(On { block: b, inner }));
        }
        Ok(Built {
            kernel: Box::new(Seq { items }),
            lines,
            labels,
            dims,
        })
    }
}

/// Building one kernel line (private to [`SamplerRuntime::instantiate`]).
trait Build {
    fn kernel_file(&self, path: &str) -> Arc<LapKernelFile>;
    fn provenance(&self, k: &KExpr) -> (String, String);
    fn build(
        &self,
        k: &KExpr,
        block: &Block,
        uses: f64,
        full: &Arc<dyn Target>,
        warmup: u64,
        draws: u64,
    ) -> anyhow::Result<Box<dyn Kernel>>;
}

impl Build for SamplerFile {
    fn kernel_file(&self, path: &str) -> Arc<LapKernelFile> {
        self.kernels
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, f)| f.clone())
            .expect("loaded at check time")
    }

    /// The provenance version of a kernel line and its leaves' sources.
    /// Built-ins only: laplace-sample's version. Otherwise every library
    /// kernel's `package@version` (from laplace.lock), sorted, joined with
    /// `+`, and `local` when a local file is among the leaves.
    fn provenance(&self, k: &KExpr) -> (String, String) {
        let mut local = false;
        let mut libs: Vec<String> = vec![];
        let mut sources = String::new();
        let mut walk = vec![k];
        while let Some(k) = walk.pop() {
            match k {
                KExpr::BuiltIn { name, .. } => {
                    let b = BuiltIn::parse(name).expect("checked");
                    sources += &builtin_source(b);
                }
                KExpr::Path { path, .. } => {
                    local = true;
                    sources += &self.kernel_file(path).source;
                }
                KExpr::Library { package, .. } => {
                    let lib = self
                        .libraries
                        .iter()
                        .find(|l| &l.name == package)
                        .expect("resolved at check time");
                    libs.push(format!("{}@{}", lib.name, lib.version));
                    sources += &self.kernel_file(&kernel_key(k).unwrap()).source;
                }
                KExpr::Seq(ks, _) => walk.extend(ks.iter()),
                KExpr::Mix(ks, _) => walk.extend(ks.iter().map(|(_, k)| k)),
                KExpr::Repeat(_, k, _) | KExpr::Adapt(k, _, _) | KExpr::On(_, k, _) => walk.push(k),
            }
        }
        libs.sort();
        libs.dedup();
        if local {
            libs.push("local".into());
        }
        let version = if libs.is_empty() {
            crate::VERSION.to_string()
        } else {
            libs.join("+")
        };
        (version, sources)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn build(
        &self,
        k: &KExpr,
        block: &Block,
        uses: f64,
        full: &Arc<dyn Target>,
        warmup: u64,
        draws: u64,
    ) -> anyhow::Result<Box<dyn Kernel>> {
        let steps = |n: u64| (n as f64 * uses).round() as u64;
        Ok(match k {
            KExpr::BuiltIn { name, args, .. } => {
                let b = BuiltIn::parse(name)?;
                let sets: Vec<String> = args
                    .iter()
                    .map(|a| match a.value {
                        ArgValue::Number(v) => format!("{}={v}", a.name),
                        ArgValue::Bool(v) => format!("{}={v}", a.name),
                    })
                    .collect();
                let spec = SamplerSpec::new(b, &sets)?;
                if b == BuiltIn::Nuts {
                    let mut n = Nuts::for_block(
                        full.clone(),
                        block.indices.clone(),
                        steps(warmup),
                        steps(draws),
                    );
                    n.set_initial_step_size(spec.f64("step_size")?);
                    n.set_target_accept(spec.f64("target_accept")?);
                    n.set_max_depth(spec.u64("max_depth")?);
                    Box::new(n)
                } else {
                    spec.kernel(full.clone(), steps(warmup), steps(draws))?
                }
            }
            KExpr::Path { path, args, .. } => {
                let sets: Vec<(String, f64)> = args
                    .iter()
                    .filter_map(|a| Some((a.name.clone(), number(a)?)))
                    .collect();
                Box::new(InterpretedKernel::new(
                    self.kernel_file(path),
                    block.dim(),
                    &sets,
                )?)
            }
            KExpr::Library { args, .. } => {
                let sets: Vec<(String, f64)> = args
                    .iter()
                    .filter_map(|a| Some((a.name.clone(), number(a)?)))
                    .collect();
                Box::new(InterpretedKernel::new(
                    self.kernel_file(&kernel_key(k).unwrap()),
                    block.dim(),
                    &sets,
                )?)
            }
            KExpr::Seq(ks, _) => Box::new(Seq {
                items: ks
                    .iter()
                    .map(|k| self.build(k, block, uses, full, warmup, draws))
                    .collect::<anyhow::Result<_>>()?,
            }),
            KExpr::Mix(ks, _) => {
                let total: f64 = ks.iter().map(|(w, _)| w).sum();
                let items = ks
                    .iter()
                    .map(|(w, k)| self.build(k, block, uses * w / total, full, warmup, draws))
                    .collect::<anyhow::Result<_>>()?;
                Box::new(Mix::new(
                    &ks.iter().map(|(w, _)| *w).collect::<Vec<_>>(),
                    items,
                )?)
            }
            KExpr::Repeat(n, k, _) => Box::new(Repeat {
                n: *n as u64,
                inner: self.build(k, block, uses * n, full, warmup, draws)?,
            }),
            KExpr::Adapt(k, args, _) => {
                let inner = self.build(k, block, uses, full, warmup, draws)?;
                let ta = args
                    .iter()
                    .find(|a| a.name == "target_accept")
                    .and_then(number);
                Box::new(Adapt::new(inner, ta, steps(warmup))?)
            }
            KExpr::On(..) => unreachable!("rejected at check time"),
        })
    }
}

fn builtin_source(b: BuiltIn) -> String {
    match b {
        BuiltIn::Nuts => include_str!("../kernels/nuts.rs"),
        BuiltIn::Rwm => include_str!("../kernels/rwm.rs"),
        BuiltIn::Mala => include_str!("../kernels/mala.rs"),
        BuiltIn::Hmc => include_str!("../kernels/hmc.rs"),
    }
    .to_string()
}

#[cfg(test)]
mod tests;
