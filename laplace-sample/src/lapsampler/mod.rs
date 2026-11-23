//! `.lapsampler` files: composing kernels over parameter blocks (Level 2,
//! Phase 3.5).
//!
//! [`SamplerFile::load`] parses a file and checks everything that does not
//! need the model: kernel names and settings, local `.lapkernel` files
//! (parsed and checked), library kernels (`import` + `pkg::kernel`, resolved
//! through the project's laplace.lock, Phase 4.2), the adapt contracts,
//! structure. [`SamplerFile::instantiate`] then checks the blocks against
//! the model's parameters (coverage, D2) and builds the kernel tree.
//! `laplace sampler check` runs both, the second when given a model.
//!
//! Every error is `file:line:column`.

pub mod parse;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::adapt::Adapt;
use crate::blocks::{Block, Param};
use crate::combinators::{Mix, On, Repeat, Seq};
use crate::csv::KernelLine;
use crate::kernel::Kernel;
use crate::kernels::nuts::Nuts;
use crate::lapkernel::ast::Span;
use crate::lapkernel::check::AdaptKind;
use crate::lapkernel::kernel::{InterpretedKernel, LapKernelFile};
use crate::lapkernel::LapError;
use crate::libraries::{display_path, resolve_import, KernelPackage, Project};
use crate::runner::{BuiltIn, SamplerSpec};
use crate::target::Target;
use parse::{Arg, ArgValue, KExpr, LineTarget, SamplerAst};

/// Settings each built-in accepts inside a `.lapsampler`. Adaptation is
/// explicit at Level 2 (`adapt(...)`), so `adapt` and `target_accept` are
/// not kernel settings here, except for nuts, which adapts itself.
fn builtin_settings(b: BuiltIn) -> &'static [&'static str] {
    match b {
        BuiltIn::Nuts => &["step_size", "target_accept", "max_depth"],
        BuiltIn::Rwm | BuiltIn::Mala => &["step_size"],
        BuiltIn::Hmc => &["step_size", "steps", "jitter"],
    }
}

/// A loaded, model-free-checked `.lapsampler` file.
#[derive(Debug)]
pub struct SamplerFile {
    pub path: String,
    pub source: String,
    pub ast: SamplerAst,
    /// `.lapkernel` files by key: the path written in the sampler for a
    /// local file, `package::kernel` for a library kernel.
    kernels: Vec<(String, Arc<LapKernelFile>)>,
    /// The kernel packages this file imports, as locked and installed.
    libraries: Vec<KernelPackage>,
}

/// What a kernel expression is checked against.
struct Scope<'a> {
    /// The sampler file's directory (local paths resolve against it).
    dir: &'a Path,
    /// Every name in `library { }`, resolved or not.
    imported: Vec<String>,
    /// The imports that resolved.
    libraries: &'a [KernelPackage],
}

/// The key a `.lapkernel` leaf is stored under.
fn kernel_key(k: &KExpr) -> Option<String> {
    match k {
        KExpr::Path { path, .. } => Some(path.clone()),
        KExpr::Library { package, name, .. } => Some(format!("{package}::{name}")),
        _ => None,
    }
}

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

/// Canonical text of a kernel expression (provenance names).
pub fn render(k: &KExpr) -> String {
    let args = |a: &[Arg]| {
        a.iter()
            .map(|x| match x.value {
                ArgValue::Number(v) => format!("{} = {v}", x.name),
                ArgValue::Bool(v) => format!("{} = {v}", x.name),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    match k {
        KExpr::BuiltIn { name, args: a, .. } => format!("{name}({})", args(a)),
        KExpr::Path { path, args: a, .. } if a.is_empty() => path.clone(),
        KExpr::Path { path, args: a, .. } => format!("{path}({})", args(a)),
        KExpr::Library {
            package,
            name,
            args: a,
            ..
        } if a.is_empty() => format!("{package}::{name}"),
        KExpr::Library {
            package,
            name,
            args: a,
            ..
        } => format!("{package}::{name}({})", args(a)),
        KExpr::Seq(ks, _) => format!(
            "seq({})",
            ks.iter().map(render).collect::<Vec<_>>().join(", ")
        ),
        KExpr::Mix(ks, _) => format!(
            "mix({})",
            ks.iter()
                .map(|(w, k)| format!("{w}: {}", render(k)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        KExpr::Repeat(n, k, _) => format!("repeat({n}, {})", render(k)),
        KExpr::Adapt(k, a, _) if a.is_empty() => format!("adapt({})", render(k)),
        KExpr::Adapt(k, a, _) => format!("adapt({}, {})", render(k), args(a)),
        KExpr::On(b, k, _) => format!("on(block({}), {})", b.join(", "), render(k)),
    }
}

fn number(a: &Arg) -> Option<f64> {
    match a.value {
        ArgValue::Number(v) => Some(v),
        ArgValue::Bool(_) => None,
    }
}

impl SamplerFile {
    pub fn load(path: &Path) -> Result<Self, Vec<LapError>> {
        Self::load_in(path, &Project::current())
    }

    /// [`SamplerFile::load`] with library imports resolved in `project`.
    pub fn load_in(path: &Path, project: &Project) -> Result<Self, Vec<LapError>> {
        let shown = path.display().to_string();
        let source = std::fs::read_to_string(path).map_err(|e| {
            vec![LapError::new(
                &shown,
                Span::default(),
                format!("cannot read: {e}"),
            )]
        })?;
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::from_source_in(&shown, source, &dir, project)
    }

    /// Parse and check `source`; relative `.lapkernel` paths resolve
    /// against `dir` (the sampler file's directory), imports in the current
    /// directory's project.
    pub fn from_source(path: &str, source: String, dir: &Path) -> Result<Self, Vec<LapError>> {
        Self::from_source_in(path, source, dir, &Project::current())
    }

    pub fn from_source_in(
        path: &str,
        source: String,
        dir: &Path,
        project: &Project,
    ) -> Result<Self, Vec<LapError>> {
        let ast = parse::parse(path, &source).map_err(|e| vec![e])?;
        let mut c = Ctx {
            file: path,
            errors: vec![],
        };
        let mut kernels = vec![];

        // Structure.
        let mut seen: Vec<&str> = vec![];
        for (s, sp) in &ast.sections {
            if seen.contains(s) {
                c.err(*sp, format!("duplicate `{s}` block"));
            }
            seen.push(s);
        }
        match ast.kernel_block {
            None => c.err(
                ast.span,
                format!("sampler `{}` has no `kernel {{ }}` block", ast.name),
            ),
            Some(sp) if ast.lines.is_empty() => c.err(sp, "`kernel { }` is empty"),
            _ => {}
        }
        for (list, allowed) in [
            (&ast.warmup, &["iterations"][..]),
            (&ast.draws, &["iterations", "thin"][..]),
        ] {
            for s in list.iter() {
                if !allowed.contains(&s.name.as_str()) {
                    c.err(
                        s.span,
                        format!(
                            "unknown setting `{}`; expected {}",
                            s.name,
                            allowed.join(" or ")
                        ),
                    );
                } else if !(s.value >= 0.0 && s.value.fract() == 0.0) {
                    c.err(
                        s.span,
                        format!("`{}` must be a whole number, got {}", s.name, s.value),
                    );
                } else if s.name == "thin" && s.value < 1.0 {
                    c.err(s.span, "`thin` must be at least 1");
                }
            }
        }

        // Imports: kernel packages from laplace.lock, never a version here.
        let mut libraries: Vec<KernelPackage> = vec![];
        let mut imported: Vec<String> = vec![];
        for imp in &ast.imports {
            if imported.contains(&imp.name) {
                c.err(imp.span, format!("`{}` is imported twice", imp.name));
                continue;
            }
            imported.push(imp.name.clone());
            if let Some(v) = &imp.version {
                c.err(
                    imp.span,
                    format!(
                        "`import {name}@{v}`: versions come only from laplace.toml and laplace.lock; write \
                         `import {name}` and choose the version with `laplace add {name}@{v}` (or --tag)",
                        name = imp.name
                    ),
                );
                continue;
            }
            match resolve_import(project, &imp.name) {
                Ok(lib) => libraries.push(lib),
                Err(e) => c.err(imp.span, e),
            }
        }
        let scope = Scope {
            dir,
            imported,
            libraries: &libraries,
        };

        // Lines: rest at most once and last; labels unique; no repeats in a block.
        let mut labels: Vec<(String, Span)> = vec![];
        for (i, l) in ast.lines.iter().enumerate() {
            let label = match &l.target {
                LineTarget::Rest => {
                    if i + 1 != ast.lines.len() {
                        c.err(l.span, "`rest` must be the last kernel line: it takes whatever the lines above left");
                    }
                    "rest".to_string()
                }
                LineTarget::Block { params, label } => {
                    for (j, p) in params.iter().enumerate() {
                        if params[..j].contains(p) {
                            c.err(l.span, format!("`{p}` is listed twice in this block"));
                        }
                    }
                    label.clone().unwrap_or_else(|| params.join("_"))
                }
            };
            if let Some((_, sp)) = labels.iter().find(|(x, _)| *x == label) {
                let sp = *sp;
                c.err(
                    l.span,
                    format!(
                        "block label `{label}` is already used at {sp}; name one with `as <label>`"
                    ),
                );
            }
            labels.push((label, l.span));
            check_kexpr(&mut c, &l.kernel, &scope, &mut kernels, false);
        }

        if c.errors.is_empty() {
            Ok(Self {
                path: path.to_string(),
                source,
                ast,
                kernels,
                libraries,
            })
        } else {
            c.errors
                .sort_by_key(|e| (e.file.clone(), e.span.line, e.span.col));
            Err(c.errors)
        }
    }

    pub fn warmup(&self) -> Option<u64> {
        self.ast
            .warmup
            .iter()
            .find(|s| s.name == "iterations")
            .map(|s| s.value as u64)
    }
    pub fn draws(&self) -> Option<u64> {
        self.ast
            .draws
            .iter()
            .find(|s| s.name == "iterations")
            .map(|s| s.value as u64)
    }
    pub fn thin(&self) -> Option<u64> {
        self.ast
            .draws
            .iter()
            .find(|s| s.name == "thin")
            .map(|s| s.value as u64)
    }

    /// Blocks of each kernel line against the model's parameters: unknown
    /// names, overlaps, `rest`, and coverage (D2: every parameter in a
    /// block). Errors point at the line.
    pub fn blocks(&self, params: &[Param]) -> Result<Vec<Block>, Vec<LapError>> {
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

    /// The kernel tree for one chain: seq(on(block, kernel), ...).
    /// `full` is the whole-model target (nuts uses it for its block target).
    pub fn instantiate(
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

pub fn errors(es: Vec<LapError>) -> anyhow::Error {
    anyhow::anyhow!(
        "{}",
        es.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    )
}

/// Can `adapt(k)` drive `k`? `Ok(())` or the reason. Model-free.
fn adaptable(k: &KExpr, kernels: &[(String, Arc<LapKernelFile>)]) -> Result<(), String> {
    match k {
        KExpr::BuiltIn { name, .. } => match name.as_str() {
            "nuts" => {
                Err("nuts() adapts its own step size and metric; use it without adapt()".into())
            }
            _ => Ok(()),
        },
        KExpr::Path { .. } | KExpr::Library { .. } => {
            let key = kernel_key(k).unwrap();
            match kernels.iter().find(|(p, _)| *p == key) {
                Some((_, f)) if f.has_adapt(AdaptKind::StepSize) || f.has_adapt(AdaptKind::Metric) => Ok(()),
                Some(_) => Err(format!(
                    "{key} has nothing to adapt: tag its step size @adapt(step_size, target_accept = ...) \
                     and/or its inverse metric @adapt(metric)"
                )),
                None => Ok(()), // its own load error is reported
            }
        }
        other => Err(format!(
            "adapt() wraps a single kernel, not {}(...); put adapt() around each kernel inside it",
            match other {
                KExpr::Seq(..) => "seq",
                KExpr::Mix(..) => "mix",
                KExpr::Repeat(..) => "repeat",
                KExpr::Adapt(..) => "adapt",
                _ => "on",
            }
        )),
    }
}

/// Model-free checks of one kernel expression.
fn check_kexpr(
    c: &mut Ctx,
    k: &KExpr,
    scope: &Scope,
    kernels: &mut Vec<(String, Arc<LapKernelFile>)>,
    under_mix: bool,
) {
    match k {
        KExpr::BuiltIn { name, args, span } => match BuiltIn::parse(name) {
            Err(_) => c.err(
                *span,
                format!(
                    "unknown kernel `{name}`: the built-in kernels are nuts, rwm, mala and hmc; a local kernel is a \
                     path such as ./kernels/{name}.lapkernel; a library kernel is <package>::{name}"
                ),
            ),
            Ok(b) => {
                if b == BuiltIn::Nuts && under_mix {
                    c.err(
                        *span,
                        "nuts() cannot sit under mix(): it adapts for a fixed number of warmup steps, and \
                         mix() makes that number random",
                    );
                }
                let allowed = builtin_settings(b);
                for a in args {
                    if !allowed.contains(&a.name.as_str()) {
                        c.err(
                            a.span,
                            format!(
                                "{name}() has no setting `{}`; its settings are: {}{}",
                                a.name,
                                allowed.join(", "),
                                if a.name == "target_accept" || a.name == "adapt" {
                                    " (adaptation is explicit here: adapt(kernel, target_accept = ...))"
                                } else {
                                    ""
                                }
                            ),
                        );
                    }
                }
                let sets: Vec<String> = args
                    .iter()
                    .filter(|a| allowed.contains(&a.name.as_str()))
                    .map(|a| match a.value {
                        ArgValue::Number(v) => format!("{}={v}", a.name),
                        ArgValue::Bool(v) => format!("{}={v}", a.name),
                    })
                    .collect();
                if let Err(e) = SamplerSpec::new(b, &sets) {
                    c.err(*span, format!("{name}(): {e}"));
                }
            }
        },
        KExpr::Path { path, args, span } => {
            if !path.ends_with(".lapkernel") {
                c.err(*span, format!("`{path}`: a local kernel is a .lapkernel file"));
                return;
            }
            let full = scope.dir.join(path);
            load_kernel(c, path, LapKernelFile::load(&full), args, *span, kernels);
        }
        KExpr::Library {
            package,
            name,
            args,
            span,
        } => {
            if !scope.imported.contains(package) {
                c.err(
                    *span,
                    format!(
                        "`{package}::{name}`: `{package}` is not imported; add `import {package}` to this \
                         file's library {{ }} block"
                    ),
                );
                return;
            }
            // An import that did not resolve has its own error.
            let Some(lib) = scope.libraries.iter().find(|l| &l.name == package) else {
                return;
            };
            match lib.kernel_file(name) {
                Err(e) => c.err(*span, e),
                Ok(file) => {
                    let loaded = std::fs::read_to_string(&file)
                        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", file.display()))
                        .and_then(|src| LapKernelFile::from_source(&display_path(package, name), src));
                    load_kernel(c, &format!("{package}::{name}"), loaded, args, *span, kernels);
                }
            }
        }
        KExpr::Seq(ks, _) => ks.iter().for_each(|k| check_kexpr(c, k, scope, kernels, under_mix)),
        KExpr::Mix(ks, span) => {
            if ks.iter().any(|(w, _)| !(w.is_finite() && *w > 0.0)) {
                c.err(*span, "mix() weights must be positive numbers");
            }
            ks.iter().for_each(|(_, k)| check_kexpr(c, k, scope, kernels, true));
        }
        KExpr::Repeat(n, k, span) => {
            if !(*n >= 1.0 && n.fract() == 0.0) {
                c.err(*span, format!("repeat(n, ...) needs a whole number n >= 1, got {n}"));
            }
            check_kexpr(c, k, scope, kernels, under_mix);
        }
        KExpr::Adapt(k, args, span) => {
            check_kexpr(c, k, scope, kernels, under_mix);
            for a in args {
                match (a.name.as_str(), number(a)) {
                    ("target_accept", Some(v)) if v > 0.0 && v < 1.0 => {}
                    ("target_accept", _) => c.err(a.span, "target_accept must be a number strictly between 0 and 1"),
                    (other, _) => c.err(a.span, format!("adapt() takes only target_accept, not `{other}`")),
                }
            }
            if let Err(m) = adaptable(k, kernels) {
                c.err(*span, format!("adapt(): {m}"));
            }
        }
        KExpr::On(_, _, span) => c.err(
            *span,
            "on(block(...), ...) cannot be used inside a kernel line: the line already names its block",
        ),
    }
}

/// Record a `.lapkernel` leaf under `key` (once) and check the tuning
/// values given to it. A file that does not load or check is reported at
/// the leaf, with the kernel's own errors indented below it.
fn load_kernel(
    c: &mut Ctx,
    key: &str,
    loaded: anyhow::Result<LapKernelFile>,
    args: &[Arg],
    span: Span,
    kernels: &mut Vec<(String, Arc<LapKernelFile>)>,
) {
    let file = if let Some((_, f)) = kernels.iter().find(|(p, _)| p == key) {
        Some(f.clone())
    } else {
        match loaded {
            Ok(f) => {
                let f = Arc::new(f);
                kernels.push((key.to_string(), f.clone()));
                Some(f)
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.starts_with("cannot read") {
                    c.err(span, msg);
                } else {
                    // Its own errors keep their file:line:col.
                    let indented: Vec<String> = msg.lines().map(|l| format!("  {l}")).collect();
                    c.err(
                        span,
                        format!("{key} does not check:\n{}", indented.join("\n")),
                    );
                }
                None
            }
        }
    };
    if let Some(f) = file {
        let settable = f.settable();
        for a in args {
            if !settable.contains(&a.name) {
                c.err(
                    a.span,
                    format!(
                        "{key} has no tuning value `{}`; its tuning values are: {}",
                        a.name,
                        settable.join(", ")
                    ),
                );
            } else if number(a).is_none() {
                c.err(a.span, format!("`{}` must be a number", a.name));
            }
        }
    }
}

#[cfg(test)]
mod tests;
