//! `laplace-sample sample`: load a model, run chains in parallel, write one
//! CSV per chain.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bridgestan::{Model, StanLibrary};

use crate::csv::{
    adaptation_lines, block_columns, block_values, composite_values, sampler_values,
    single_block_values, Header, Writer,
};
use crate::model;
use crate::runner::{chain_rng, gq_seed, run_chain, Init, Sampler};
use crate::target::{BridgeStanTarget, Target};

/// Everything `laplace-sample sample` was asked to do.
#[derive(Clone, Debug)]
pub struct SampleArgs {
    pub model: PathBuf,
    pub data: Option<PathBuf>,
    pub sampler: String,
    pub chains: u64,
    pub parallel_chains: Option<u64>,
    /// `None`: the .lapsampler's value, else 1000.
    pub warmup: Option<u64>,
    /// `None`: the .lapsampler's value, else 1000.
    pub draws: Option<u64>,
    /// Keep every `thin`-th draw. `None`: the .lapsampler's, else 1.
    pub thin: Option<u64>,
    pub seed: Option<u64>,
    /// Empty: radius 2. One value: a radius (number) or a JSON file for every
    /// chain. `chains` values: one JSON file per chain.
    pub init: Vec<String>,
    pub set: Vec<String>,
    pub output_dir: PathBuf,
}

/// Where each chain's inits come from.
#[derive(Clone, Debug)]
enum InitSpec {
    Radius(f64),
    Files(Vec<PathBuf>),
}

fn parse_init(init: &[String], chains: u64) -> anyhow::Result<InitSpec> {
    match init {
        [] => Ok(InitSpec::Radius(2.0)),
        [one] => match one.parse::<f64>() {
            Ok(r) => {
                anyhow::ensure!(
                    r.is_finite() && r >= 0.0,
                    "--init radius must be >= 0, got {one}"
                );
                Ok(InitSpec::Radius(r))
            }
            Err(_) => Ok(InitSpec::Files(vec![PathBuf::from(one); chains as usize])),
        },
        many => {
            anyhow::ensure!(
                many.len() as u64 == chains,
                "--init was given {} times; give it once (for every chain) or once per chain ({chains})",
                many.len()
            );
            Ok(InitSpec::Files(many.iter().map(PathBuf::from).collect()))
        }
    }
}

fn names(s: &str) -> Vec<String> {
    if s.is_empty() {
        vec![]
    } else {
        s.split(',').map(str::to_owned).collect()
    }
}

/// Run lengths after the CLI overrides the `.lapsampler` defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Run {
    pub warmup: u64,
    pub draws: u64,
    pub thin: u64,
}

pub const DEFAULT_WARMUP: u64 = 1000;
pub const DEFAULT_DRAWS: u64 = 1000;

/// CLI value, else the sampler file's, else the default.
pub fn resolve_run(args: &SampleArgs, spec: &Sampler) -> anyhow::Result<Run> {
    let (fw, fd, ft) = spec.file_run();
    let run = Run {
        warmup: args.warmup.or(fw).unwrap_or(DEFAULT_WARMUP),
        draws: args.draws.or(fd).unwrap_or(DEFAULT_DRAWS),
        thin: args.thin.or(ft).unwrap_or(1),
    };
    anyhow::ensure!(run.draws >= 1, "draws must be at least 1");
    anyhow::ensure!(run.thin >= 1, "thin must be at least 1");
    Ok(run)
}

/// Run the sampler; returns the CSV paths, in chain order.
pub fn sample(args: &SampleArgs) -> anyhow::Result<Vec<PathBuf>> {
    anyhow::ensure!(args.chains >= 1, "--chains must be at least 1");
    let spec = Sampler::from_arg(&args.sampler, &args.set)?;
    let run = resolve_run(args, &spec)?;
    let init = parse_init(&args.init, args.chains)?;
    let seed = args.seed.unwrap_or_else(|| rand::random::<u32>() as u64);
    if let Some(d) = &args.data {
        anyhow::ensure!(d.is_file(), "--data {} does not exist", d.display());
    }

    let stan = model::stan_file_for(&args.model)?;
    let prepared = model::prepare(&stan)?;
    if prepared.cache_hit {
        eprintln!("Using cached model build {}", prepared.library.display());
    }
    let lib = Arc::new(bridgestan::open_library(&prepared.library)?);
    std::fs::create_dir_all(&args.output_dir)?;

    let parallel = args
        .parallel_chains
        .unwrap_or(args.chains)
        .clamp(1, args.chains) as usize;
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<anyhow::Result<PathBuf>>>> =
        Mutex::new((0..args.chains).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..parallel {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i as u64 >= args.chains {
                    break;
                }
                let r = run_one_chain(
                    args,
                    &spec,
                    run,
                    &init,
                    seed,
                    i as u64 + 1,
                    &lib,
                    &prepared.stem,
                );
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    let mut paths = vec![];
    for (i, r) in results.into_inner().unwrap().into_iter().enumerate() {
        match r.expect("every chain ran") {
            Ok(p) => paths.push(p),
            Err(e) => return Err(e.context(format!("chain {}", i + 1))),
        }
    }
    Ok(paths)
}

fn data_arg(data: &Option<PathBuf>) -> anyhow::Result<Option<CString>> {
    data.as_ref()
        .map(|p| Ok(CString::new(p.to_string_lossy().as_bytes())?))
        .transpose()
}

/// The model's parameters, from BridgeStan's unconstrained names.
pub fn model_params(model: &Model<Arc<StanLibrary>>) -> anyhow::Result<Vec<crate::blocks::Param>> {
    crate::blocks::parameters(&names(model.param_unc_names()))
}

#[allow(clippy::too_many_arguments)]
fn run_one_chain(
    args: &SampleArgs,
    spec: &Sampler,
    run: Run,
    init: &InitSpec,
    seed: u64,
    chain_id: u64,
    lib: &Arc<StanLibrary>,
    stem: &str,
) -> anyhow::Result<PathBuf> {
    // Transformed-data RNG: the run's seed, the same for every chain (as in
    // CmdStan), so every chain sees the same data.
    let model = Model::new(lib.clone(), data_arg(&args.data)?, seed as u32)?;
    let init = match init {
        InitSpec::Radius(r) => Init::Radius(*r),
        InitSpec::Files(files) => {
            let path = &files[chain_id as usize - 1];
            let json = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("--init {}: {e}", path.display()))?;
            let mut q = vec![0.0; model.param_unc_num()];
            model
                .param_unconstrain_json(CString::new(json)?, &mut q)
                .map_err(|e| anyhow::anyhow!("--init {}: {e}", path.display()))?;
            Init::Point(q)
        }
    };
    let param_names = names(model.param_names(true, true));
    let params = model_params(&model)?;
    let model_name = model.name()?.to_string();
    let mut gq_rng = bridgestan::Rng::new(lib.clone(), gq_seed(seed, chain_id))?;
    let target = Arc::new(BridgeStanTarget { model });
    let dyn_target: Arc<dyn Target> = target.clone();
    let built = spec.build_chain(dyn_target, &params, run.warmup, run.draws)?;
    let mut kernel = built.kernel;
    // Columns (Phase 3.6): a kernel on the whole model, or a single block
    // covering it, writes the standard columns; several blocks add
    // per-block columns and leave the standard ones NaN where one value
    // cannot describe several kernels.
    let n_coords = target.dim();
    let layout = match built.blocks.len() {
        0 => Layout::Whole,
        1 if built.block_dims.first() == Some(&n_coords) => Layout::SingleBlock,
        _ => Layout::Blocks(built.blocks.clone()),
    };

    let header = Header {
        model: model_name,
        chain_id,
        seed,
        warmup: run.warmup,
        draws: run.draws,
        thin: run.thin,
        sampler: spec.name(),
        settings: spec.settings_text(),
        kernels: built.lines,
    };
    let path = args.output_dir.join(format!("{stem}-{chain_id}.csv"));
    let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let block_cols = match &layout {
        Layout::Blocks(labels) => block_columns(labels),
        _ => vec![],
    };
    let w = std::cell::RefCell::new(Writer::with_block_columns(
        file,
        &header,
        &block_cols,
        &param_names,
    )?);
    let mut out = vec![0.0; param_names.len()];
    let mut rng = chain_rng(seed, chain_id);
    let adapted = spec.adapts();
    // Wall-clock timing goes to stderr only: the CSV must stay
    // byte-identical across runs.
    let started = std::time::Instant::now();
    let warmup_done = std::cell::Cell::new(None);
    let mut i = 0u64;
    let mut since = (0, 0);
    let summary = run_chain(
        target.as_ref(),
        kernel.as_mut(),
        &init,
        run.warmup,
        run.draws,
        &mut rng,
        |k| {
            warmup_done.set(Some(std::time::Instant::now()));
            // CmdStan's adaptation block: the whole-model kernel's tuning,
            // or the single block's.
            let t = match &layout {
                Layout::Whole if adapted => Some(k.tuning()),
                Layout::SingleBlock => k
                    .tunings()
                    .into_iter()
                    .next()
                    .map(|(_, t)| t)
                    .filter(|t| t.step_size.is_some() || t.inv_metric.is_some()),
                _ => None,
            };
            if let (Some(t), true) = (t, run.warmup > 0) {
                w.borrow_mut().comment_lines(&adaptation_lines(&t))?;
            }
            Ok(())
        },
        |state, info, tuning| {
            let keep = i.is_multiple_of(run.thin);
            i += 1;
            if let Some(info) = saved_row_cost(&mut since, info, keep) {
                let info = &info;
                target
                    .model
                    .param_constrain(&state.q, true, true, &mut out, Some(&mut gq_rng))?;
                let cols = match &layout {
                    Layout::Whole => sampler_values(state.lp, info, tuning),
                    Layout::SingleBlock => single_block_values(state.lp, info),
                    Layout::Blocks(labels) => {
                        let mut v = composite_values(state.lp, info);
                        v.extend(block_values(info, labels));
                        v
                    }
                };
                w.borrow_mut().row(&cols, &out)?;
            }
            Ok(())
        },
    )?;
    w.into_inner().finish()?;
    let sampling = warmup_done.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
    let warmup_secs = started.elapsed().as_secs_f64() - sampling;
    // Density-only evaluations (rwm, propose()/move()) are listed apart:
    // they skip the reverse sweep, so they cost less than a gradient.
    let logps = if summary.warmup_logps + summary.draw_logps > 0 {
        format!(
            " and {} + {} density-only",
            summary.warmup_logps, summary.draw_logps
        )
    } else {
        String::new()
    };
    eprintln!(
        "chain {chain_id}: {} warmup + {} draws, {} + {} gradient evaluations{logps}; \
         {warmup_secs:.2} s warmup, {sampling:.2} s sampling",
        run.warmup, run.draws, summary.warmup_grads, summary.draw_grads,
    );
    Ok(path)
}

/// For messages: a path relative to the current directory when possible.
pub fn display(p: &Path) -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| p.strip_prefix(cwd).ok().map(Path::to_path_buf))
        .unwrap_or_else(|| p.to_path_buf())
        .display()
        .to_string()
}

/// `laplace sampler check <file> [--model m --data d]`: a summary on
/// success; every error (each `file:line:col`) on failure.
pub fn check_sampler(
    file: &Path,
    model_path: Option<&Path>,
    data: Option<&Path>,
) -> anyhow::Result<String> {
    let f = crate::lapsampler::SamplerFile::load(file).map_err(crate::lapsampler::errors)?;
    let mut out = format!(
        "ok: {} ({} kernel line(s); kernels resolve, settings and adapt() contracts check)",
        f.path,
        f.ast.lines.len()
    );
    match model_path {
        None => {
            out +=
                "\nblocks not checked against a model: pass --model (and --data) to check coverage"
        }
        Some(m) => {
            let stan = model::stan_file_for(m)?;
            let prepared = model::prepare(&stan)?;
            let lib = Arc::new(bridgestan::open_library(&prepared.library)?);
            let data = data.map(Path::to_path_buf);
            let model = Model::new(lib, data_arg(&data)?, 1)?;
            let params = model_params(&model)?;
            let blocks = f.blocks(&params).map_err(crate::lapsampler::errors)?;
            for b in blocks {
                out += &format!(
                    "\n  block {}: {} ({} unconstrained coordinate(s))",
                    b.label,
                    b.params.join(", "),
                    b.dim()
                );
            }
            out += &format!(
                "\nevery parameter of {} is in exactly one block",
                stan.display()
            );
        }
    }
    Ok(out)
}

/// Which sampler columns a run writes (Phase 3.6).
enum Layout {
    /// A kernel on the whole model (Level 1): the standard columns.
    Whole,
    /// One block covering the whole model: its statistics, unsuffixed.
    SingleBlock,
    /// Several blocks: per-block columns for these labels.
    Blocks(Vec<String>),
}

/// The statistics a saved row reports, or `None` for an iteration thinning
/// drops. The cost columns count every iteration since the previous saved
/// row, so summing `n_grad__` / `n_logp__` over the file gives the sampling
/// cost whatever `thin` is (up to the last saved draw); every other column
/// describes the saved iteration itself, as in CmdStan.
fn saved_row_cost(
    since: &mut (u64, u64),
    info: &crate::kernel::StepInfo,
    keep: bool,
) -> Option<crate::kernel::StepInfo> {
    since.0 += info.n_grad;
    since.1 += info.n_logp;
    if !keep {
        return None;
    }
    let mut row = info.clone();
    (row.n_grad, row.n_logp) = std::mem::take(since);
    Some(row)
}

#[cfg(test)]
mod cost_tests {
    use super::*;
    use crate::kernel::StepInfo;

    #[test]
    fn thinned_rows_carry_the_cost_of_the_iterations_dropped_before_them() {
        let mut since = (0, 0);
        let mut rows = vec![];
        for i in 0..10u64 {
            let info = StepInfo {
                n_grad: i,
                n_logp: 1,
                accept_prob: Some(i as f64),
                ..Default::default()
            };
            if let Some(r) = saved_row_cost(&mut since, &info, i % 4 == 0) {
                rows.push((r.n_grad, r.n_logp, r.accept_prob));
            }
        }
        // Saved: 0, 4, 8. Costs: {0}, {1..4}, {5..8}; 9 comes after the last.
        assert_eq!(
            rows,
            vec![
                (0, 1, Some(0.0)),
                (1 + 2 + 3 + 4, 4, Some(4.0)),
                (5 + 6 + 7 + 8, 4, Some(8.0))
            ]
        );
        assert_eq!(since, (9, 1), "the tail after the last saved draw");
    }
}
