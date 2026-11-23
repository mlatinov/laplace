//! `laplace-sample sample`: load a model, run chains in parallel, write one
//! CSV per chain.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bridgestan::{Model, StanLibrary};

use crate::csv::{adaptation_lines, sampler_values, Header, KernelLine, Writer};
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
    pub warmup: u64,
    pub draws: u64,
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

/// Run the sampler; returns the CSV paths, in chain order.
pub fn sample(args: &SampleArgs) -> anyhow::Result<Vec<PathBuf>> {
    anyhow::ensure!(args.chains >= 1, "--chains must be at least 1");
    anyhow::ensure!(args.draws >= 1, "--draws must be at least 1");
    let spec = Sampler::from_arg(&args.sampler, &args.set)?;
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
                let r = run_one_chain(args, &spec, &init, seed, i as u64 + 1, &lib, &prepared.stem);
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

#[allow(clippy::too_many_arguments)]
fn run_one_chain(
    args: &SampleArgs,
    spec: &Sampler,
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
    let model_name = model.name()?.to_string();
    let mut gq_rng = bridgestan::Rng::new(lib.clone(), gq_seed(seed, chain_id))?;
    let target = Arc::new(BridgeStanTarget { model });
    let dyn_target: Arc<dyn Target> = target.clone();
    let mut kernel = spec.build(dyn_target, args.warmup, args.draws)?;

    let header = Header {
        model: model_name,
        chain_id,
        seed,
        warmup: args.warmup,
        draws: args.draws,
        sampler: spec.name(),
        settings: spec.settings_text(),
        // Level 1: one kernel over the whole model, i.e. `rest : <kernel>`.
        kernels: vec![KernelLine {
            label: "rest".to_string(),
            name: kernel.name().to_string(),
            version: kernel.version(),
            source: kernel.source().into_owned(),
        }],
    };
    let path = args.output_dir.join(format!("{stem}-{chain_id}.csv"));
    let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let w = std::cell::RefCell::new(Writer::new(file, &header, &param_names)?);
    let mut out = vec![0.0; param_names.len()];
    let mut rng = chain_rng(seed, chain_id);
    let adapted = spec.adapts();
    // Wall-clock timing goes to stderr only: the CSV must stay
    // byte-identical across runs.
    let started = std::time::Instant::now();
    let warmup_done = std::cell::Cell::new(None);
    let summary = run_chain(
        target.as_ref(),
        kernel.as_mut(),
        &init,
        args.warmup,
        args.draws,
        &mut rng,
        |t| {
            warmup_done.set(Some(std::time::Instant::now()));
            if adapted && args.warmup > 0 {
                w.borrow_mut().comment_lines(&adaptation_lines(t))?;
            }
            Ok(())
        },
        |state, info, tuning| {
            target
                .model
                .param_constrain(&state.q, true, true, &mut out, Some(&mut gq_rng))?;
            w.borrow_mut()
                .row(&sampler_values(state.lp, info, tuning), &out)?;
            Ok(())
        },
    )?;
    w.into_inner().finish()?;
    let sampling = warmup_done.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
    let warmup_secs = started.elapsed().as_secs_f64() - sampling;
    eprintln!(
        "chain {chain_id}: {} warmup + {} draws, {} + {} gradient evaluations; \
         {warmup_secs:.2} s warmup, {sampling:.2} s sampling ({:.1} us per gradient, sampler and output included)",
        args.warmup,
        args.draws,
        summary.warmup_grads,
        summary.draw_grads,
        1e6 * sampling / summary.draw_grads.max(1) as f64
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
