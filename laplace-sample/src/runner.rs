//! Chains: random streams, initial values, the warmup/draw loop, and the
//! built-in sampler table (`--sampler` and `--set`).

use std::collections::BTreeMap;
use std::sync::Arc;

use rand::{RngExt, SeedableRng};

use crate::adapt::{AdaptConfig, Adapter};
use crate::kernel::{ChainRng, Kernel, StepInfo, Tuning};
use crate::kernels::{hmc::Hmc, mala::Mala, nuts::Nuts, rwm::Rwm};
use crate::lapkernel::check::AdaptKind;
use crate::lapkernel::kernel::{InterpretedKernel, LapKernelFile};
use crate::state::{ChainState, Evaluator};
use crate::target::Target;

/// The random stream of chain `chain_id`: ChaCha8 seeded with `seed`, on
/// stream `chain_id`. Chains of one run share the seed and differ by stream,
/// so they are independent and each is reproducible on its own.
pub fn chain_rng(seed: u64, chain_id: u64) -> ChainRng {
    let mut r = ChainRng::seed_from_u64(seed);
    r.set_stream(chain_id);
    r
}

/// Seed for BridgeStan's generated-quantities RNG of one chain. Derived
/// separately from the kernel stream (SplitMix64 of seed and chain id), so
/// generated quantities never shift the kernel's random numbers.
pub fn gq_seed(seed: u64, chain_id: u64) -> u32 {
    let mut z = seed ^ chain_id.rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

/// Where a chain starts.
#[derive(Clone, Debug, PartialEq)]
pub enum Init {
    /// Uniform(-r, r) on every unconstrained coordinate (Stan's default
    /// r = 2). `r = 0` means all zeros and draws nothing.
    Radius(f64),
    /// An explicit unconstrained point (from a user JSON via
    /// `param_unconstrain`).
    Point(Vec<f64>),
}

/// Stan's limit on attempts to find a random initial point with a finite
/// density and gradient.
pub const MAX_INIT_TRIES: usize = 100;

/// RNG: `Init::Radius(r > 0)` draws `dim` uniforms (`rng.random_range(-r..r)`)
/// per attempt, in index order, before the kernel draws anything.
pub fn initial_state(
    init: &Init,
    ev: &mut Evaluator,
    rng: &mut ChainRng,
) -> anyhow::Result<ChainState> {
    let d = ev.dim();
    let ok = |s: &ChainState| s.lp.is_finite() && s.grad.iter().all(|g| g.is_finite());
    match init {
        Init::Point(q) => {
            anyhow::ensure!(q.len() == d, "init has {} values, model has {d}", q.len());
            let s = ChainState::at(q.clone(), ev)?;
            anyhow::ensure!(
                ok(&s),
                "the log density or its gradient is not finite at the given initial values"
            );
            Ok(s)
        }
        Init::Radius(r) if *r == 0.0 => {
            let s = ChainState::at(vec![0.0; d], ev)?;
            anyhow::ensure!(
                ok(&s),
                "the log density or its gradient is not finite at zero"
            );
            Ok(s)
        }
        Init::Radius(r) => {
            anyhow::ensure!(
                r.is_finite() && *r > 0.0,
                "init radius must be >= 0, got {r}"
            );
            for _ in 0..MAX_INIT_TRIES {
                let q: Vec<f64> = (0..d).map(|_| rng.random_range(-r..*r)).collect();
                let s = ChainState::at(q, ev)?;
                if ok(&s) {
                    return Ok(s);
                }
            }
            anyhow::bail!(
                "no finite log density after {MAX_INIT_TRIES} random initial points in (-{r}, {r}); \
                 try --init with a JSON file or a smaller radius"
            )
        }
    }
}

/// What a finished chain reports beyond its draws.
#[derive(Clone, Debug)]
pub struct ChainSummary {
    /// Tuning in force for every saved draw (frozen after warmup).
    pub final_tuning: Tuning,
    pub warmup_grads: u64,
    pub draw_grads: u64,
}

/// Run one chain: `warmup` adapted iterations, then `draws` saved ones.
/// `on_draw` sees every saved iteration with the state after the step, the
/// step's statistics, and the tuning that step used.
#[allow(clippy::too_many_arguments)]
pub fn run_chain(
    target: &dyn Target,
    kernel: &mut dyn Kernel,
    mut adapter: Option<Adapter>,
    init: &Init,
    warmup: u64,
    draws: u64,
    rng: &mut ChainRng,
    mut on_end_of_warmup: impl FnMut(&Tuning) -> anyhow::Result<()>,
    mut on_draw: impl FnMut(&ChainState, &StepInfo, &Tuning) -> anyhow::Result<()>,
) -> anyhow::Result<ChainSummary> {
    anyhow::ensure!(
        adapter.is_none() || !kernel.adapts_itself(),
        "{} adapts itself and cannot also be wrapped in adaptation",
        kernel.name()
    );
    let mut ev = Evaluator::new(target);
    let mut state = initial_state(init, &mut ev, rng)?;
    kernel.init(&state, &mut ev, rng)?;
    let (mut warmup_grads, mut draw_grads) = (0, 0);
    for it in 0..warmup + draws {
        if it == warmup {
            on_end_of_warmup(&kernel.tuning())?;
        }
        let tuning = kernel.tuning();
        let info = kernel.step(&mut state, &mut ev, rng)?;
        if it < warmup {
            warmup_grads += info.n_grad;
            if let Some(a) = adapter.as_mut() {
                a.after_step(it, &state.q, &info, kernel)?;
            }
        } else {
            draw_grads += info.n_grad;
            on_draw(&state, &info, &tuning)?;
        }
    }
    if warmup + draws == warmup {
        on_end_of_warmup(&kernel.tuning())?;
    }
    Ok(ChainSummary {
        final_tuning: kernel.tuning(),
        warmup_grads,
        draw_grads,
    })
}

/// The built-in samplers of `--sampler`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltIn {
    Nuts,
    Rwm,
    Mala,
    Hmc,
}

impl BuiltIn {
    pub const ALL: [BuiltIn; 4] = [BuiltIn::Nuts, BuiltIn::Rwm, BuiltIn::Mala, BuiltIn::Hmc];

    pub fn name(self) -> &'static str {
        match self {
            BuiltIn::Nuts => "nuts",
            BuiltIn::Rwm => "rwm",
            BuiltIn::Mala => "mala",
            BuiltIn::Hmc => "hmc",
        }
    }

    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Self::ALL
            .into_iter()
            .find(|b| b.name() == s)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown sampler `{s}`; built-in samplers are: {}",
                    Self::ALL.map(|b| b.name()).join(", ")
                )
            })
    }

    /// `--set` keys and their defaults, in the order they are documented.
    pub fn defaults(self) -> Vec<(&'static str, &'static str)> {
        match self {
            BuiltIn::Nuts => vec![
                ("step_size", "1"),
                ("target_accept", "0.8"),
                ("max_depth", "10"),
            ],
            BuiltIn::Rwm => vec![
                ("step_size", "1"),
                ("target_accept", "0.234"),
                ("adapt", "true"),
            ],
            BuiltIn::Mala => vec![
                ("step_size", "1"),
                ("target_accept", "0.574"),
                ("adapt", "true"),
            ],
            BuiltIn::Hmc => vec![
                ("step_size", "1"),
                ("target_accept", "0.8"),
                ("steps", "10"),
                ("jitter", "true"),
                ("adapt", "true"),
            ],
        }
    }
}

/// A built-in sampler with its `--set` values resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerSpec {
    pub kind: BuiltIn,
    /// Every setting, defaults filled in, sorted by key.
    pub settings: BTreeMap<String, String>,
}

impl SamplerSpec {
    /// `sets` are the raw `key=value` strings of `--set`.
    pub fn new(kind: BuiltIn, sets: &[String]) -> anyhow::Result<Self> {
        let mut settings: BTreeMap<String, String> = kind
            .defaults()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        for s in sets {
            let (k, v) = s
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--set expects key=value, got `{s}`"))?;
            let (k, v) = (k.trim(), v.trim());
            anyhow::ensure!(
                settings.contains_key(k),
                "`{}` has no setting `{k}`; its settings are: {}",
                kind.name(),
                kind.defaults()
                    .iter()
                    .map(|(k, _)| *k)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            settings.insert(k.to_string(), v.to_string());
        }
        let spec = Self { kind, settings };
        // Validate every value now, before any chain starts.
        spec.f64("step_size")?;
        let ta = spec.f64("target_accept")?;
        anyhow::ensure!(
            ta > 0.0 && ta < 1.0,
            "target_accept must be in (0, 1), got {ta}"
        );
        for key in ["adapt", "jitter"] {
            if spec.settings.contains_key(key) {
                spec.bool(key)?;
            }
        }
        if spec.settings.contains_key("steps") {
            anyhow::ensure!(spec.u64("steps")? >= 1, "steps must be at least 1");
        }
        if spec.settings.contains_key("max_depth") {
            anyhow::ensure!(spec.u64("max_depth")? >= 1, "max_depth must be at least 1");
        }
        Ok(spec)
    }

    fn get(&self, k: &str) -> &str {
        &self.settings[k]
    }
    pub fn f64(&self, k: &str) -> anyhow::Result<f64> {
        let v = self.get(k);
        let x: f64 = v
            .parse()
            .map_err(|_| anyhow::anyhow!("{k}: `{v}` is not a number"))?;
        anyhow::ensure!(x.is_finite(), "{k}: `{v}` is not finite");
        if k == "step_size" {
            anyhow::ensure!(x > 0.0, "step_size must be positive, got {x}");
        }
        Ok(x)
    }
    pub fn u64(&self, k: &str) -> anyhow::Result<u64> {
        let v = self.get(k);
        v.parse()
            .map_err(|_| anyhow::anyhow!("{k}: `{v}` is not a whole number"))
    }
    pub fn bool(&self, k: &str) -> anyhow::Result<bool> {
        match self.get(k) {
            "true" => Ok(true),
            "false" => Ok(false),
            v => anyhow::bail!("{k}: expected true or false, got `{v}`"),
        }
    }

    /// `key:value` pairs, sorted, space separated (the CSV settings line). Not
    /// `key=value`: cmdstanr drops a `# key = value` header line whose value
    /// contains another `=`, so `$metadata()` would lose it.
    pub fn settings_text(&self) -> String {
        self.settings
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Build the kernel and its warmup adapter for one chain.
    pub fn build(
        &self,
        target: Arc<dyn Target>,
        warmup: u64,
        draws: u64,
    ) -> anyhow::Result<(Box<dyn Kernel>, Option<Adapter>)> {
        let eps = self.f64("step_size")?;
        let delta = self.f64("target_accept")?;
        let dim = target.dim();
        let kernel: Box<dyn Kernel> = match self.kind {
            BuiltIn::Nuts => {
                let mut k = Nuts::new(target, warmup, draws);
                k.set_initial_step_size(eps);
                k.set_target_accept(delta);
                k.set_max_depth(self.u64("max_depth")?);
                return Ok((Box::new(k), None));
            }
            BuiltIn::Rwm => Box::new(Rwm::new(eps)),
            BuiltIn::Mala => Box::new(Mala::new(eps)),
            BuiltIn::Hmc => {
                let mut k = Hmc::new(eps, self.u64("steps")?);
                k.jitter = self.bool("jitter")?;
                Box::new(k)
            }
        };
        let adapter = if self.bool("adapt")? {
            // Stan's windowed adaptation: dual averaging toward the target
            // acceptance, diagonal metric from the slow windows.
            let config = AdaptConfig {
                target_accept: delta,
                step_size: true,
                metric: true,
            };
            Some(Adapter::new(config, warmup, dim, eps))
        } else {
            None
        };
        Ok((kernel, adapter))
    }
}

/// What `--sampler` names: a built-in, or a `.lapkernel` file (Level 3,
/// run at Level 1 with the same automatic warmup adaptation).
#[derive(Clone, Debug)]
pub enum Sampler {
    BuiltIn(SamplerSpec),
    File(FileSampler),
}

/// A `.lapkernel` file plus its `--set` values.
#[derive(Clone, Debug)]
pub struct FileSampler {
    pub file: Arc<LapKernelFile>,
    /// Scalar tuning values from `--set name=value`, in order.
    pub sets: Vec<(String, f64)>,
    /// `--set adapt=false` turns warmup adaptation off.
    pub adapt: bool,
    /// `--set target_accept=...`; else the kernel's own (its `@adapt` tag
    /// must carry one, Q14).
    pub target_accept: Option<f64>,
}

impl Sampler {
    /// `--sampler` value: a path ending in `.lapkernel`, or a built-in name.
    pub fn from_arg(arg: &str, sets: &[String]) -> anyhow::Result<Self> {
        if !arg.ends_with(".lapkernel") {
            return Ok(Sampler::BuiltIn(SamplerSpec::new(
                BuiltIn::parse(arg)?,
                sets,
            )?));
        }
        let file = Arc::new(LapKernelFile::load(std::path::Path::new(arg))?);
        let mut fs = FileSampler {
            file: file.clone(),
            sets: vec![],
            adapt: true,
            target_accept: None,
        };
        let has_step = file.has_adapt(AdaptKind::StepSize);
        for s in sets {
            let (k, v) = s
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--set expects key=value, got `{s}`"))?;
            let (k, v) = (k.trim(), v.trim());
            match k {
                "adapt" => {
                    fs.adapt = match v {
                        "true" => true,
                        "false" => false,
                        _ => anyhow::bail!("adapt: expected true or false, got `{v}`"),
                    }
                }
                "target_accept" if has_step => {
                    let x: f64 = v
                        .parse()
                        .map_err(|_| anyhow::anyhow!("target_accept: `{v}` is not a number"))?;
                    anyhow::ensure!(
                        x > 0.0 && x < 1.0,
                        "target_accept must be in (0, 1), got {x}"
                    );
                    fs.target_accept = Some(x);
                }
                name if file.settable().iter().any(|n| n == name) => {
                    let x: f64 = v
                        .parse()
                        .map_err(|_| anyhow::anyhow!("{name}: `{v}` is not a number"))?;
                    fs.sets.push((name.to_string(), x));
                }
                other => {
                    let mut keys = file.settable();
                    keys.push("adapt".into());
                    if has_step {
                        keys.push("target_accept".into());
                    }
                    anyhow::bail!(
                        "`{}` has no setting `{other}`; its settings are: {}",
                        file.path,
                        keys.join(", ")
                    )
                }
            }
        }
        Ok(Sampler::File(fs))
    }

    /// The name in the CSV `laplace_sampler` line.
    pub fn name(&self) -> String {
        match self {
            Sampler::BuiltIn(s) => s.kind.name().to_string(),
            Sampler::File(f) => f.file.path.clone(),
        }
    }

    pub fn settings_text(&self) -> String {
        match self {
            Sampler::BuiltIn(s) => s.settings_text(),
            Sampler::File(f) => {
                let mut items: Vec<String> = vec![format!("adapt:{}", f.adapt)];
                if let Some(t) = f.target_accept {
                    items.push(format!("target_accept:{t}"));
                }
                items.extend(f.sets.iter().map(|(k, v)| format!("{k}:{v}")));
                items.sort();
                items.join(" ")
            }
        }
    }

    pub fn build(
        &self,
        target: Arc<dyn Target>,
        warmup: u64,
        draws: u64,
    ) -> anyhow::Result<(Box<dyn Kernel>, Option<Adapter>)> {
        let f = match self {
            Sampler::BuiltIn(s) => return s.build(target, warmup, draws),
            Sampler::File(f) => f,
        };
        let dim = target.dim();
        let kernel = InterpretedKernel::new(f.file.clone(), dim, &f.sets)?;
        let step_size = kernel.adapts(AdaptKind::StepSize);
        let metric = kernel.adapts(AdaptKind::Metric);
        let adapter = if f.adapt && (step_size || metric) {
            let config = AdaptConfig {
                // A step_size tag always carries one (checked); a
                // metric-only kernel adapts no step size and ignores it.
                target_accept: f.target_accept.or(kernel.target_accept()).unwrap_or(0.5),
                step_size,
                metric,
            };
            let initial = kernel.tuning().step_size.unwrap_or(1.0);
            Some(Adapter::new(config, warmup, dim, initial))
        } else {
            None
        };
        Ok((Box::new(kernel), adapter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::analytic::DiagGaussian;

    fn run(
        kind: BuiltIn,
        sets: &[&str],
        target: Arc<dyn Target>,
        warmup: u64,
        draws: u64,
        seed: u64,
    ) -> (Vec<(ChainState, StepInfo, Tuning)>, ChainSummary) {
        let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        let spec = SamplerSpec::new(kind, &sets).unwrap();
        let (mut k, a) = spec.build(target.clone(), warmup, draws).unwrap();
        let mut rng = chain_rng(seed, 1);
        let mut out = vec![];
        let s = run_chain(
            target.as_ref(),
            k.as_mut(),
            a,
            &Init::Radius(2.0),
            warmup,
            draws,
            &mut rng,
            |_| Ok(()),
            |st, info, t| {
                out.push((st.clone(), info.clone(), t.clone()));
                Ok(())
            },
        )
        .unwrap();
        (out, s)
    }

    /// Accepted plan change (Phase 0 review): never re-tune after warmup.
    /// The tuning in force is identical at every post-warmup iteration, for
    /// every built-in sampler, and equals the chain's final tuning.
    #[test]
    fn tuning_is_frozen_after_warmup_for_every_sampler() {
        for kind in BuiltIn::ALL {
            let target: Arc<dyn Target> = Arc::new(DiagGaussian {
                mu: vec![1.0, -2.0, 0.5],
                sd: vec![1.0, 3.0, 0.2],
            });
            let (draws, summary) = run(kind, &[], target, 300, 200, 7);
            assert_eq!(draws.len(), 200);
            let first = &draws[0].2;
            assert!(
                first.step_size.is_some(),
                "{}: no step size reported",
                kind.name()
            );
            for (i, (_, _, t)) in draws.iter().enumerate() {
                assert_eq!(
                    t,
                    first,
                    "{}: tuning changed at post-warmup draw {i}",
                    kind.name()
                );
            }
            assert_eq!(&summary.final_tuning, first, "{}", kind.name());
            {
                assert!(
                    first.inv_metric.is_some(),
                    "{}: metric not adapted",
                    kind.name()
                );
            }
        }
    }

    #[test]
    fn same_seed_same_chain_different_stream_different_chain() {
        let target: Arc<dyn Target> = Arc::new(DiagGaussian::std_normal(4));
        for kind in BuiltIn::ALL {
            let (a, _) = run(kind, &[], target.clone(), 100, 50, 3);
            let (b, _) = run(kind, &[], target.clone(), 100, 50, 3);
            let (c, _) = run(kind, &[], target.clone(), 100, 50, 4);
            assert_eq!(a, b, "{}", kind.name());
            assert_ne!(a, c, "{}", kind.name());
        }
        let mut r1 = chain_rng(1, 1);
        let mut r2 = chain_rng(1, 2);
        let x: u64 = r1.random();
        let y: u64 = r2.random();
        assert_ne!(x, y);
    }

    #[test]
    fn set_rejects_unknown_keys_and_bad_values() {
        let e = SamplerSpec::new(BuiltIn::Mala, &["eps=0.1".into()]).unwrap_err();
        assert!(e.to_string().contains("has no setting `eps`"), "{e}");
        assert!(SamplerSpec::new(BuiltIn::Mala, &["step_size=-1".into()]).is_err());
        assert!(SamplerSpec::new(BuiltIn::Mala, &["target_accept=1.5".into()]).is_err());
        assert!(SamplerSpec::new(BuiltIn::Hmc, &["steps=0".into()]).is_err());
        assert!(SamplerSpec::new(BuiltIn::Rwm, &["adapt=maybe".into()]).is_err());
        assert!(SamplerSpec::new(BuiltIn::Rwm, &["noequals".into()]).is_err());
        let s = SamplerSpec::new(BuiltIn::Hmc, &["steps=20".into()]).unwrap();
        assert_eq!(
            s.settings_text(),
            "adapt:true jitter:true step_size:1 steps:20 target_accept:0.8"
        );
    }

    #[test]
    fn adapt_false_keeps_the_given_step_size() {
        let target: Arc<dyn Target> = Arc::new(DiagGaussian::std_normal(2));
        let (draws, _) = run(
            BuiltIn::Mala,
            &["adapt=false", "step_size=0.3"],
            target,
            50,
            20,
            1,
        );
        assert!(draws
            .iter()
            .all(|d| d.2.step_size == Some(0.3) && d.2.inv_metric.is_none()));
    }

    #[test]
    fn init_radius_zero_starts_at_zero_and_draws_nothing() {
        let t = DiagGaussian::std_normal(3);
        let mut ev = Evaluator::new(&t);
        let mut a = chain_rng(1, 1);
        let s = initial_state(&Init::Radius(0.0), &mut ev, &mut a).unwrap();
        assert_eq!(s.q, vec![0.0; 3]);
        let mut b = chain_rng(1, 1);
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }

    #[test]
    fn init_gives_up_on_a_target_with_no_finite_density() {
        struct Nowhere;
        impl Target for Nowhere {
            fn dim(&self) -> usize {
                1
            }
            fn log_density_gradient(&self, _: &[f64], _: &mut [f64]) -> anyhow::Result<f64> {
                Ok(f64::NEG_INFINITY)
            }
        }
        let mut ev = Evaluator::new(&Nowhere);
        let e = initial_state(&Init::Radius(2.0), &mut ev, &mut chain_rng(1, 1)).unwrap_err();
        assert!(
            e.to_string().contains("no finite log density after 100"),
            "{e}"
        );
    }
}
