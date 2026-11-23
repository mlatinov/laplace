//! `laplace sampler test` (Phase 5.1): the reference suite.
//!
//! Six targets ship inside the binary, in two tiers:
//!
//! | Target | Dim | Checked against | Tier |
//! | --- | --- | --- | --- |
//! | std_normal_10 | 10 | exact moments | must pass |
//! | corr_gauss_10 (condition number 100) | 10 | exact moments | must pass |
//! | eight_schools_nc | 10 | long CmdStan reference, shipped | must pass |
//! | funnel_10 (Neal) | 10 | exact variance of the log scale | report only |
//! | banana_2 | 2 | exact moments | report only |
//! | mixture_2 (bimodal) | 2 | mass on each side | report only |
//!
//! Every row is judged by the health-gated rule of the statistical tests
//! (|z| < 4 when R-hat < 1.01 and bulk ESS > 400 for every checked
//! quantity; otherwise did-not-converge, never a looser threshold), and
//! reports ESS per gradient, ESS per second, gradient and density-only
//! evaluations counted apart, and the ratio of both efficiencies to plain
//! NUTS on the same target (Q15: reported, never a threshold).
//!
//! Draw counts come from the kernel's `tests/<kernel>.test.toml`
//! ([`TestConfig`]); the built-ins' are embedded.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bridgestan::{Model, StanLibrary};

use crate::diagnostics::{moment, moment_vs_reference, verdict, Moment, Verdict};
use crate::runner::{chain_rng, run_chain, BuiltIn, Init, Sampler};
use crate::target::{BridgeStanTarget, Target};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    MustPass,
    ReportOnly,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::MustPass => "must pass",
            Tier::ReportOnly => "report only",
        }
    }
}

/// What one check measures from a constrained draw.
#[derive(Clone, Debug)]
enum Stat {
    /// `x^power` of the named quantity.
    Power(&'static str, i32),
    /// 1 when `x.1 + x.2 > 0` (mixture_2's mass check).
    PositiveSide,
}

#[derive(Clone, Debug)]
enum Truth {
    Exact(f64),
    /// A reference estimate with its own Monte Carlo error.
    Reference(f64, f64),
}

#[derive(Clone, Debug)]
struct Check {
    label: String,
    stat: Stat,
    truth: Truth,
}

/// One reference target.
pub struct SuiteTarget {
    pub name: &'static str,
    pub tier: Tier,
    pub stan: &'static str,
    pub data: Option<&'static str>,
    checks: Vec<Check>,
}

fn power_checks(quantities: &[(&'static str, f64, f64)]) -> Vec<Check> {
    quantities
        .iter()
        .flat_map(|&(q, m1, m2)| {
            [
                Check {
                    label: format!("E[{q}]"),
                    stat: Stat::Power(q, 1),
                    truth: Truth::Exact(m1),
                },
                Check {
                    label: format!("E[{q}^2]"),
                    stat: Stat::Power(q, 2),
                    truth: Truth::Exact(m2),
                },
            ]
        })
        .collect()
}

const STD_NORMAL: &str = include_str!("../tests/targets/std_normal_10.stan");
const CORR_GAUSS: &str = include_str!("../tests/targets/corr_gauss_10.stan");
const CORR_GAUSS_DATA: &str = include_str!("../tests/targets/corr_gauss_10.json");
const EIGHT_SCHOOLS: &str = include_str!("../tests/targets/eight_schools_nc.stan");
const EIGHT_SCHOOLS_DATA: &str = include_str!("../tests/targets/eight_schools_nc.json");
const EIGHT_SCHOOLS_REF: &str = include_str!("../tests/targets/eight_schools_nc.reference.json");
const FUNNEL: &str = include_str!("../tests/targets/funnel_10.stan");
const BANANA: &str = include_str!("../tests/targets/banana_2.stan");
const MIXTURE: &str = include_str!("../tests/targets/mixture_2.stan");

/// Leak a runtime name into the `&'static str` the checks use (the suite
/// is built once per process).
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// The six targets, in table order.
pub fn targets() -> Vec<SuiteTarget> {
    let x: Vec<&'static str> = (1..=10).map(|i| leak(format!("x.{i}"))).collect();
    let std_q: Vec<(&'static str, f64, f64)> = x.iter().map(|&q| (q, 0.0, 1.0)).collect();
    let sigma: serde_json::Value = serde_json::from_str(CORR_GAUSS_DATA).expect("shipped data");
    let corr_q: Vec<(&'static str, f64, f64)> = x
        .iter()
        .enumerate()
        .map(|(i, &q)| (q, 0.0, sigma["Sigma"][i][i].as_f64().expect("Sigma")))
        .collect();
    let reference: serde_json::Value =
        serde_json::from_str(EIGHT_SCHOOLS_REF).expect("shipped reference");
    let eight: Vec<Check> = reference["quantities"]
        .as_array()
        .expect("quantities")
        .iter()
        .flat_map(|q| {
            // CmdStan's `theta[1]` is BridgeStan's `theta.1`.
            let stan_name = q["name"].as_str().expect("name");
            let name = leak(stan_name.replace('[', ".").replace(']', ""));
            let f = |k: &str| q[k].as_f64().expect("number");
            [
                Check {
                    label: format!("E[{name}]"),
                    stat: Stat::Power(name, 1),
                    truth: Truth::Reference(f("mean"), f("mcse_mean")),
                },
                Check {
                    label: format!("E[{name}^2]"),
                    stat: Stat::Power(name, 2),
                    truth: Truth::Reference(f("m2"), f("mcse_m2")),
                },
            ]
        })
        .collect();
    vec![
        SuiteTarget {
            name: "std_normal_10",
            tier: Tier::MustPass,
            stan: STD_NORMAL,
            data: None,
            checks: power_checks(&std_q),
        },
        SuiteTarget {
            name: "corr_gauss_10",
            tier: Tier::MustPass,
            stan: CORR_GAUSS,
            data: Some(CORR_GAUSS_DATA),
            checks: power_checks(&corr_q),
        },
        SuiteTarget {
            name: "eight_schools_nc",
            tier: Tier::MustPass,
            stan: EIGHT_SCHOOLS,
            data: Some(EIGHT_SCHOOLS_DATA),
            checks: eight,
        },
        SuiteTarget {
            name: "funnel_10",
            tier: Tier::ReportOnly,
            stan: FUNNEL,
            data: None,
            checks: power_checks(&[("v", 0.0, 9.0)]),
        },
        SuiteTarget {
            name: "banana_2",
            tier: Tier::ReportOnly,
            stan: BANANA,
            data: None,
            checks: power_checks(&[("x1", 0.0, 1.0), ("x2", 1.0, 3.25)]),
        },
        SuiteTarget {
            name: "mixture_2",
            tier: Tier::ReportOnly,
            stan: MIXTURE,
            data: None,
            checks: vec![Check {
                label: "P[x.1 + x.2 > 0]".into(),
                stat: Stat::PositiveSide,
                truth: Truth::Exact(0.5),
            }],
        },
    ]
}

// ---- settings: tests/<kernel>.test.toml ------------------------------------------------

/// Draw counts for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Counts {
    pub chains: u64,
    pub warmup: u64,
    pub draws: u64,
    pub thin: u64,
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            chains: 4,
            warmup: 1000,
            draws: 1000,
            thin: 1,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialCounts {
    chains: Option<u64>,
    warmup: Option<u64>,
    draws: Option<u64>,
    thin: Option<u64>,
}

impl PartialCounts {
    fn over(&self, base: Counts) -> Counts {
        Counts {
            chains: self.chains.unwrap_or(base.chains),
            warmup: self.warmup.unwrap_or(base.warmup),
            draws: self.draws.unwrap_or(base.draws),
            thin: self.thin.unwrap_or(base.thin),
        }
    }
}

/// `tests/<kernel>.test.toml`: how the suite runs one kernel.
///
/// ```toml
/// kernel = "mala"          # informational
/// adapt = true             # wrap in adapt() (Level 1's default); false: as given
/// [defaults]               # chains, warmup, draws, thin (4, 1000, 1000, 1)
/// draws = 2000
/// [set]                    # tuning values / settings, as `--set` takes them
/// eps = 0.1
/// [targets.corr_gauss_10]  # per-target overrides of [defaults]
/// draws = 50000
/// thin = 50
/// ```
///
/// `thin` only thins what the diagnostics see; every iteration runs and
/// counts towards the cost. A did-not-converge row asks for more draws
/// here, never for a looser threshold.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestConfig {
    #[serde(default)]
    pub kernel: Option<String>,
    #[serde(default)]
    pub adapt: Option<bool>,
    #[serde(default)]
    defaults: PartialCounts,
    #[serde(default)]
    set: BTreeMap<String, toml::Value>,
    #[serde(default)]
    targets: BTreeMap<String, PartialCounts>,
}

impl TestConfig {
    pub fn parse(path: &str, text: &str) -> anyhow::Result<Self> {
        let c: Self = toml::from_str(text).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        let known: Vec<&str> = targets().iter().map(|t| t.name).collect();
        for t in c.targets.keys() {
            anyhow::ensure!(
                known.contains(&t.as_str()),
                "{path}: [targets.{t}]: no such target; the targets are {}",
                known.join(", ")
            );
        }
        Ok(c)
    }

    pub fn counts(&self, target: &str) -> Counts {
        let base = self.defaults.over(Counts::default());
        self.targets.get(target).map_or(base, |t| t.over(base))
    }

    /// `[set]` and `adapt` as `--set` arguments.
    pub fn sets(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .set
            .iter()
            .map(|(k, v)| match v {
                toml::Value::String(s) => format!("{k}={s}"),
                other => format!("{k}={other}"),
            })
            .collect();
        if let Some(a) = self.adapt {
            out.push(format!("adapt={a}"));
        }
        out
    }
}

/// The built-in kernels' settings, embedded (they are also the files
/// `laplace-sample/tests/<kernel>.test.toml`).
pub fn builtin_config(b: BuiltIn) -> TestConfig {
    let (path, text) = match b {
        BuiltIn::Nuts => (
            "tests/nuts.test.toml",
            include_str!("../tests/nuts.test.toml"),
        ),
        BuiltIn::Rwm => (
            "tests/rwm.test.toml",
            include_str!("../tests/rwm.test.toml"),
        ),
        BuiltIn::Mala => (
            "tests/mala.test.toml",
            include_str!("../tests/mala.test.toml"),
        ),
        BuiltIn::Hmc => (
            "tests/hmc.test.toml",
            include_str!("../tests/hmc.test.toml"),
        ),
    };
    TestConfig::parse(path, text).expect("shipped test config")
}

/// Where a kernel's test file is by convention: `<package>/tests/<name>.test.toml`
/// for `<package>/kernels/<name>.lapkernel`; `<stem>.test.toml` beside a
/// `.lapsampler`.
pub fn conventional_config(kernel_file: &Path) -> PathBuf {
    let stem = kernel_file
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let dir = kernel_file.parent().unwrap_or(Path::new("."));
    if kernel_file.extension().and_then(|e| e.to_str()) == Some("lapsampler") {
        return dir.join(format!("{stem}.test.toml"));
    }
    dir.join("..")
        .join("tests")
        .join(format!("{stem}.test.toml"))
}

// ---- running ---------------------------------------------------------------------------

/// Sampling-phase cost, summed over chains.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub grads: u64,
    pub logps: u64,
    pub secs: f64,
}

/// Constrained draws by quantity: `draws[c][i]` is chain c's i-th kept
/// draw, in `names` order.
pub struct Draws {
    pub names: Vec<String>,
    pub draws: Vec<Vec<Vec<f64>>>,
    pub cost: Cost,
}

/// A compiled reference model and its data, ready for chains.
pub struct Compiled {
    pub lib: Arc<StanLibrary>,
    pub data: Option<PathBuf>,
}

/// Write a shipped target to the cache and compile it (cached by content).
pub fn compile(t: &SuiteTarget) -> anyhow::Result<Compiled> {
    let dir = crate::model::cache_root()?.join("suite");
    std::fs::create_dir_all(&dir)?;
    let write = |file: &str, text: &str| -> anyhow::Result<PathBuf> {
        let p = dir.join(file);
        if std::fs::read_to_string(&p).ok().as_deref() != Some(text) {
            std::fs::write(&p, text)?;
        }
        Ok(p)
    };
    let stan = write(&format!("{}.stan", t.name), t.stan)?;
    let data = t
        .data
        .map(|d| write(&format!("{}.json", t.name), d))
        .transpose()?;
    let prepared = crate::model::prepare(&stan)?;
    Ok(Compiled {
        lib: Arc::new(bridgestan::open_library(&prepared.library)?),
        data,
    })
}

/// One chain: quantity names, kept constrained draws, sampling cost.
type ChainDraws = (Vec<String>, Vec<Vec<f64>>, Cost);

/// Run `n.chains` chains in parallel; keep every `thin`-th constrained
/// draw; count every sampling iteration's cost.
pub fn run(m: &Compiled, sampler: &Sampler, n: Counts, seed: u64) -> anyhow::Result<Draws> {
    let per_chain: Vec<anyhow::Result<ChainDraws>> = std::thread::scope(|s| {
        let hs: Vec<_> = (1..=n.chains)
            .map(|chain| {
                s.spawn(move || -> anyhow::Result<_> {
                    let data = m
                        .data
                        .as_ref()
                        .map(|p| CString::new(p.to_string_lossy().as_bytes()))
                        .transpose()?;
                    let model = Model::new(m.lib.clone(), data, seed as u32)?;
                    let names: Vec<String> = model
                        .param_names(true, false)
                        .split(',')
                        .map(str::to_owned)
                        .collect();
                    let params = crate::sample::model_params(&model)?;
                    let target = Arc::new(BridgeStanTarget { model });
                    let dyn_t: Arc<dyn Target> = target.clone();
                    let mut built = sampler.build_chain(dyn_t, &params, n.warmup, n.draws)?;
                    let mut out = vec![0.0; names.len()];
                    let mut kept = vec![];
                    let mut cost = Cost::default();
                    let mut i = 0u64;
                    let started = std::cell::Cell::new(None);
                    run_chain(
                        target.as_ref(),
                        built.kernel.as_mut(),
                        &Init::Radius(2.0),
                        n.warmup,
                        n.draws,
                        &mut chain_rng(seed, chain),
                        |_| {
                            started.set(Some(std::time::Instant::now()));
                            Ok(())
                        },
                        |st, info, _| {
                            cost.grads += info.n_grad;
                            cost.logps += info.n_logp;
                            if i.is_multiple_of(n.thin) {
                                target.model.param_constrain::<&StanLibrary>(
                                    &st.q, true, false, &mut out, None,
                                )?;
                                kept.push(out.clone());
                            }
                            i += 1;
                            Ok(())
                        },
                    )?;
                    cost.secs = started.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
                    Ok((names, kept, cost))
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("chain thread"))
            .collect()
    });
    let mut names = vec![];
    let mut draws = vec![];
    let mut cost = Cost::default();
    for r in per_chain {
        let (n, d, c) = r?;
        names = n;
        draws.push(d);
        cost.grads += c.grads;
        cost.logps += c.logps;
        cost.secs += c.secs;
    }
    Ok(Draws { names, draws, cost })
}

fn stat_chains(d: &Draws, stat: &Stat) -> anyhow::Result<Vec<Vec<f64>>> {
    let idx = |q: &str| {
        d.names
            .iter()
            .position(|n| n == q)
            .ok_or_else(|| anyhow::anyhow!("the model has no quantity `{q}`"))
    };
    Ok(match stat {
        Stat::Power(q, p) => {
            let k = idx(q)?;
            d.draws
                .iter()
                .map(|c| c.iter().map(|x| x[k].powi(*p)).collect())
                .collect()
        }
        Stat::PositiveSide => {
            let (a, b) = (idx("x.1")?, idx("x.2")?);
            d.draws
                .iter()
                .map(|c| {
                    c.iter()
                        .map(|x| f64::from(u8::from(x[a] + x[b] > 0.0)))
                        .collect()
                })
                .collect()
        }
    })
}

/// The checks of `t` on `d`.
pub fn moments(t: &SuiteTarget, d: &Draws) -> anyhow::Result<Vec<Moment>> {
    t.checks
        .iter()
        .map(|c| {
            let chains = stat_chains(d, &c.stat)?;
            Ok(match c.truth {
                Truth::Exact(v) => moment(&c.label, &chains, v),
                Truth::Reference(v, mcse) => moment_vs_reference(&c.label, &chains, v, mcse),
            })
        })
        .collect()
}

/// One row of the result table.
#[derive(Clone, Debug)]
pub struct Row {
    pub target: &'static str,
    pub tier: Tier,
    pub counts: Counts,
    /// `Err(reason)`: the sampler cannot run on this target (a
    /// `.lapsampler` whose blocks name parameters the target lacks).
    pub outcome: Result<Outcome, String>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub verdict: Verdict,
    pub max_abs_z: f64,
    pub max_rhat: f64,
    pub min_ess: f64,
    pub cost: Cost,
    /// Plain NUTS on the same target, same seed, NUTS's own counts.
    pub nuts: Option<(f64, Cost)>,
    /// The unhealthy or failing checks, for the report.
    pub notes: Vec<String>,
}

impl Outcome {
    pub fn ess_per_grad(&self) -> Option<f64> {
        (self.cost.grads > 0).then(|| self.min_ess / self.cost.grads as f64)
    }
    pub fn ess_per_s(&self) -> f64 {
        self.min_ess / self.cost.secs
    }
}

fn outcome(rows: &[Moment], cost: Cost) -> Outcome {
    let v = verdict(rows);
    let notes = rows
        .iter()
        .filter(|m| !crate::diagnostics::healthy(m) || m.z.abs() >= crate::diagnostics::MAX_ABS_Z)
        .map(|m| {
            format!(
                "{}: estimate {:.4}, truth {:.4}, z {:.2}, R-hat {:.4}, bulk ESS {:.0}",
                m.name, m.estimate, m.truth, m.z, m.rhat, m.ess_bulk
            )
        })
        .collect();
    Outcome {
        verdict: v,
        max_abs_z: rows.iter().map(|m| m.z.abs()).fold(0.0, f64::max),
        max_rhat: rows.iter().map(|m| m.rhat).fold(0.0, f64::max),
        min_ess: rows
            .iter()
            .map(|m| m.ess_bulk)
            .fold(f64::INFINITY, f64::min),
        cost,
        nuts: None,
        notes,
    }
}

/// What `laplace sampler test` was asked to do.
pub struct Request<'a> {
    pub sampler: &'a Sampler,
    pub config: &'a TestConfig,
    pub tier: Option<Tier>,
    pub seed: u64,
    /// Also run plain NUTS on each target for the ratios.
    pub compare: bool,
}

/// Run the suite. `progress` hears about each target as it starts.
pub fn run_suite(req: &Request, mut progress: impl FnMut(&str)) -> anyhow::Result<Vec<Row>> {
    let nuts = Sampler::from_arg("nuts", &[])?;
    let nuts_config = builtin_config(BuiltIn::Nuts);
    let mut rows = vec![];
    for t in targets() {
        if req.tier.is_some_and(|tier| tier != t.tier) {
            continue;
        }
        progress(t.name);
        let counts = req.config.counts(t.name);
        let m = compile(&t)?;
        let outcome = match run(&m, req.sampler, counts, req.seed) {
            // A .lapsampler written for one model names parameters the other
            // targets lack: not applicable there (first reason, one line).
            Err(e) if matches!(req.sampler, Sampler::Composite(_)) => {
                let all = format!("{e:#}");
                let first = all.lines().next().unwrap_or_default();
                // Drop the `file:line:column: ` prefix.
                let reason = first.splitn(4, ':').nth(3).map_or(first, str::trim);
                Err(format!("not applicable: {reason}"))
            }
            Err(e) => return Err(e.context(format!("on {}", t.name))),
            Ok(d) => {
                let mut o = outcome(&moments(&t, &d)?, d.cost);
                if req.compare {
                    let dn = run(&m, &nuts, nuts_config.counts(t.name), req.seed)?;
                    let on = outcome(&moments(&t, &dn)?, dn.cost);
                    o.nuts = Some((on.min_ess, on.cost));
                }
                Ok(o)
            }
        };
        rows.push(Row {
            target: t.name,
            tier: t.tier,
            counts,
            outcome,
        });
    }
    Ok(rows)
}

/// A sampler to test and how: `spec` is a built-in, `pkg::kernel` (an
/// installed kernel package), a `.lapkernel` or a `.lapsampler`. Settings
/// come from `config` if given, else the kernel's conventional test file if
/// it exists, else the defaults (the built-ins' are embedded). `sets` are
/// added after the file's `[set]`.
pub struct Prepared {
    pub sampler: Sampler,
    pub config: TestConfig,
    /// Where the settings came from, for the report.
    pub config_source: String,
}

pub fn prepare(
    spec: &str,
    sets: &[String],
    config: Option<&Path>,
    project: &crate::libraries::Project,
) -> anyhow::Result<Prepared> {
    let load = |p: &Path| -> anyhow::Result<TestConfig> {
        let text = std::fs::read_to_string(p)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", p.display()))?;
        TestConfig::parse(&p.display().to_string(), &text)
    };
    // The file the kernel lives in, if it is one.
    let (file, builtin) = if let Some((package, name)) = spec.split_once("::") {
        let lib = crate::libraries::resolve_import(project, package).map_err(anyhow::Error::msg)?;
        (
            Some(lib.kernel_file(name).map_err(anyhow::Error::msg)?),
            None,
        )
    } else if spec.ends_with(".lapkernel") || spec.ends_with(".lapsampler") {
        (Some(PathBuf::from(spec)), None)
    } else {
        (None, Some(BuiltIn::parse(spec)?))
    };
    let (cfg, config_source) = match (config, &file, builtin) {
        (Some(p), _, _) => (load(p)?, p.display().to_string()),
        (None, Some(f), _) => {
            let conv = conventional_config(f);
            if conv.is_file() {
                (load(&conv)?, conv.display().to_string())
            } else {
                (
                    TestConfig::default(),
                    format!("defaults (no {})", conv.display()),
                )
            }
        }
        (None, None, Some(b)) => (
            builtin_config(b),
            format!("built-in tests/{}.test.toml", b.name()),
        ),
        (None, None, None) => unreachable!(),
    };
    let mut all_sets = cfg.sets();
    all_sets.extend(sets.iter().cloned());
    let arg = file
        .as_ref()
        .map_or(spec.to_string(), |f| f.display().to_string());
    let sampler = if arg.ends_with(".lapsampler") {
        anyhow::ensure!(
            all_sets.is_empty(),
            "a .lapsampler takes no settings: they are in the file (drop --set and the test file's [set]/adapt)"
        );
        Sampler::from_arg(&arg, &[])?
    } else {
        Sampler::from_arg(&arg, &all_sets)?
    };
    Ok(Prepared {
        sampler,
        config: cfg,
        config_source,
    })
}

/// Did every must-pass row that could run pass, and could at least one
/// run? (A `.lapsampler` for one model is not applicable to the others.)
pub fn passed(rows: &[Row]) -> bool {
    let must: Vec<&Row> = rows.iter().filter(|r| r.tier == Tier::MustPass).collect();
    must.iter().any(|r| r.outcome.is_ok())
        && must.iter().all(|r| {
            r.outcome
                .as_ref()
                .map_or(true, |o| o.verdict == Verdict::Pass)
        })
}

fn ratio(a: Option<f64>, b: Option<f64>) -> String {
    match (a, b) {
        (Some(a), Some(b)) if b > 0.0 && a.is_finite() => format!("{:.2}", a / b),
        _ => "n/a".into(),
    }
}

/// The pass/fail table and what to do about anything that did not pass.
pub fn render(name: &str, rows: &[Row]) -> String {
    let mut out = format!("laplace sampler test: {name}\n\n");
    out += &format!(
        "{:<17} {:<12} {:<17} {:>6} {:>7} {:>7} {:>10} {:>10} {:>9} {:>9} {:>8} {:>8}\n",
        "target",
        "tier",
        "verdict",
        "max|z|",
        "R-hat",
        "min ESS",
        "grads",
        "logps",
        "ESS/grad",
        "ESS/s",
        "vs NUTS",
        "(per s)"
    );
    for r in rows {
        match &r.outcome {
            Err(why) => {
                out += &format!("{:<17} {:<12} {why}\n", r.target, r.tier.label());
            }
            Ok(o) => {
                let nuts_grad = o
                    .nuts
                    .and_then(|(e, c)| (c.grads > 0).then(|| e / c.grads as f64));
                let nuts_s = o.nuts.map(|(e, c)| e / c.secs);
                out += &format!(
                    "{:<17} {:<12} {:<17} {:>6.2} {:>7.4} {:>7.0} {:>10} {:>10} {:>9} {:>9.0} {:>8} {:>8}\n",
                    r.target,
                    r.tier.label(),
                    o.verdict.label(),
                    o.max_abs_z,
                    o.max_rhat,
                    o.min_ess,
                    o.cost.grads,
                    o.cost.logps,
                    o.ess_per_grad().map_or("n/a".into(), |v| format!("{v:.5}")),
                    o.ess_per_s(),
                    ratio(o.ess_per_grad(), nuts_grad),
                    ratio(Some(o.ess_per_s()), nuts_s),
                );
            }
        }
    }
    out += "\nESS = minimum bulk ESS over the checked quantities; grads and logps = gradient and \
            density-only evaluations while sampling; ESS/s = per second of sampling, summed over chains; \
            vs NUTS = ESS/grad ratio to plain NUTS on the same target (then ESS/s). Ratios are \
            information, never a threshold.\n";
    for r in rows {
        let Ok(o) = &r.outcome else { continue };
        let c = r.counts;
        match o.verdict {
            Verdict::Pass => {}
            Verdict::Fail => {
                out += &format!(
                    "\n{} FAILED: healthy chains, wrong answer (|z| >= 4) -- the kernel does not target the \
                     right distribution:\n",
                    r.target
                );
                for n in &o.notes {
                    out += &format!("  {n}\n");
                }
            }
            Verdict::DidNotConverge => {
                out += &format!(
                    "\n{} did not converge ({} chains x {} draws, thin {}): the run says nothing either way. \
                     Give it more draws in tests/<kernel>.test.toml:\n  [targets.{}]\n  draws = {}\n  thin = {}\n",
                    r.target,
                    c.chains,
                    c.draws,
                    c.thin,
                    r.target,
                    c.draws * 10,
                    c.thin * 10
                );
                for n in o.notes.iter().take(4) {
                    out += &format!("  {n}\n");
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_six_targets_and_their_tiers() {
        let t = targets();
        let names: Vec<(&str, Tier)> = t.iter().map(|t| (t.name, t.tier)).collect();
        assert_eq!(
            names,
            vec![
                ("std_normal_10", Tier::MustPass),
                ("corr_gauss_10", Tier::MustPass),
                ("eight_schools_nc", Tier::MustPass),
                ("funnel_10", Tier::ReportOnly),
                ("banana_2", Tier::ReportOnly),
                ("mixture_2", Tier::ReportOnly),
            ]
        );
        assert_eq!(t[0].checks.len(), 20);
        // eight schools: mu, tau, theta_tilde[8], theta[8], two moments each.
        assert_eq!(t[2].checks.len(), 2 * 18);
        assert!(t[2].checks.iter().any(|c| c.label == "E[theta.3^2]"));
    }

    #[test]
    fn test_config_parses_overrides_and_sets() {
        let c = TestConfig::parse(
            "k.test.toml",
            "kernel = \"k\"\nadapt = false\n[defaults]\ndraws = 2000\n[set]\neps = 0.1\n\
             [targets.corr_gauss_10]\ndraws = 50000\nthin = 50\n",
        )
        .unwrap();
        assert_eq!(
            c.counts("std_normal_10"),
            Counts {
                draws: 2000,
                ..Counts::default()
            }
        );
        assert_eq!(
            c.counts("corr_gauss_10"),
            Counts {
                draws: 50000,
                thin: 50,
                ..Counts::default()
            }
        );
        assert_eq!(c.sets(), vec!["eps=0.1", "adapt=false"]);
        let e = TestConfig::parse("k.test.toml", "[targets.nope]\ndraws = 1\n").unwrap_err();
        assert!(
            e.to_string().contains("[targets.nope]: no such target"),
            "{e}"
        );
        assert!(
            TestConfig::parse("k.test.toml", "draws = 1\n").is_err(),
            "unknown keys refused"
        );
        for b in BuiltIn::ALL {
            builtin_config(b);
        }
    }

    #[test]
    fn the_scaffolded_test_file_parses() {
        let tmp = tempfile::tempdir().unwrap();
        crate::authoring::new_kernel(
            "k",
            crate::authoring::KernelKind::Proposal,
            tmp.path(),
            None,
        )
        .unwrap();
        let text = std::fs::read_to_string(tmp.path().join("k/tests/k.test.toml")).unwrap();
        let c = TestConfig::parse("k.test.toml", &text).unwrap();
        assert_eq!(c.counts("std_normal_10"), Counts::default());
        assert_eq!(
            conventional_config(&tmp.path().join("k/kernels/k.lapkernel")),
            tmp.path().join("k/kernels/../tests/k.test.toml")
        );
    }

    #[test]
    fn must_pass_rows_decide() {
        let ok = |v| {
            Ok(Outcome {
                verdict: v,
                max_abs_z: 0.0,
                max_rhat: 1.0,
                min_ess: 500.0,
                cost: Cost::default(),
                nuts: None,
                notes: vec![],
            })
        };
        let row = |tier, outcome| Row {
            target: "t",
            tier,
            counts: Counts::default(),
            outcome,
        };
        assert!(passed(&[
            row(Tier::MustPass, ok(Verdict::Pass)),
            row(Tier::ReportOnly, ok(Verdict::Fail))
        ]));
        assert!(!passed(&[row(Tier::MustPass, ok(Verdict::DidNotConverge))]));
        assert!(
            !passed(&[row(Tier::MustPass, Err("n/a".into()))]),
            "nothing ran"
        );
        assert!(passed(&[
            row(Tier::MustPass, Err("n/a".into())),
            row(Tier::MustPass, ok(Verdict::Pass))
        ]));
    }
}
