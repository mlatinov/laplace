//! The built-in kernels (`nuts`, `rwm`, `mala`, `hmc`): their names,
//! settings and defaults, and the validation of settings. Model-free, so
//! the `.lapsampler` checker and the editor use it; laplace-sample builds
//! the kernels (`runner::BuildKernel`).

use std::collections::BTreeMap;

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

    /// Settings it accepts inside a `.lapsampler`. Adaptation is explicit
    /// there (`adapt(...)`), so `adapt` and `target_accept` are not kernel
    /// settings, except for nuts, which adapts itself.
    pub fn sampler_settings(self) -> &'static [&'static str] {
        match self {
            BuiltIn::Nuts => &["step_size", "target_accept", "max_depth"],
            BuiltIn::Rwm | BuiltIn::Mala => &["step_size"],
            BuiltIn::Hmc => &["step_size", "steps", "jitter"],
        }
    }

    /// One line: what it is (from the first line of its implementation's
    /// module doc).
    pub fn brief(self) -> &'static str {
        match self {
            BuiltIn::Nuts => {
                "the No-U-Turn sampler (upstream nuts-rs), with its own warmup adaptation"
            }
            BuiltIn::Rwm => "random-walk Metropolis with a diagonal metric",
            BuiltIn::Mala => "Metropolis-adjusted Langevin algorithm with a diagonal metric",
            BuiltIn::Hmc => {
                "static Hamiltonian Monte Carlo: leapfrog with a diagonal metric and no U-turn \
                 criterion, then a Metropolis correction"
            }
        }
    }

    /// How it adapts in a `.lapsampler`.
    pub fn adaptation(self) -> &'static str {
        match self {
            BuiltIn::Nuts => "adapts itself (nuts-rs's warmup adaptation)",
            BuiltIn::Rwm => "adapt(rwm()) tunes step size and metric (target 0.234)",
            BuiltIn::Mala => "adapt(mala()) tunes step size and metric (target 0.574)",
            BuiltIn::Hmc => "adapt(hmc()) tunes step size and metric (target 0.8)",
        }
    }

    /// Its line in `laplace sampler list`.
    pub fn line(self) -> String {
        format!(
            "  {:<6} settings: {}; {}",
            self.name(),
            self.sampler_settings().join(", "),
            self.adaptation()
        )
    }

    /// What `laplace sampler doc <name>` prints.
    pub fn doc(self) -> String {
        format!(
            "{} is a built-in kernel; `laplace sampler list` shows its settings.\n{}",
            self.name(),
            self.line()
        )
    }

    /// Its implementation: the path in the laplace repository, and the
    /// source (provenance hashes it; an editor can show it).
    pub fn source(self) -> (&'static str, &'static str) {
        match self {
            BuiltIn::Nuts => (
                "laplace-sample/src/kernels/nuts.rs",
                include_str!("../../laplace-sample/src/kernels/nuts.rs"),
            ),
            BuiltIn::Rwm => (
                "laplace-sample/src/kernels/rwm.rs",
                include_str!("../../laplace-sample/src/kernels/rwm.rs"),
            ),
            BuiltIn::Mala => (
                "laplace-sample/src/kernels/mala.rs",
                include_str!("../../laplace-sample/src/kernels/mala.rs"),
            ),
            BuiltIn::Hmc => (
                "laplace-sample/src/kernels/hmc.rs",
                include_str!("../../laplace-sample/src/kernels/hmc.rs"),
            ),
        }
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
}
