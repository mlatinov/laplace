//! CmdStan-shaped CSV output that `cmdstanr::as_cmdstan_fit()` reads.
//!
//! The header contract comes from Phase 0.4 (`docs/sampler/phase-0-report.md`)
//! and its review:
//! - the seven comment keys cmdstanr needs (`model`, `method`,
//!   `num_samples`, `save_warmup`, `thin`, `metric`, `id`), plus honest
//!   extras (`num_warmup`, `seed`);
//! - provenance: one `# laplace_sampler = <name> (sha256: …)` line, one
//!   `# laplace_kernel_<label> = <name> <version> <hash>` line per block, and
//!   `# laplace_settings = …`, the text the sampler hash covers;
//! - **always** the seven NUTS columns. A value that does not apply to the
//!   kernel is written as `NaN` (Q4), never `0`;
//! - no timestamps or timings anywhere, so the same seed gives a
//!   byte-identical file.

use std::io::Write;

use crate::kernel::{StepInfo, Tuning};
use crate::sha256_hex;

pub const SAMPLER_COLUMNS: [&str; 8] = [
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
    "n_grad__",
];

/// Everything the header says about a run.
#[derive(Clone, Debug)]
pub struct Header {
    /// Stan's model name (`<stem>_model`).
    pub model: String,
    pub chain_id: u64,
    pub seed: u64,
    pub warmup: u64,
    pub draws: u64,
    pub thin: u64,
    /// `--sampler` name.
    pub sampler: String,
    /// `key:value ...` of every resolved setting.
    pub settings: String,
    /// One entry per block.
    pub kernels: Vec<KernelLine>,
}

/// One block's kernel in the provenance header.
#[derive(Clone, Debug)]
pub struct KernelLine {
    pub label: String,
    pub name: String,
    /// The laplace-sample version for built-ins, `local` for a file.
    pub version: String,
    /// Hashed, never written out.
    pub source: String,
}

impl Header {
    /// The text the `laplace_sampler` hash covers: sampler, settings, and
    /// each block's label, kernel and kernel hash, one per line.
    pub fn sampler_text(&self) -> String {
        let mut s = format!("sampler {}\nsettings {}\n", self.sampler, self.settings);
        for k in &self.kernels {
            s += &format!(
                "block {} {} {} sha256:{}\n",
                k.label,
                k.name,
                k.version,
                sha256_hex(k.source.as_bytes())
            );
        }
        s
    }

    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![
            format!("# model = {}", self.model),
            "# method = sample".to_string(),
            format!("# num_samples = {}", self.draws),
            format!("# num_warmup = {}", self.warmup),
            "# save_warmup = false".to_string(),
            format!("# thin = {}", self.thin),
            "# metric = diag_e".to_string(),
            format!("# id = {}", self.chain_id),
            format!("# seed = {}", self.seed),
            format!("# laplace_sample_version = {}", crate::VERSION),
            format!(
                "# laplace_sampler = {} (sha256: {})",
                self.sampler,
                sha256_hex(self.sampler_text().as_bytes())
            ),
        ];
        for k in &self.kernels {
            out.push(format!(
                "# laplace_kernel_{} = {} {} sha256:{}",
                k.label,
                k.name,
                k.version,
                sha256_hex(k.source.as_bytes())
            ));
        }
        out.push(format!("# laplace_settings = {}", self.settings));
        out
    }
}

/// CmdStan's post-warmup adaptation block (after the column names). cmdstanr
/// reads the step size and the diagonal inverse metric from it.
pub fn adaptation_lines(t: &Tuning) -> Vec<String> {
    let mut out = vec!["# Adaptation terminated".to_string()];
    if let Some(eps) = t.step_size {
        out.push(format!("# Step size = {}", num(eps)));
    }
    if let Some(m) = &t.inv_metric {
        out.push("# Diagonal elements of inverse mass matrix:".to_string());
        out.push(format!(
            "# {}",
            m.iter().map(|x| num(*x)).collect::<Vec<_>>().join(", ")
        ));
    }
    out
}

/// Shortest round-trip decimal; `NaN`, `inf`, `-inf` as R reads them.
pub fn num(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x == f64::INFINITY {
        "inf".to_string()
    } else if x == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{x}")
    }
}

fn opt_f(x: Option<f64>) -> String {
    x.map(num).unwrap_or_else(|| "NaN".to_string())
}

fn opt_u(x: Option<u64>) -> String {
    x.map(|v| v.to_string())
        .unwrap_or_else(|| "NaN".to_string())
}

/// The sampler columns of one row, in [`SAMPLER_COLUMNS`] order.
pub fn sampler_values(lp: f64, info: &StepInfo, tuning: &Tuning) -> Vec<String> {
    vec![
        num(lp),
        opt_f(info.accept_prob),
        opt_f(tuning.step_size),
        opt_u(info.treedepth),
        opt_u(info.n_leapfrog),
        opt_u(info.divergent.map(u64::from)),
        opt_f(info.energy),
        info.n_grad.to_string(),
    ]
}

pub struct Writer<W: Write> {
    out: W,
}

impl<W: Write> Writer<W> {
    pub fn new(mut out: W, header: &Header, param_names: &[String]) -> std::io::Result<Self> {
        for l in header.lines() {
            writeln!(out, "{l}")?;
        }
        let mut cols: Vec<String> = SAMPLER_COLUMNS.iter().map(|s| s.to_string()).collect();
        cols.extend(param_names.iter().cloned());
        writeln!(out, "{}", cols.join(","))?;
        Ok(Self { out })
    }

    pub fn comment_lines(&mut self, lines: &[String]) -> std::io::Result<()> {
        for l in lines {
            writeln!(self.out, "{l}")?;
        }
        Ok(())
    }

    pub fn row(&mut self, sampler: &[String], params: &[f64]) -> std::io::Result<()> {
        let mut first = true;
        for v in sampler {
            if !first {
                self.out.write_all(b",")?;
            }
            first = false;
            self.out.write_all(v.as_bytes())?;
        }
        for p in params {
            self.out.write_all(b",")?;
            self.out.write_all(num(*p).as_bytes())?;
        }
        self.out.write_all(b"\n")
    }

    pub fn finish(mut self) -> std::io::Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            model: "m_model".into(),
            chain_id: 2,
            seed: 11,
            warmup: 100,
            draws: 50,
            thin: 1,
            sampler: "mala".into(),
            settings: "adapt:true step_size:1 target_accept:0.574".into(),
            kernels: vec![KernelLine {
                label: "rest".into(),
                name: "mala".into(),
                version: crate::VERSION.into(),
                source: "fn step() {}".into(),
            }],
        }
    }

    #[test]
    fn header_has_the_cmdstanr_keys_and_provenance() {
        let l = header().lines();
        for key in [
            "# model = m_model",
            "# method = sample",
            "# num_samples = 50",
            "# save_warmup = false",
            "# thin = 1",
            "# metric = diag_e",
            "# id = 2",
        ] {
            assert!(l.iter().any(|x| x == key), "missing {key}");
        }
        assert!(l
            .iter()
            .any(|x| x.starts_with("# laplace_sampler = mala (sha256: ")));
        let k = l
            .iter()
            .find(|x| x.starts_with("# laplace_kernel_rest = "))
            .unwrap();
        assert_eq!(
            k,
            &format!(
                "# laplace_kernel_rest = mala {} sha256:{}",
                crate::VERSION,
                sha256_hex(b"fn step() {}")
            )
        );
    }

    #[test]
    fn sampler_hash_changes_with_settings() {
        let a = header();
        let mut b = header();
        b.settings = "adapt:true step_size:0.5 target_accept:0.574".into();
        assert_ne!(a.lines(), b.lines());
    }

    #[test]
    fn inapplicable_columns_are_nan_never_zero() {
        let info = StepInfo {
            accept_prob: Some(0.5),
            n_grad: 1,
            ..Default::default()
        };
        let t = Tuning {
            step_size: Some(0.25),
            inv_metric: None,
        };
        assert_eq!(
            sampler_values(-1.5, &info, &t),
            vec!["-1.5", "0.5", "0.25", "NaN", "NaN", "NaN", "NaN", "1"]
        );
    }

    #[test]
    fn rows_are_plain_csv() {
        let mut buf = vec![];
        let mut w = Writer::new(&mut buf, &header(), &["x.1".into(), "x.2".into()]).unwrap();
        w.row(&["1".into(), "2".into()], &[0.5, f64::NAN]).unwrap();
        w.finish().unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains(
            "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,n_grad__,x.1,x.2\n"
        ));
        assert!(s.ends_with("1,2,0.5,NaN\n"));
    }
}
