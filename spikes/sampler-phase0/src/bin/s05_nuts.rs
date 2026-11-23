//! 0.5 — nuts-rs over BridgeStan.
//!
//! Part A: full-model NUTS on std_normal and gp -> out/nutsrs/<model>-<chain>.csv,
//!         compared against CmdStan in R/check_nuts.R.
//! Part B: two-block Gibbs sweep on `blocks`, NUTS on each block, with three
//!         ways of telling a chain that the *other* block moved:
//!           set_position     — official API (re-initialises adaptation)
//!           stale            — tell it nothing (cached logp/grad are stale)
//!           refresh_position — vendored one-method patch (state only)
//!         -> out/gibbs/<variant>-<chain>.csv
use anyhow::Result;
use bridgestan::{Model, StanLibrary};
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings};
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::ChaCha8Rng;
use sampler_phase0_spike::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

type M = Model<&'static StanLibrary>;

#[derive(Debug)]
struct BsErr(String);
impl std::fmt::Display for BsErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BsErr {}
impl LogpError for BsErr {
    // Stan rejections (domain errors) are recoverable: treat as divergence.
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// log p(block | rest) via the full BridgeStan model.
struct BlockLogp {
    m: &'static M,
    full: Rc<RefCell<Vec<f64>>>,
    start: usize,
    len: usize,
    grad_full: Vec<f64>,
    n_grad: Rc<Cell<u64>>,
}

impl HasDims for BlockLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [("unconstrained_parameter".to_string(), self.len as u64)].into_iter().collect()
    }
}

impl CpuLogpFunc for BlockLogp {
    type LogpError = BsErr;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;
    fn dim(&self) -> usize {
        self.len
    }
    fn logp(&mut self, pos: &[f64], grad: &mut [f64]) -> Result<f64, BsErr> {
        let mut q = self.full.borrow().clone();
        q[self.start..self.start + self.len].copy_from_slice(pos);
        self.n_grad.set(self.n_grad.get() + 1);
        let lp = self
            .m
            .log_density_gradient(&q, true, true, &mut self.grad_full)
            .map_err(|e| BsErr(e.to_string()))?;
        grad.copy_from_slice(&self.grad_full[self.start..self.start + self.len]);
        Ok(lp)
    }
    fn expand_vector<R: Rng + ?Sized>(&mut self, _: &mut R, a: &[f64]) -> Result<Vec<f64>, CpuMathError> {
        Ok(a.to_vec())
    }
}

fn leak_model(name: &str) -> Result<&'static M> {
    let lib: &'static StanLibrary = Box::leak(Box::new(library(name)?));
    Ok(Box::leak(Box::new(model(lib, name, 1)?)))
}

fn settings(tune: u64, draws: u64) -> DiagNutsSettings {
    let mut s = DiagNutsSettings::default();
    s.num_tune = tune;
    s.num_draws = draws;
    s
}

fn init(rng: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.random_range(-2.0..2.0)).collect()
}

struct Csv {
    f: std::io::BufWriter<std::fs::File>,
}
impl Csv {
    fn new(path: &str, cols: &[String]) -> Result<Self> {
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "{}", cols.join(","))?;
        Ok(Self { f })
    }
    fn row(&mut self, v: &[f64]) -> Result<()> {
        let s: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
        writeln!(self.f, "{}", s.join(","))?;
        Ok(())
    }
}

fn part_a(name: &str) -> Result<()> {
    let m = leak_model(name)?;
    let n = m.param_unc_num();
    let mut cols: Vec<String> = ["lp__", "stepsize__", "n_grad__", "divergent__", "n_leapfrog__"]
        .map(String::from)
        .to_vec();
    cols.extend(names(m.param_names(false, false)));
    let (tune, draws) = (1000, 1000);
    let mut total_grad = 0;
    let t = std::time::Instant::now();
    for chain in 0..4u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(1000 + chain);
        let q0 = init(&mut rng, n);
        let n_grad = Rc::new(Cell::new(0));
        let logp = BlockLogp {
            m,
            full: Rc::new(RefCell::new(q0.clone())),
            start: 0,
            len: n,
            grad_full: vec![0.0; n],
            n_grad: n_grad.clone(),
        };
        let mut c = settings(tune, draws).new_chain(chain, CpuMath::new(logp), &mut rng)?;
        c.set_position(&q0)?;
        let mut csv = Csv::new(&format!("out/nutsrs/{name}-{}.csv", chain + 1), &cols)?;
        let mut out = vec![0.0; m.param_num(false, false)];
        for i in 0..tune + draws {
            let (pos, p) = c.draw()?;
            if i >= tune {
                let lp = m.log_density(&pos, true, true)?;
                m.param_constrain::<&StanLibrary>(&pos, false, false, &mut out, None)?;
                let mut row = vec![lp, p.step_size, p.num_gradients as f64, p.diverging as u8 as f64, p.num_steps as f64];
                row.extend_from_slice(&out);
                csv.row(&row)?;
            }
        }
        total_grad += n_grad.get();
    }
    println!("A {name:>10}: 4 chains x ({tune}+{draws}) in {:.2}s, {total_grad} gradient evals",
             t.elapsed().as_secs_f64());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    SetPosition,
    Stale,
    Refresh,
}

fn part_b(variant: Variant) -> Result<()> {
    let m = leak_model("blocks")?;
    let n = m.param_unc_num(); // a = 0..3, b = 3..6
    let (tune, draws) = (1000u64, 2000u64);
    let mut cols: Vec<String> = ["stepsize_a__", "stepsize_b__", "n_grad__"].map(String::from).to_vec();
    cols.extend(names(m.param_names(false, false)));
    let mut ss_a_sampling = vec![];
    let mut grads = 0;
    for chain in 0..4u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(2000 + chain);
        let full = Rc::new(RefCell::new(init(&mut rng, n)));
        let n_grad = Rc::new(Cell::new(0));
        let mk = |start| BlockLogp { m, full: full.clone(), start, len: 3, grad_full: vec![0.0; n], n_grad: n_grad.clone() };
        let mut ca = settings(tune, draws).new_chain(chain, CpuMath::new(mk(0)), &mut rng)?;
        let mut cb = settings(tune, draws).new_chain(100 + chain, CpuMath::new(mk(3)), &mut rng)?;
        let a0 = full.borrow()[0..3].to_vec();
        let b0 = full.borrow()[3..6].to_vec();
        ca.set_position(&a0)?;
        cb.set_position(&b0)?;
        let mut csv = Csv::new(&format!("out/gibbs/{variant:?}-{}.csv", chain + 1), &cols)?;
        for i in 0..tune + draws {
            let g0 = n_grad.get();
            // block a | b
            let a_cur = full.borrow()[0..3].to_vec();
            if i > 0 {
                match variant {
                    Variant::SetPosition => ca.set_position(&a_cur)?,
                    Variant::Refresh => ca.refresh_position(&a_cur)?,
                    Variant::Stale => {}
                }
            }
            let (a_new, pa) = ca.draw()?;
            full.borrow_mut()[0..3].copy_from_slice(&a_new);
            // block b | a
            let b_cur = full.borrow()[3..6].to_vec();
            match variant {
                Variant::SetPosition => cb.set_position(&b_cur)?,
                Variant::Refresh => cb.refresh_position(&b_cur)?,
                Variant::Stale => {}
            }
            let (b_new, pb) = cb.draw()?;
            full.borrow_mut()[3..6].copy_from_slice(&b_new);
            if i >= tune {
                ss_a_sampling.push(pa.step_size);
                let mut row = vec![pa.step_size, pb.step_size, (n_grad.get() - g0) as f64];
                row.extend_from_slice(&full.borrow());
                csv.row(&row)?;
            }
        }
        grads += n_grad.get();
    }
    let mean = ss_a_sampling.iter().sum::<f64>() / ss_a_sampling.len() as f64;
    let sd = (ss_a_sampling.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ss_a_sampling.len() as f64).sqrt();
    println!("B {variant:>16?}: block-a step size in sampling mean={mean:.4} sd={sd:.4}; total gradient evals={grads}");
    Ok(())
}

/// Reference: NUTS on the fixed conditional a | b = 0 (true conditional sd 0.436).
fn part_b_reference() -> Result<()> {
    let m = leak_model("blocks")?;
    let full = Rc::new(RefCell::new(vec![0.0; 6]));
    let logp = BlockLogp { m, full, start: 0, len: 3, grad_full: vec![0.0; 6], n_grad: Rc::new(Cell::new(0)) };
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let mut c = settings(1000, 2000).new_chain(0, CpuMath::new(logp), &mut rng)?;
    c.set_position(&[0.5, -0.5, 0.1])?;
    let mut ss = vec![];
    for i in 0..3000 {
        let (_, p) = c.draw()?;
        if i >= 1000 {
            ss.push(p.step_size);
        }
    }
    let mean = ss.iter().sum::<f64>() / ss.len() as f64;
    println!("B {:>16}: block-a step size on the fixed conditional mean={mean:.4}", "reference");
    Ok(())
}

fn main() -> Result<()> {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let all = which.is_empty();
    if all || which.iter().any(|w| w == "a") {
        part_a("std_normal")?;
        part_a("gp")?;
    }
    if all || which.iter().any(|w| w == "b") {
        part_b_reference()?;
        for v in [Variant::SetPosition, Variant::Stale, Variant::Refresh] {
            part_b(v)?;
        }
    }
    Ok(())
}
