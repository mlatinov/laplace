//! 0.2 — Parameter mapping: unconstrained names -> parameters.
use rand::{RngExt, SeedableRng};
use rand::rngs::ChaCha8Rng;
use sampler_phase0_spike::*;

/// The base parameter name of a flattened name: everything before the first '.'.
/// (Stan's flattened names are `name`, `name.1`, `name.1.2`; '.' cannot occur in
/// an identifier, so the split is unambiguous.)
fn base(s: &str) -> &str {
    s.split('.').next().unwrap()
}

/// Contiguous runs of the same base name, as (base, start, len).
fn runs(names: &[String]) -> Vec<(String, usize, usize)> {
    let mut out: Vec<(String, usize, usize)> = vec![];
    for (i, n) in names.iter().enumerate() {
        match out.last_mut() {
            Some((b, _, len)) if b == base(n) => *len += 1,
            _ => out.push((base(n).to_owned(), i, 1)),
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    for name in ["simplex", "mapping"] {
        let lib = library(name)?;
        let m = model(&lib, name, 1)?;
        let unc = names(m.param_unc_names());
        let par = names(m.param_names(false, false));
        let all = names(m.param_names(true, true));
        println!("=== {name}");
        println!("param_unc_num = {}", m.param_unc_num());
        println!("param_num(false,false) = {}  param_num(true,true) = {}",
                 m.param_num(false, false), m.param_num(true, true));
        println!("param_unc_names = {unc:?}");
        println!("param_names(tp,gq) = {all:?}");

        let ur = runs(&unc);
        let pr = runs(&par);
        println!("{:<8} {:>9} {:>8} {:>9} {:>8}", "param", "unc_start", "unc_len", "con_start", "con_len");
        assert_eq!(ur.len(), pr.len(), "same parameters, same order");
        for (u, p) in ur.iter().zip(&pr) {
            assert_eq!(u.0, p.0, "declaration order agrees");
            println!("{:<8} {:>9} {:>8} {:>9} {:>8}", u.0, u.1, u.2, p.1, p.2);
        }
        // unc runs tile 0..n with no gaps
        let mut next = 0;
        for (_, s, l) in &ur {
            assert_eq!(*s, next);
            next += l;
        }
        assert_eq!(next, m.param_unc_num());

        // constrain with TP + GQ (needs an Rng), then round-trip
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let q: Vec<f64> = (0..m.param_unc_num()).map(|_| rng.random_range(-2.0..2.0)).collect();
        let mut out = vec![0.0; m.param_num(true, true)];
        let mut srng = m.new_rng(42)?;
        m.param_constrain(&q, true, true, &mut out, Some(&mut srng))?;
        let mut q2 = vec![0.0; q.len()];
        m.param_unconstrain(&out[..m.param_num(false, false)], &mut q2)?;
        let rt = q.iter().zip(&q2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("constrained draw = {:?}", out.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
        println!("unconstrain(constrain(q)) max abs err = {rt:.2e}");

        // same rng seed -> same GQ; fresh rng stream advances
        let mut a = vec![0.0; out.len()];
        let mut r1 = m.new_rng(42)?;
        m.param_constrain(&q, true, true, &mut a, Some(&mut r1))?;
        assert_eq!(a, out, "GQ deterministic given rng seed");
    }
    Ok(())
}
