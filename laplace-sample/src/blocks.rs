//! Blocks: which unconstrained coordinates each parameter owns (Phase 3.1).
//!
//! BridgeStan names unconstrained coordinates `name` or `name.k`, in
//! declaration order and contiguous per parameter, with `k` counting
//! *unconstrained* coordinates (a `simplex[K]` has `K - 1`; a
//! `cholesky_factor_corr[3]` has 3), not indexing the parameter's shape
//! (Phase 0.2). A parameter is the run of names whose text before the first
//! `.` is its name. A block is a set of whole parameters (D7: no slices).

/// One model parameter's unconstrained coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub start: usize,
    pub len: usize,
}

/// Group BridgeStan's `param_unc_names` into parameters. Errors if a name
/// reappears after another parameter (the runs must be contiguous).
pub fn parameters(unc_names: &[String]) -> anyhow::Result<Vec<Param>> {
    let mut out: Vec<Param> = vec![];
    for (i, n) in unc_names.iter().enumerate() {
        let base = n.split('.').next().unwrap_or(n);
        match out.last_mut() {
            Some(p) if p.name == base => p.len += 1,
            _ => {
                anyhow::ensure!(
                    !out.iter().any(|p| p.name == base),
                    "unconstrained coordinates of `{base}` are not contiguous"
                );
                out.push(Param {
                    name: base.to_string(),
                    start: i,
                    len: 1,
                });
            }
        }
    }
    Ok(out)
}

/// A set of whole parameters updated together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The `as` name, or the parameter names joined by `_`.
    pub label: String,
    pub params: Vec<String>,
    /// Unconstrained coordinates, ascending.
    pub indices: Vec<usize>,
}

impl Block {
    /// The block of `names` (in the order given); `label` overrides the
    /// default label.
    pub fn new(params: &[Param], names: &[String], label: Option<&str>) -> anyhow::Result<Self> {
        let mut indices = vec![];
        for n in names {
            let p = params.iter().find(|p| &p.name == n).ok_or_else(|| {
                anyhow::anyhow!(
                    "the model has no parameter `{n}`; its parameters are: {}",
                    params
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
            indices.extend(p.start..p.start + p.len);
        }
        indices.sort_unstable();
        indices.dedup();
        Ok(Self {
            label: label.map_or_else(|| names.join("_"), str::to_string),
            params: names.to_vec(),
            indices,
        })
    }

    /// Every parameter, labelled `label` (Level 1's whole model).
    pub fn all(params: &[Param], label: &str) -> Self {
        let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
        Self::new(params, &names, Some(label)).expect("own parameters")
    }

    pub fn dim(&self) -> usize {
        self.indices.len()
    }

    /// This block's coordinates of a full vector.
    pub fn take(&self, full: &[f64]) -> Vec<f64> {
        self.indices.iter().map(|&i| full[i]).collect()
    }

    /// Write this block's coordinates into a full vector.
    pub fn put(&self, full: &mut [f64], part: &[f64]) {
        for (k, &i) in self.indices.iter().enumerate() {
            full[i] = part[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Evaluator, View};
    use crate::target::analytic::Gaussian;
    use crate::target::{BlockTarget, Target};
    use std::sync::Arc;

    fn names(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// The Phase 0.2 mapping model's unconstrained names, verbatim.
    fn mapping_model() -> Vec<String> {
        names(&[
            "mu", "sigma", "beta.1", "beta.2", "beta.3", "theta.1", "theta.2", "theta.3", "L.1",
            "L.2", "L.3", "M.1.1", "M.2.1", "M.1.2", "M.2.2", "r.1", "r.2",
        ])
    }

    #[test]
    fn parameters_follow_the_phase_0_mapping() {
        let p = parameters(&mapping_model()).unwrap();
        let got: Vec<(&str, usize, usize)> = p
            .iter()
            .map(|p| (p.name.as_str(), p.start, p.len))
            .collect();
        assert_eq!(
            got,
            vec![
                ("mu", 0, 1),
                ("sigma", 1, 1),
                ("beta", 2, 3),
                ("theta", 5, 3), // simplex[4]: 3 unconstrained coordinates
                ("L", 8, 3),     // cholesky_factor_corr[3]: 3
                ("M", 11, 4),
                ("r", 15, 2),
            ]
        );
    }

    #[test]
    fn non_contiguous_names_are_an_error() {
        assert!(parameters(&names(&["a.1", "b", "a.2"])).is_err());
    }

    #[test]
    fn blocks_from_names() {
        let p = parameters(&mapping_model()).unwrap();
        let b = Block::new(&p, &names(&["theta", "mu"]), None).unwrap();
        assert_eq!(b.label, "theta_mu");
        assert_eq!(b.indices, vec![0, 5, 6, 7]);
        let b = Block::new(&p, &names(&["sigma"]), Some("sc")).unwrap();
        assert_eq!((b.label.as_str(), b.indices.clone()), ("sc", vec![1]));
        let e = Block::new(&p, &names(&["tau"]), None).unwrap_err();
        assert!(e.to_string().contains("no parameter `tau`"), "{e}");
        let full: Vec<f64> = (0..17).map(|i| i as f64).collect();
        let b = Block::new(&p, &names(&["beta"]), None).unwrap();
        assert_eq!(b.take(&full), vec![2.0, 3.0, 4.0]);
        let mut f2 = full.clone();
        b.put(&mut f2, &[-1.0, -2.0, -3.0]);
        assert_eq!(&f2[2..5], &[-1.0, -2.0, -3.0]);
    }

    /// A 6-d Gaussian whose two 3-coordinate blocks are correlated: every
    /// entry of the precision's off-diagonal block is non-zero, so the
    /// conditional of one block depends on the other.
    fn coupled() -> Gaussian {
        let mut p = vec![vec![0.0; 6]; 6];
        for (i, row) in p.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = if i == j {
                    2.0
                } else {
                    0.3 + 0.05 * (i + j) as f64
                };
            }
        }
        Gaussian { precision: p }
    }

    /// Phase 3.1's check: when *another* block moves, re-evaluating this
    /// block at the same point must give the new value, never the cached
    /// one. On correlated blocks the two differ (asserted), so a stale
    /// cache would fail this test; on independent blocks it could not.
    #[test]
    fn moving_another_block_never_returns_a_stale_gradient() {
        let t = coupled();
        let mut ev = Evaluator::new(&t);
        let a = [0.4, -0.2, 0.7]; // block A = coordinates 0..3
        let base0 = vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.5];
        let mut base1 = base0.clone();
        base1[3..].copy_from_slice(&[-0.8, 0.9, 0.1]); // block B moves
        ev.set_view(Some(View {
            indices: vec![0, 1, 2],
            base: base0.clone(),
        }));
        let (lp0, g0) = ev.eval(&a).unwrap();
        ev.set_view(Some(View {
            indices: vec![0, 1, 2],
            base: base1.clone(),
        }));
        let before = ev.n_grad();
        let (lp1, g1) = ev.eval(&a).unwrap();
        assert_eq!(ev.n_grad(), before + 1, "a new evaluation, not a cache hit");
        // The truth after B moved...
        let mut full = base1.clone();
        full[..3].copy_from_slice(&a);
        let mut g = vec![0.0; 6];
        let lp_true = t.log_density_gradient(&full, &mut g).unwrap();
        assert_eq!(lp1, lp_true);
        assert_eq!(g1, g[..3].to_vec());
        // ...is not what was cached before it moved.
        assert_ne!(lp1, lp0, "blocks are correlated: the stale value differs");
        assert!(
            g1.iter().zip(&g0).all(|(x, y)| x != y),
            "every gradient entry changed"
        );
        // Moving B back hits the cache again: the key is the full position.
        ev.set_view(Some(View {
            indices: vec![0, 1, 2],
            base: base0,
        }));
        let n = ev.n_grad();
        assert_eq!(ev.eval(&a).unwrap(), (lp0, g0));
        assert_eq!(ev.n_grad(), n);
    }

    #[test]
    fn block_target_reads_the_current_rest() {
        let full: Arc<dyn Target> = Arc::new(coupled());
        let bt = BlockTarget::new(full.clone(), vec![3, 4, 5], vec![0.1; 6]);
        let b = [0.5, 0.5, 0.5];
        let mut g0 = vec![0.0; 3];
        let lp0 = bt.log_density_gradient(&b, &mut g0).unwrap();
        bt.rest.lock().unwrap()[..3].copy_from_slice(&[1.0, -1.0, 2.0]); // block A moves
        let mut g1 = vec![0.0; 3];
        let lp1 = bt.log_density_gradient(&b, &mut g1).unwrap();
        let fullq = vec![1.0, -1.0, 2.0, 0.5, 0.5, 0.5];
        let mut g = vec![0.0; 6];
        assert_eq!(lp1, full.log_density_gradient(&fullq, &mut g).unwrap());
        assert_eq!(g1, g[3..].to_vec());
        assert_ne!((lp0, g0), (lp1, g1.clone()));
        assert_eq!(bt.n_grad.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn a_view_reports_the_block_dimension() {
        let t = coupled();
        let mut ev = Evaluator::new(&t);
        assert_eq!(ev.dim(), 6);
        ev.set_view(Some(View {
            indices: vec![2, 4],
            base: vec![0.0; 6],
        }));
        assert_eq!(ev.dim(), 2);
        assert!(ev.eval(&[1.0, 2.0, 3.0]).is_err(), "wrong block size");
    }
}
