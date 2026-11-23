//! Chain state and the cached evaluator every kernel goes through.

use crate::target::Target;

/// Where a chain is. Invariant: `lp` and `grad` are the target's values at
/// `q`, under the target as it is *now*. Kernels keep it true; [`Evaluator`]
/// is how they get values that satisfy it.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainState {
    pub q: Vec<f64>,
    pub lp: f64,
    pub grad: Vec<f64>,
}

impl ChainState {
    /// Evaluate the target at `q` and build a consistent state.
    pub fn at(q: Vec<f64>, ev: &mut Evaluator) -> anyhow::Result<Self> {
        let (lp, grad) = ev.eval(&q)?;
        Ok(Self { q, lp, grad })
    }
}

/// One cached evaluation. The key is the exact bit pattern of the position,
/// so a hit can only ever return the value computed at that very point.
struct Entry {
    key: Vec<u64>,
    lp: f64,
    grad: Vec<f64>,
}

/// Evaluates a [`Target`] with a small cache keyed by position, and counts
/// real gradient evaluations (cache hits are free and not counted).
///
/// The cache never returns a stale value for a *moved point*: keys are exact
/// bit patterns (1 ulp apart is a miss, and so is `-0.0` vs `0.0`). It cannot
/// see the *target* change; whoever changes the target must call
/// [`Evaluator::invalidate`]. Phase 3's block targets do that whenever another
/// block moves, which is the bug this API is shaped to prevent.
pub struct Evaluator<'t> {
    target: &'t dyn Target,
    cache: Vec<Entry>, // most recent last
    capacity: usize,
    n_grad: u64,
    hits: u64,
    view: Option<View>,
}

/// A block's view of the full target (Phase 3.1): positions passed to
/// [`Evaluator::eval`] are the block's coordinates, placed at `indices` into
/// `base` (the full position, other blocks at their current values);
/// gradients come back restricted to `indices`.
///
/// The cache is keyed by the *full* position, so once another block moves,
/// the same block point is a different key: a block can never be handed a
/// value computed while the other blocks were somewhere else.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub indices: Vec<usize>,
    pub base: Vec<f64>,
}

impl<'t> Evaluator<'t> {
    /// Cache size used by the runner. Small on purpose: kernels revisit the
    /// current point and the last proposal, nothing older.
    pub const DEFAULT_CAPACITY: usize = 4;

    pub fn new(target: &'t dyn Target) -> Self {
        Self::with_capacity(target, Self::DEFAULT_CAPACITY)
    }

    pub fn with_capacity(target: &'t dyn Target, capacity: usize) -> Self {
        Self {
            target,
            cache: Vec::with_capacity(capacity),
            capacity,
            n_grad: 0,
            hits: 0,
            view: None,
        }
    }

    /// Evaluate through a block view from now on (`None`: the full target).
    pub fn set_view(&mut self, view: Option<View>) {
        self.view = view;
    }

    pub fn view(&self) -> Option<&View> {
        self.view.as_ref()
    }

    /// The dimension kernels see: the block's under a view.
    pub fn dim(&self) -> usize {
        match &self.view {
            Some(v) => v.indices.len(),
            None => self.target.dim(),
        }
    }

    pub fn target(&self) -> &'t dyn Target {
        self.target
    }

    /// Log density and gradient at `q`.
    pub fn eval(&mut self, q: &[f64]) -> anyhow::Result<(f64, Vec<f64>)> {
        let Some(view) = &self.view else {
            return self.eval_full(q);
        };
        anyhow::ensure!(
            q.len() == view.indices.len(),
            "block position has {} values, the block has {}",
            q.len(),
            view.indices.len()
        );
        let mut full = view.base.clone();
        for (i, &j) in view.indices.iter().enumerate() {
            full[j] = q[i];
        }
        let indices = view.indices.clone();
        let (lp, g) = self.eval_full(&full)?;
        Ok((lp, indices.iter().map(|&j| g[j]).collect()))
    }

    /// Log density and full gradient at a full position, ignoring any view.
    pub fn eval_full(&mut self, q: &[f64]) -> anyhow::Result<(f64, Vec<f64>)> {
        let key: Vec<u64> = q.iter().map(|x| x.to_bits()).collect();
        if let Some(pos) = self.cache.iter().position(|e| e.key == key) {
            self.hits += 1;
            let e = self.cache.remove(pos);
            let out = (e.lp, e.grad.clone());
            self.cache.push(e);
            return Ok(out);
        }
        let mut grad = vec![0.0; q.len()];
        let lp = self.target.log_density_gradient(q, &mut grad)?;
        self.n_grad += 1;
        if self.capacity > 0 {
            if self.cache.len() == self.capacity {
                self.cache.remove(0);
            }
            self.cache.push(Entry {
                key,
                lp,
                grad: grad.clone(),
            });
        }
        Ok((lp, grad))
    }

    /// Forget every cached value. Required whenever the target itself
    /// changes (e.g. another block moved).
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    /// Real gradient evaluations so far (what `n_grad__` reports).
    pub fn n_grad(&self) -> u64 {
        self.n_grad
    }

    pub fn cache_hits(&self) -> u64 {
        self.hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// `lp = -0.5 * shift * |q|^2`, where `shift` can change under the
    /// evaluator's feet, like a block target whose other blocks moved.
    struct Mutable {
        shift: AtomicU64,
    }
    impl Mutable {
        fn set(&self, v: f64) {
            self.shift.store(v.to_bits(), Ordering::SeqCst)
        }
        fn get(&self) -> f64 {
            f64::from_bits(self.shift.load(Ordering::SeqCst))
        }
    }
    impl Target for Mutable {
        fn dim(&self) -> usize {
            2
        }
        fn log_density_gradient(&self, q: &[f64], g: &mut [f64]) -> anyhow::Result<f64> {
            let s = self.get();
            g[0] = -s * q[0];
            g[1] = -s * q[1];
            Ok(-0.5 * s * (q[0] * q[0] + q[1] * q[1]))
        }
    }
    fn mutable(s: f64) -> Mutable {
        let m = Mutable {
            shift: AtomicU64::new(0),
        };
        m.set(s);
        m
    }

    #[test]
    fn same_point_is_a_free_hit() {
        let t = mutable(1.0);
        let mut ev = Evaluator::new(&t);
        let a = ev.eval(&[1.0, 2.0]).unwrap();
        let b = ev.eval(&[1.0, 2.0]).unwrap();
        assert_eq!(a, b);
        assert_eq!(ev.n_grad(), 1);
        assert_eq!(ev.cache_hits(), 1);
    }

    #[test]
    fn a_point_one_ulp_away_is_a_miss() {
        let t = mutable(1.0);
        let mut ev = Evaluator::new(&t);
        ev.eval(&[1.0, 2.0]).unwrap();
        let next = f64::from_bits(2.0f64.to_bits() + 1);
        let (lp, g) = ev.eval(&[1.0, next]).unwrap();
        assert_eq!(ev.n_grad(), 2);
        assert_eq!(g[1], -next);
        assert_eq!(lp, -0.5 * (1.0 + next * next));
    }

    #[test]
    fn signed_zero_is_a_different_key() {
        let t = mutable(1.0);
        let mut ev = Evaluator::new(&t);
        ev.eval(&[0.0, 0.0]).unwrap();
        ev.eval(&[-0.0, 0.0]).unwrap();
        assert_eq!(ev.n_grad(), 2);
    }

    #[test]
    fn stale_value_is_never_returned_after_invalidate() {
        let t = mutable(1.0);
        let mut ev = Evaluator::new(&t);
        let (lp1, g1) = ev.eval(&[1.0, 1.0]).unwrap();
        t.set(3.0); // the target changed under the evaluator
        ev.invalidate();
        let (lp2, g2) = ev.eval(&[1.0, 1.0]).unwrap();
        assert_eq!(lp1, -1.0);
        assert_eq!(lp2, -3.0);
        assert_eq!(g1, vec![-1.0, -1.0]);
        assert_eq!(g2, vec![-3.0, -3.0]);
        assert_eq!(ev.n_grad(), 2);
    }

    #[test]
    fn mutating_the_callers_vector_does_not_alias_the_key() {
        let t = mutable(1.0);
        let mut ev = Evaluator::new(&t);
        let mut q = vec![1.0, 1.0];
        ev.eval(&q).unwrap();
        q[0] = 5.0;
        let (lp, _) = ev.eval(&q).unwrap();
        assert_eq!(lp, -0.5 * 26.0);
        assert_eq!(ev.n_grad(), 2);
    }

    #[test]
    fn eviction_is_least_recently_used() {
        let t = mutable(1.0);
        let mut ev = Evaluator::with_capacity(&t, 2);
        ev.eval(&[1.0, 0.0]).unwrap(); // A
        ev.eval(&[2.0, 0.0]).unwrap(); // B
        ev.eval(&[1.0, 0.0]).unwrap(); // A again: hit, now most recent
        ev.eval(&[3.0, 0.0]).unwrap(); // C evicts B
        assert_eq!(ev.n_grad(), 3);
        ev.eval(&[1.0, 0.0]).unwrap(); // A still cached
        assert_eq!(ev.n_grad(), 3);
        ev.eval(&[2.0, 0.0]).unwrap(); // B was evicted
        assert_eq!(ev.n_grad(), 4);
    }

    #[test]
    fn zero_capacity_never_caches() {
        let t = mutable(1.0);
        let mut ev = Evaluator::with_capacity(&t, 0);
        ev.eval(&[1.0, 0.0]).unwrap();
        ev.eval(&[1.0, 0.0]).unwrap();
        assert_eq!(ev.n_grad(), 2);
    }

    #[test]
    fn returned_gradient_is_a_copy() {
        let t = mutable(1.0);
        let mut ev = Evaluator::new(&t);
        let (_, mut g) = ev.eval(&[1.0, 1.0]).unwrap();
        g[0] = 99.0;
        let (_, g2) = ev.eval(&[1.0, 1.0]).unwrap();
        assert_eq!(g2[0], -1.0);
    }
}
