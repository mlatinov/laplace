//! The implementations of the built-in kernel context and function library:
//! what each built-in in `laplace_sampler_syntax::lapkernel::builtins` does
//! when a chain runs, keyed by the same name.
//!
//! The registry there owns each built-in's name, documentation, type rule
//! and RNG use (the checker and the editor need those without a runtime);
//! this table owns its value or function. A test below checks that every
//! non-terminal registry entry has exactly one implementation here, of the
//! same kind, and that nothing here is missing from the registry.
//! Terminal statements (`propose`, `move`) have none: the interpreter
//! implements them.

use nalgebra::DVector;
use rand::RngExt;
use rand_distr::StandardNormal;

use super::types::Value;
use crate::kernel::ChainRng;
use crate::state::{ChainState, Evaluator};

/// What a built-in may touch when it runs. Outside a running chain (tuning
/// defaults are evaluated before any chain exists) the chain parts are
/// `None`, and the checker only lets chain-free entries appear there.
pub struct Ctx<'a, 't> {
    pub dim: usize,
    pub state: Option<&'a ChainState>,
    pub ev: Option<&'a mut Evaluator<'t>>,
    pub rng: Option<&'a mut ChainRng>,
    pub metric: Option<&'a DVector<f64>>,
}

impl<'a, 't> Ctx<'a, 't> {
    fn state(&self) -> Result<&ChainState, String> {
        self.state
            .ok_or_else(|| "needs a running chain".to_string())
    }
    fn ev(&mut self) -> Result<&mut Evaluator<'t>, String> {
        self.ev
            .as_deref_mut()
            .ok_or_else(|| "needs a running chain".to_string())
    }
    fn rng(&mut self) -> Result<&mut ChainRng, String> {
        self.rng
            .as_deref_mut()
            .ok_or_else(|| "needs a running chain".to_string())
    }
}

pub type CallFn = for<'a, 't> fn(&mut Ctx<'a, 't>, &[Value]) -> Result<Value, String>;
pub type GetFn = for<'a, 't> fn(&Ctx<'a, 't>) -> Result<Value, String>;

/// One built-in's implementation.
#[derive(Clone, Copy)]
pub enum Native {
    /// A context value such as `q`.
    Value(GetFn),
    /// A (non-terminal) function.
    Function(CallFn),
}

pub struct Natives {
    entries: Vec<(&'static str, Native)>,
}

impl Natives {
    pub fn get(&self, name: &str) -> Option<Native> {
        self.entries
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, f)| *f)
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.iter().map(|(n, _)| *n)
    }

    /// The v1 implementations, one per non-terminal entry of
    /// `Registry::standard()`.
    pub fn standard() -> Natives {
        let mut entries: Vec<(&'static str, Native)> = vec![];

        // ---- context values ----------------------------------------------
        entries.push((
            "q",
            Native::Value(|c| Ok(Value::Vector(DVector::from_column_slice(&c.state()?.q)))),
        ));
        entries.push(("lp", Native::Value(|c| Ok(Value::Real(c.state()?.lp)))));
        entries.push((
            "grad",
            // The interpreter computes it before the step of a kernel
            // that reads `grad` (Checked::uses_grad).
            Native::Value(|c| {
                let g = c
                    .state()?
                    .grad
                    .as_deref()
                    .ok_or("the gradient at q was not computed")?;
                Ok(Value::Vector(DVector::from_column_slice(g)))
            }),
        ));
        entries.push(("dim", Native::Value(|c| Ok(Value::Int(c.dim as i64)))));
        entries.push((
            "metric",
            Native::Value(|c| {
                Ok(Value::Vector(match c.metric {
                    Some(m) => m.clone(),
                    None => DVector::from_element(c.dim, 1.0),
                }))
            }),
        ));

        // ---- context functions -------------------------------------------
        entries.push((
            "log_density",
            Native::Function(|c, a| {
                let x = vec_arg(&a[0])?;
                let (lp, _) = c.ev()?.eval(x.as_slice()).map_err(|e| e.to_string())?;
                Ok(Value::Real(lp))
            }),
        ));
        entries.push((
            "log_density_grad",
            Native::Function(|c, a| {
                let x = vec_arg(&a[0])?;
                let (lp, g) = c.ev()?.eval(x.as_slice()).map_err(|e| e.to_string())?;
                Ok(Value::Tuple(vec![
                    Value::Real(lp),
                    Value::Vector(DVector::from_vec(g)),
                ]))
            }),
        ));
        entries.push(("leapfrog", Native::Function(leapfrog)));

        // ---- random numbers ----------------------------------------------
        entries.push((
            "std_normal_rng",
            Native::Function(|c, a| {
                let rng = c.rng()?;
                match a {
                    [] => Ok(Value::Real(rng.sample(StandardNormal))),
                    [n] => {
                        let n = size_arg(n)?;
                        Ok(Value::Vector(DVector::from_iterator(
                            n,
                            (0..n).map(|_| rng.sample::<f64, _>(StandardNormal)),
                        )))
                    }
                    _ => unreachable!(),
                }
            }),
        ));
        entries.push((
            "normal_rng",
            Native::Function(|c, a| {
                let n = broadcast_len(a)?;
                let rng = c.rng()?;
                match n {
                    None => {
                        let z: f64 = rng.sample(StandardNormal);
                        Ok(Value::Real(elem(&a[0], 0) + elem(&a[1], 0) * z))
                    }
                    Some(n) => Ok(Value::Vector(DVector::from_iterator(
                        n,
                        (0..n).map(|i| {
                            let z: f64 = rng.sample(StandardNormal);
                            elem(&a[0], i) + elem(&a[1], i) * z
                        }),
                    ))),
                }
            }),
        ));
        entries.push((
            "uniform_rng",
            Native::Function(|c, a| {
                let (lo, hi) = (real_arg(&a[0])?, real_arg(&a[1])?);
                let u: f64 = c.rng()?.random();
                Ok(Value::Real(lo + (hi - lo) * u))
            }),
        ));

        // ---- densities -----------------------------------------------------
        entries.push((
            "normal_lpdf",
            Native::Function(|_, a| {
                let n = broadcast_len(a)?.unwrap_or(1);
                let mut s = 0.0;
                for i in 0..n {
                    let sigma = elem(&a[2], i);
                    if sigma.is_nan() || sigma <= 0.0 {
                        return Err(format!("normal_lpdf: sigma must be positive, got {sigma}"));
                    }
                    let z = (elem(&a[0], i) - elem(&a[1], i)) / sigma;
                    s += -0.5 * z * z - sigma.ln() - HALF_LN_2PI;
                }
                Ok(Value::Real(s))
            }),
        ));
        entries.push((
            "std_normal_lpdf",
            Native::Function(|_, a| {
                let n = broadcast_len(a)?.unwrap_or(1);
                Ok(Value::Real(
                    (0..n)
                        .map(|i| {
                            let y = elem(&a[0], i);
                            -0.5 * y * y - HALF_LN_2PI
                        })
                        .sum(),
                ))
            }),
        ));

        // ---- math ------------------------------------------------------------
        entries.push((
            "dot_product",
            Native::Function(|_, a| {
                let (x, y) = (vec_arg(&a[0])?, vec_arg(&a[1])?);
                same_len("dot_product", x, y)?;
                Ok(Value::Real(
                    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum(),
                ))
            }),
        ));
        entries.push((
            "sum",
            Native::Function(|_, a| Ok(Value::Real(vec_arg(&a[0])?.iter().sum()))),
        ));
        entries.push((
            "rep_vector",
            Native::Function(|_, a| {
                let x = real_arg(&a[0])?;
                Ok(Value::Vector(DVector::from_element(size_arg(&a[1])?, x)))
            }),
        ));
        let unary: [(&'static str, CallFn); 4] = [
            ("exp", |_, a| unary_math(&a[0], f64::exp)),
            ("log", |_, a| unary_math(&a[0], f64::ln)),
            ("sqrt", |_, a| unary_math(&a[0], f64::sqrt)),
            ("square", |_, a| unary_math(&a[0], |x| x * x)),
        ];
        for (name, call) in unary {
            entries.push((name, Native::Function(call)));
        }

        Natives { entries }
    }
}

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

fn unary_math(v: &Value, f: fn(f64) -> f64) -> Result<Value, String> {
    match v {
        Value::Vector(x) => Ok(Value::Vector(x.map(f))),
        other => Ok(Value::Real(f(real_arg(other)?))),
    }
}

fn broadcast_len(a: &[Value]) -> Result<Option<usize>, String> {
    let mut n: Option<usize> = None;
    for v in a {
        if let Value::Vector(x) = v {
            match n {
                Some(m) if m != x.len() => {
                    return Err(format!("vector sizes {m} and {} differ", x.len()))
                }
                _ => n = Some(x.len()),
            }
        }
    }
    Ok(n)
}

/// Element `i` of a scalar-or-vector argument (scalars broadcast).
fn elem(v: &Value, i: usize) -> f64 {
    match v {
        Value::Vector(x) => x[i],
        other => other.as_real().unwrap_or(f64::NAN),
    }
}

fn real_arg(v: &Value) -> Result<f64, String> {
    v.as_real()
        .ok_or_else(|| format!("expected a real, got {}", v.type_name()))
}

fn vec_arg(v: &Value) -> Result<&DVector<f64>, String> {
    v.as_vector()
        .ok_or_else(|| format!("expected a vector, got {}", v.type_name()))
}

fn size_arg(v: &Value) -> Result<usize, String> {
    match v.as_int() {
        Some(n) if n >= 0 => Ok(n as usize),
        Some(n) => Err(format!("size must be non-negative, got {n}")),
        None => Err(format!("expected an int size, got {}", v.type_name())),
    }
}

fn same_len(f: &str, x: &DVector<f64>, y: &DVector<f64>) -> Result<(), String> {
    if x.len() == y.len() {
        Ok(())
    } else {
        Err(format!("{f}: sizes {} and {} differ", x.len(), y.len()))
    }
}

fn leapfrog(c: &mut Ctx<'_, '_>, a: &[Value]) -> Result<Value, String> {
    let mut q = vec_arg(&a[0])?.clone();
    let mut p = vec_arg(&a[1])?.clone();
    let eps = real_arg(&a[2])?;
    let steps = a[3].as_int().ok_or("leapfrog: steps must be an int")?;
    let m = vec_arg(&a[4])?.clone();
    same_len("leapfrog", &q, &p)?;
    same_len("leapfrog", &q, &m)?;
    let ev = c.ev()?;
    let (mut lp, g) = ev.eval(q.as_slice()).map_err(|e| e.to_string())?;
    let mut g = DVector::from_vec(g);
    for _ in 0..steps.max(0) {
        for i in 0..q.len() {
            p[i] += 0.5 * eps * g[i];
            q[i] += eps * m[i] * p[i];
        }
        let (lp_new, g_new) = ev.eval(q.as_slice()).map_err(|e| e.to_string())?;
        lp = lp_new;
        g = DVector::from_vec(g_new);
        if !lp.is_finite() {
            lp = f64::NEG_INFINITY;
            break;
        }
        for i in 0..q.len() {
            p[i] += 0.5 * eps * g[i];
        }
    }
    Ok(Value::Tuple(vec![
        Value::Vector(q),
        Value::Vector(p),
        Value::Real(lp),
        Value::Vector(g),
    ]))
}

#[cfg(test)]
mod tests {
    use super::super::builtins::{Kind, Registry};
    use super::*;

    #[test]
    fn every_registry_entry_has_its_implementation_and_nothing_else() {
        let reg = Registry::standard();
        let nat = Natives::standard();
        for e in reg.entries() {
            match (&e.kind, nat.get(e.name)) {
                (Kind::Value { .. }, Some(Native::Value(_))) => {}
                (
                    Kind::Function {
                        terminal: false, ..
                    },
                    Some(Native::Function(_)),
                ) => {}
                (Kind::Function { terminal: true, .. }, None) => {}
                (_, got) => panic!(
                    "`{}`: registry kind and implementation disagree (implementation present: {})",
                    e.name,
                    got.is_some()
                ),
            }
        }
        for n in nat.names() {
            assert!(
                reg.get(n).is_some(),
                "`{n}` is implemented but not in the registry"
            );
        }
        let mut names: Vec<_> = nat.names().collect();
        names.sort();
        names.dedup();
        assert_eq!(
            names.len(),
            nat.names().count(),
            "a name is implemented twice"
        );
    }
}
