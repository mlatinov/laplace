//! The built-in kernel context and function library, as one registry.
//!
//! Everything a `.lapkernel` can name without declaring it lives here:
//! context values (`q`, `lp`, `grad`, `dim`, `metric`), context functions
//! (`log_density`, `log_density_grad`, `leapfrog`), the terminal statements
//! (`propose`, `move`), and the small function library. Each [`Entry`]
//! carries its type rule (used by the checker, 2.2), its implementation
//! (used by the interpreter, 2.3), and its RNG use, documented because the
//! Phase 2 equivalence tests replay the random stream draw for draw.
//!
//! Adding a built-in is one `entries.push(...)` in [`Registry::standard`].
//! Candidates recorded for after v1 (`docs/sampler/open-questions.md`):
//! Hessian, Hessian-vector product, `is_warmup`, `iteration`.
//!
//! Elementwise arithmetic (`+ - * / .* ./ ^`, unary `-`) is built into the
//! interpreter, not this table.

use nalgebra::DVector;
use rand::RngExt;
use rand_distr::StandardNormal;

use super::types::{Size, Ty, Value};
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

/// Type rule: argument types, their values when they are compile-time
/// integer constants, and `dim` when known.
pub type SigFn = fn(&[Ty], &[Option<i64>], Option<usize>) -> Result<Ty, String>;
pub type CallFn = for<'a, 't> fn(&mut Ctx<'a, 't>, &[Value]) -> Result<Value, String>;
pub type GetFn = for<'a, 't> fn(&Ctx<'a, 't>) -> Result<Value, String>;

pub enum Kind {
    /// A context value such as `q`.
    Value {
        ty: fn(Option<usize>) -> Ty,
        get: GetFn,
    },
    /// A function. Terminal functions (`propose`, `move`) end a step; the
    /// interpreter implements them, so `call` is `None`.
    Function {
        sig: SigFn,
        call: Option<CallFn>,
        terminal: bool,
    },
}

pub struct Entry {
    pub name: &'static str,
    pub doc: &'static str,
    /// Random numbers drawn from the chain stream per call ("none" if none).
    pub rng: &'static str,
    /// Needs a running chain (chain state, evaluator or RNG): not allowed in
    /// tuning defaults.
    pub needs_chain: bool,
    pub kind: Kind,
}

pub struct Registry {
    entries: Vec<Entry>,
}

impl Registry {
    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The v1 registry.
    pub fn standard() -> Registry {
        let mut entries = vec![];

        // ---- context values ----------------------------------------------
        entries.push(Entry {
            name: "q",
            doc: "current position (unconstrained), vector[dim]",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value {
                ty: dim_vector,
                get: |c| Ok(Value::Vector(DVector::from_column_slice(&c.state()?.q))),
            },
        });
        entries.push(Entry {
            name: "lp",
            doc: "log density at q (propto, jacobian)",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value {
                ty: |_| Ty::Real,
                get: |c| Ok(Value::Real(c.state()?.lp)),
            },
        });
        entries.push(Entry {
            name: "grad",
            doc: "gradient of the log density at q, vector[dim]",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value {
                ty: dim_vector,
                // The interpreter computes it before the step of a kernel
                // that reads `grad` (Checked::uses_grad).
                get: |c| {
                    let g = c
                        .state()?
                        .grad
                        .as_deref()
                        .ok_or("the gradient at q was not computed")?;
                    Ok(Value::Vector(DVector::from_column_slice(g)))
                },
            },
        });
        entries.push(Entry {
            name: "dim",
            doc: "number of unconstrained coordinates the kernel updates",
            rng: "none",
            needs_chain: false,
            kind: Kind::Value {
                ty: |_| Ty::Int,
                get: |c| Ok(Value::Int(c.dim as i64)),
            },
        });
        entries.push(Entry {
            name: "metric",
            doc: "diagonal inverse metric, vector[dim]: the kernel's @adapt(metric) tuning value if it has one, else all ones",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value {
                ty: dim_vector,
                get: |c| {
                    Ok(Value::Vector(match c.metric {
                        Some(m) => m.clone(),
                        None => DVector::from_element(c.dim, 1.0),
                    }))
                },
            },
        });

        // ---- context functions -------------------------------------------
        entries.push(Entry {
            name: "log_density",
            doc: "log_density(vector x): log density at x (through the chain's cached evaluator)",
            rng: "none",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    expect(a, &[Ty::Vector(dim_size(d))], "log_density(vector[dim] x)")?;
                    Ok(Ty::Real)
                },
                call: Some(|c, a| {
                    let x = vec_arg(&a[0])?;
                    let (lp, _) = c.ev()?.eval(x.as_slice()).map_err(|e| e.to_string())?;
                    Ok(Value::Real(lp))
                }),
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "log_density_grad",
            doc: "log_density_grad(vector x): (log density, gradient) at x, as tuple(real, vector[dim])",
            rng: "none",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    expect(a, &[Ty::Vector(dim_size(d))], "log_density_grad(vector[dim] x)")?;
                    Ok(Ty::Tuple(vec![Ty::Real, Ty::Vector(dim_size(d))]))
                },
                call: Some(|c, a| {
                    let x = vec_arg(&a[0])?;
                    let (lp, g) = c.ev()?.eval(x.as_slice()).map_err(|e| e.to_string())?;
                    Ok(Value::Tuple(vec![Value::Real(lp), Value::Vector(DVector::from_vec(g))]))
                }),
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "leapfrog",
            doc: "leapfrog(vector q, vector p, real eps, int steps, vector metric): `steps` leapfrog steps with \
                  diagonal inverse metric; returns tuple(vector q, vector p, real lp, vector grad). Stops early \
                  (returning lp = -inf) if the density stops being finite",
            rng: "none",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    let v = Ty::Vector(dim_size(d));
                    expect(
                        a,
                        &[v.clone(), v.clone(), Ty::Real, Ty::Int, v.clone()],
                        "leapfrog(vector q, vector p, real eps, int steps, vector metric)",
                    )?;
                    Ok(Ty::Tuple(vec![v.clone(), v.clone(), Ty::Real, v]))
                },
                call: Some(leapfrog),
                terminal: false,
            },
        });

        // ---- terminal statements -----------------------------------------
        entries.push(Entry {
            name: "propose",
            doc: "propose(vector q_new, real log_correction): Metropolis-Hastings step to q_new with \
                  log_alpha = lp(q_new) - lp + log_correction; ends the step",
            rng: "one uniform (the MH accept draw), always",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    expect(a, &[Ty::Vector(dim_size(d)), Ty::Real], "propose(vector[dim] q_new, real log_correction)")?;
                    Ok(Ty::Void)
                },
                call: None,
                terminal: true,
            },
        });
        entries.push(Entry {
            name: "move",
            doc: "move(vector q_new): move to q_new unconditionally; ends the step",
            rng: "none",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    expect(a, &[Ty::Vector(dim_size(d))], "move(vector[dim] q_new)")?;
                    Ok(Ty::Void)
                },
                call: None,
                terminal: true,
            },
        });

        // ---- random numbers ----------------------------------------------
        entries.push(Entry {
            name: "std_normal_rng",
            doc: "std_normal_rng(): real; std_normal_rng(int n): vector[n] of independent N(0, 1)",
            rng: "one normal draw (rand_distr::StandardNormal) per element, in index order",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, k, _| match a {
                    [] => Ok(Ty::Real),
                    [Ty::Int] => Ok(Ty::Vector(const_size(k[0])?)),
                    _ => Err(bad_args("std_normal_rng() or std_normal_rng(int n)", a)),
                },
                call: Some(|c, a| {
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
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "normal_rng",
            doc: "normal_rng(mu, sigma): mu + sigma * z with z ~ N(0, 1); vector if mu or sigma is a vector",
            rng: "one normal draw (rand_distr::StandardNormal) per element, in index order",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, _| {
                    if a.len() != 2 || !a.iter().all(|t| t.is_scalar() || matches!(t, Ty::Vector(_))) {
                        return Err(bad_args("normal_rng(real|vector mu, real|vector sigma)", a));
                    }
                    broadcast_type(a)
                },
                call: Some(|c, a| {
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
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "uniform_rng",
            doc: "uniform_rng(real a, real b): a + (b - a) * u with u uniform on [0, 1)",
            rng: "one uniform (rng.random::<f64>())",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, _| {
                    expect(a, &[Ty::Real, Ty::Real], "uniform_rng(real a, real b)")?;
                    Ok(Ty::Real)
                },
                call: Some(|c, a| {
                    let (lo, hi) = (real_arg(&a[0])?, real_arg(&a[1])?);
                    let u: f64 = c.rng()?.random();
                    Ok(Value::Real(lo + (hi - lo) * u))
                }),
                terminal: false,
            },
        });

        // ---- densities -----------------------------------------------------
        entries.push(Entry {
            name: "normal_lpdf",
            doc: "normal_lpdf(y | mu, sigma): sum of normal log densities, with constants; real or vector arguments",
            rng: "none",
            needs_chain: false,
            kind: Kind::Function {
                sig: |a, _, _| {
                    if a.len() != 3 || !a.iter().all(|t| t.is_scalar() || matches!(t, Ty::Vector(_))) {
                        return Err(bad_args("normal_lpdf(real|vector y | real|vector mu, real|vector sigma)", a));
                    }
                    broadcast_type(a)?;
                    Ok(Ty::Real)
                },
                call: Some(|_, a| {
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
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "std_normal_lpdf",
            doc: "std_normal_lpdf(y): sum of standard normal log densities, with constants",
            rng: "none",
            needs_chain: false,
            kind: Kind::Function {
                sig: |a, _, _| match a {
                    [t] if t.is_scalar() || matches!(t, Ty::Vector(_)) => Ok(Ty::Real),
                    _ => Err(bad_args("std_normal_lpdf(real|vector y)", a)),
                },
                call: Some(|_, a| {
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
                terminal: false,
            },
        });

        // ---- math ------------------------------------------------------------
        entries.push(Entry {
            name: "dot_product",
            doc: "dot_product(vector a, vector b): sum of a[i] * b[i]",
            rng: "none",
            needs_chain: false,
            kind: Kind::Function {
                sig: |a, _, _| match a {
                    [Ty::Vector(x), Ty::Vector(y)] if x.agrees(*y) => Ok(Ty::Real),
                    [Ty::Vector(x), Ty::Vector(y)] => {
                        Err(format!("dot_product: sizes {x} and {y} differ"))
                    }
                    _ => Err(bad_args("dot_product(vector a, vector b)", a)),
                },
                call: Some(|_, a| {
                    let (x, y) = (vec_arg(&a[0])?, vec_arg(&a[1])?);
                    same_len("dot_product", x, y)?;
                    Ok(Value::Real(
                        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum(),
                    ))
                }),
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "sum",
            doc: "sum(vector v): sum of the elements",
            rng: "none",
            needs_chain: false,
            kind: Kind::Function {
                sig: |a, _, _| match a {
                    [Ty::Vector(_)] => Ok(Ty::Real),
                    _ => Err(bad_args("sum(vector v)", a)),
                },
                call: Some(|_, a| Ok(Value::Real(vec_arg(&a[0])?.iter().sum()))),
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "rep_vector",
            doc: "rep_vector(real x, int n): vector[n] with every element x",
            rng: "none",
            needs_chain: false,
            kind: Kind::Function {
                sig: |a, k, _| match a {
                    [x, Ty::Int] if x.is_scalar() => Ok(Ty::Vector(const_size(k[1])?)),
                    _ => Err(bad_args("rep_vector(real x, int n)", a)),
                },
                call: Some(|_, a| {
                    let x = real_arg(&a[0])?;
                    Ok(Value::Vector(DVector::from_element(size_arg(&a[1])?, x)))
                }),
                terminal: false,
            },
        });
        let unary: [(&'static str, &'static str, CallFn); 4] = [
            ("exp", "exp(x): e^x, elementwise for vectors", |_, a| {
                unary_math(&a[0], f64::exp)
            }),
            (
                "log",
                "log(x): natural logarithm, elementwise for vectors",
                |_, a| unary_math(&a[0], f64::ln),
            ),
            (
                "sqrt",
                "sqrt(x): square root, elementwise for vectors",
                |_, a| unary_math(&a[0], f64::sqrt),
            ),
            (
                "square",
                "square(x): x * x, elementwise for vectors",
                |_, a| unary_math(&a[0], |x| x * x),
            ),
        ];
        for (name, doc, call) in unary {
            entries.push(Entry {
                name,
                doc,
                rng: "none",
                needs_chain: false,
                kind: Kind::Function {
                    sig: unary_math_sig,
                    call: Some(call),
                    terminal: false,
                },
            });
        }

        Registry { entries }
    }
}

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

fn dim_size(d: Option<usize>) -> Size {
    d.map_or(Size::Unknown, Size::Known)
}

fn dim_vector(d: Option<usize>) -> Ty {
    Ty::Vector(dim_size(d))
}

fn const_size(k: Option<i64>) -> Result<Size, String> {
    match k {
        Some(n) if n < 0 => Err(format!("size must be non-negative, got {n}")),
        Some(n) => Ok(Size::Known(n as usize)),
        None => Ok(Size::Unknown),
    }
}

fn bad_args(sig: &str, got: &[Ty]) -> String {
    let got: Vec<String> = got.iter().map(|t| t.to_string()).collect();
    format!("expected {sig}, got ({})", got.join(", "))
}

/// Exact argument list, with `int` promoting to `real` and agreeing sizes.
fn expect(got: &[Ty], want: &[Ty], sig: &str) -> Result<(), String> {
    if got.len() == want.len() && want.iter().zip(got).all(|(w, g)| w.accepts(g)) {
        Ok(())
    } else {
        Err(bad_args(sig, got))
    }
}

fn unary_math_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [t] if t.is_scalar() => Ok(Ty::Real),
        [Ty::Vector(n)] => Ok(Ty::Vector(*n)),
        _ => Err(bad_args("f(real) or f(vector)", a)),
    }
}

fn unary_math(v: &Value, f: fn(f64) -> f64) -> Result<Value, String> {
    match v {
        Value::Vector(x) => Ok(Value::Vector(x.map(f))),
        other => Ok(Value::Real(f(real_arg(other)?))),
    }
}

/// Real for all-scalar arguments; vector of the common size otherwise.
fn broadcast_type(a: &[Ty]) -> Result<Ty, String> {
    let mut size: Option<Size> = None;
    for t in a {
        if let Ty::Vector(n) = t {
            match size {
                Some(s) if !s.agrees(*n) => return Err(format!("vector sizes {s} and {n} differ")),
                Some(s) => size = Some(s.meet(*n)),
                None => size = Some(*n),
            }
        }
    }
    Ok(size.map_or(Ty::Real, Ty::Vector))
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
