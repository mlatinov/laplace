//! The built-in kernel context and function library, as one registry: what
//! each built-in is called, its documentation, its type rule and its RNG use.
//!
//! Everything a `.lapkernel` can name without declaring it lives here:
//! context values (`q`, `lp`, `grad`, `dim`, `metric`), context functions
//! (`log_density`, `log_density_grad`, `leapfrog`), the terminal statements
//! (`propose`, `move`), and the small function library. Each [`Entry`]
//! carries its type rule (used by the checker, 2.2) and its RNG use,
//! documented because the Phase 2 equivalence tests replay the random
//! stream draw for draw. The implementations live in laplace-sample
//! (`lapkernel::natives`), keyed by the same names; a test there checks the
//! two lists agree.
//!
//! Adding a built-in is one `entries.push(...)` in [`Registry::standard`]
//! plus its implementation in `natives`.
//! Candidates recorded for after v1 (`docs/sampler/open-questions.md`):
//! Hessian, Hessian-vector product, `is_warmup`, `iteration`.
//!
//! Elementwise arithmetic (`+ - * / .* ./ ^`, unary `-`) is built into the
//! interpreter, not this table.

use super::types::{Size, Ty};

/// Type rule: argument types, their values when they are compile-time
/// integer constants, and `dim` when known.
pub type SigFn = fn(&[Ty], &[Option<i64>], Option<usize>) -> Result<Ty, String>;

pub enum Kind {
    /// A context value such as `q`.
    Value { ty: fn(Option<usize>) -> Ty },
    /// A function. Terminal functions (`propose`, `move`) end a step.
    Function { sig: SigFn, terminal: bool },
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
            kind: Kind::Value { ty: dim_vector },
        });
        entries.push(Entry {
            name: "lp",
            doc: "log density at q (propto, jacobian)",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: |_| Ty::Real },
        });
        entries.push(Entry {
            name: "grad",
            doc: "gradient of the log density at q, vector[dim]",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: dim_vector },
        });
        entries.push(Entry {
            name: "dim",
            doc: "number of unconstrained coordinates the kernel updates",
            rng: "none",
            needs_chain: false,
            kind: Kind::Value { ty: |_| Ty::Int },
        });
        entries.push(Entry {
            name: "metric",
            doc: "diagonal inverse metric, vector[dim]: the kernel's @adapt(metric) tuning value if it has one, else all ones",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: dim_vector },
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
                terminal: false,
            },
        });
        let unary: [(&'static str, &'static str); 4] = [
            ("exp", "exp(x): e^x, elementwise for vectors"),
            ("log", "log(x): natural logarithm, elementwise for vectors"),
            ("sqrt", "sqrt(x): square root, elementwise for vectors"),
            ("square", "square(x): x * x, elementwise for vectors"),
        ];
        for (name, doc) in unary {
            entries.push(Entry {
                name,
                doc,
                rng: "none",
                needs_chain: false,
                kind: Kind::Function {
                    sig: unary_math_sig,
                    terminal: false,
                },
            });
        }

        Registry { entries }
    }
}

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
