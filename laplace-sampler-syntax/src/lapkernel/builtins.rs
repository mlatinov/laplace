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
    /// How it is written, with its type: `vector[dim] q` for a value,
    /// `real log_density(vector[dim] x)` for a function, one per overload.
    /// A test checks each against the type rule.
    pub signatures: &'static [&'static str],
    /// What it is or does (the signature is not repeated here).
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

    /// A name a kernel may not declare as a function or variable: a context
    /// value or function, a terminal or an RNG. Library functions (pure, no
    /// chain needed) are not reserved: a kernel's own function or variable
    /// of that name wins, so adding to the library never breaks a kernel.
    pub fn reserved(&self, name: &str) -> bool {
        self.get(name).is_some_and(|e| {
            e.needs_chain
                || matches!(
                    e.kind,
                    Kind::Value { .. } | Kind::Function { terminal: true, .. }
                )
        })
    }

    /// The v1 registry.
    pub fn standard() -> Registry {
        let mut entries = vec![];

        // ---- context values ----------------------------------------------
        entries.push(Entry {
            name: "q",
            signatures: &["vector[dim] q"],
            doc: "current position (unconstrained), vector[dim]",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: dim_vector },
        });
        entries.push(Entry {
            name: "lp",
            signatures: &["real lp"],
            doc: "log density at q (propto, jacobian)",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: |_| Ty::Real },
        });
        entries.push(Entry {
            name: "grad",
            signatures: &["vector[dim] grad"],
            doc: "gradient of the log density at q, vector[dim]",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: dim_vector },
        });
        entries.push(Entry {
            name: "dim",
            signatures: &["int dim"],
            doc: "number of unconstrained coordinates the kernel updates",
            rng: "none",
            needs_chain: false,
            kind: Kind::Value { ty: |_| Ty::Int },
        });
        entries.push(Entry {
            name: "metric",
            signatures: &["vector[dim] metric"],
            doc: "diagonal inverse metric, vector[dim]: the kernel's @adapt(metric) tuning value if it has one, else all ones",
            rng: "none",
            needs_chain: true,
            kind: Kind::Value { ty: dim_vector },
        });

        // ---- context functions -------------------------------------------
        entries.push(Entry {
            name: "log_density",
            signatures: &["real log_density(vector[dim] x)"],
            doc: "log density at x (through the chain's cached evaluator)",
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
            signatures: &["tuple(real, vector[dim]) log_density_grad(vector[dim] x)"],
            doc: "(log density, gradient) at x",
            rng: "none",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    expect(
                        a,
                        &[Ty::Vector(dim_size(d))],
                        "log_density_grad(vector[dim] x)",
                    )?;
                    Ok(Ty::Tuple(vec![Ty::Real, Ty::Vector(dim_size(d))]))
                },
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "leapfrog",
            signatures: &["tuple(vector, vector, real, vector) leapfrog(vector q, vector p, real eps, int steps, vector metric)"],
            doc: "`steps` leapfrog steps with diagonal inverse metric; returns (q, p, lp, grad). Stops early \
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
            signatures: &["void propose(vector[dim] q_new, real log_correction)"],
            doc: PROPOSE_DOC,
            rng: "one uniform (the MH accept draw), always",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, d| {
                    expect(
                        a,
                        &[Ty::Vector(dim_size(d)), Ty::Real],
                        "propose(vector[dim] q_new, real log_correction)",
                    )?;
                    Ok(Ty::Void)
                },
                terminal: true,
            },
        });
        entries.push(Entry {
            name: "move",
            signatures: &["void move(vector[dim] q_new)"],
            doc: MOVE_DOC,
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
            signatures: &["real std_normal_rng()", "vector[n] std_normal_rng(int n)"],
            doc: "independent N(0, 1) draws: one real, or a vector of n",
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
            signatures: &[
                "real normal_rng(real mu, real sigma)",
                "vector normal_rng(vector mu, real|vector sigma)",
                "vector normal_rng(real mu, vector sigma)",
            ],
            doc: "mu + sigma * z with z ~ N(0, 1); a vector if mu or sigma is a vector",
            rng: "one normal draw (rand_distr::StandardNormal) per element, in index order",
            needs_chain: true,
            kind: Kind::Function {
                sig: |a, _, _| {
                    if a.len() != 2
                        || !a
                            .iter()
                            .all(|t| t.is_scalar() || matches!(t, Ty::Vector(_)))
                    {
                        return Err(bad_args("normal_rng(real|vector mu, real|vector sigma)", a));
                    }
                    broadcast_type(a)
                },
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "uniform_rng",
            signatures: &["real uniform_rng(real a, real b)"],
            doc: "a + (b - a) * u with u uniform on [0, 1)",
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
            signatures: &["real normal_lpdf(real|vector y | real|vector mu, real|vector sigma)"],
            doc: "sum of normal log densities, with constants",
            rng: "none",
            needs_chain: false,
            kind: Kind::Function {
                sig: |a, _, _| {
                    if a.len() != 3
                        || !a
                            .iter()
                            .all(|t| t.is_scalar() || matches!(t, Ty::Vector(_)))
                    {
                        return Err(bad_args(
                            "normal_lpdf(real|vector y | real|vector mu, real|vector sigma)",
                            a,
                        ));
                    }
                    broadcast_type(a)?;
                    Ok(Ty::Real)
                },
                terminal: false,
            },
        });
        entries.push(Entry {
            name: "std_normal_lpdf",
            signatures: &["real std_normal_lpdf(real|vector y)"],
            doc: "sum of standard normal log densities, with constants",
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
            signatures: &["real dot_product(vector a, vector b)"],
            doc: "sum of a[i] * b[i]",
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
            signatures: &["real sum(vector v)"],
            doc: "sum of the elements",
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
            signatures: &["vector[n] rep_vector(real x, int n)"],
            doc: "vector[n] with every element x",
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
        // Pure functions: no chain, no random numbers. Where Stan has a
        // numerically safe composite (`log1p_exp(x)` for `log(1 + exp(x))`)
        // that is the one offered; docs/sampler/kernel-functions.md lists
        // them all with the formulas the implementations use.
        type Pure = (&'static str, &'static [&'static str], &'static str, SigFn);
        let pure: &[Pure] = &[
            // ---- elementwise, one argument --------------------------------
            ("exp", &["real exp(real x)", "vector exp(vector x)"],
                "e^x, elementwise for vectors", unary_math_sig),
            ("log", &["real log(real x)", "vector log(vector x)"],
                "natural logarithm, elementwise for vectors. NaN for x < 0, -inf at 0", unary_math_sig),
            ("sqrt", &["real sqrt(real x)", "vector sqrt(vector x)"],
                "square root, elementwise for vectors. NaN for x < 0", unary_math_sig),
            ("square", &["real square(real x)", "vector square(vector x)"],
                "x * x, elementwise for vectors", unary_math_sig),
            ("abs", &["real abs(real x)", "vector abs(vector x)"],
                "absolute value, elementwise for vectors", unary_math_sig),
            ("inv", &["real inv(real x)", "vector inv(vector x)"],
                "1 / x, elementwise for vectors", unary_math_sig),
            ("inv_sqrt", &["real inv_sqrt(real x)", "vector inv_sqrt(vector x)"],
                "1 / sqrt(x), elementwise for vectors", unary_math_sig),
            ("inv_square", &["real inv_square(real x)", "vector inv_square(vector x)"],
                "1 / x^2, elementwise for vectors", unary_math_sig),
            ("expm1", &["real expm1(real x)", "vector expm1(vector x)"],
                "exp(x) - 1, accurate for x near 0, elementwise for vectors", unary_math_sig),
            ("log1p", &["real log1p(real x)", "vector log1p(vector x)"],
                "log(1 + x), accurate for x near 0, elementwise for vectors. NaN for x < -1", unary_math_sig),
            ("log1m", &["real log1m(real x)", "vector log1m(vector x)"],
                "log(1 - x), accurate for x near 0, elementwise for vectors. NaN for x > 1", unary_math_sig),
            ("log1p_exp", &["real log1p_exp(real x)", "vector log1p_exp(vector x)"],
                "log(1 + exp(x)) (softplus) without overflow: x + log1p(exp(-x)) for x > 0, \
                 elementwise for vectors", unary_math_sig),
            ("log1m_exp", &["real log1m_exp(real x)", "vector log1m_exp(vector x)"],
                "log(1 - exp(x)) for x <= 0, accurate at both ends (log(-expm1(x)) near 0, \
                 log1p(-exp(x)) below -log 2), elementwise for vectors. NaN for x > 0, -inf at 0", unary_math_sig),
            ("inv_logit", &["real inv_logit(real x)", "vector inv_logit(vector x)"],
                "1 / (1 + exp(-x)) without overflow for large |x|, elementwise for vectors", unary_math_sig),
            ("logit", &["real logit(real p)", "vector logit(vector p)"],
                "log(p / (1 - p)), computed as log(p) - log1m(p), elementwise for vectors. \
                 NaN outside [0, 1]", unary_math_sig),
            ("log_inv_logit", &["real log_inv_logit(real x)", "vector log_inv_logit(vector x)"],
                "log(inv_logit(x)) = -log1p_exp(-x), finite for very negative x, elementwise for vectors",
                unary_math_sig),
            ("log1m_inv_logit", &["real log1m_inv_logit(real x)", "vector log1m_inv_logit(vector x)"],
                "log(1 - inv_logit(x)) = -log1p_exp(x), finite for very positive x, elementwise for vectors",
                unary_math_sig),
            ("tanh", &["real tanh(real x)", "vector tanh(vector x)"],
                "hyperbolic tangent, elementwise for vectors", unary_math_sig),
            // ---- elementwise, two arguments (a real broadcasts) ------------
            ("fmin", &["real fmin(real x, real y)", "vector fmin(real|vector x, vector y)",
                       "vector fmin(vector x, real y)"],
                "the smaller of x and y, elementwise; a NaN argument is ignored", binary_math_sig),
            ("fmax", &["real fmax(real x, real y)", "vector fmax(real|vector x, vector y)",
                       "vector fmax(vector x, real y)"],
                "the larger of x and y, elementwise; a NaN argument is ignored", binary_math_sig),
            ("log_diff_exp", &["real log_diff_exp(real a, real b)",
                               "vector log_diff_exp(real|vector a, vector b)",
                               "vector log_diff_exp(vector a, real b)"],
                "log(exp(a) - exp(b)) for a >= b, as a + log1m_exp(b - a), elementwise. \
                 -inf when a == b, NaN when b > a", binary_math_sig),
            // ---- reductions --------------------------------------------------
            ("log_sum_exp", &["real log_sum_exp(real a, real b)", "real log_sum_exp(vector x)"],
                "log(exp(a) + exp(b)), or log(sum(exp(x))), shifted by the maximum so it never \
                 overflows; -inf for an empty vector or all -inf", log_sum_exp_sig),
            ("max", &["real max(real a, real b)", "real max(vector x)"],
                "the larger of a and b, or the largest element (-inf for an empty vector)", min_max_sig),
            ("min", &["real min(real a, real b)", "real min(vector x)"],
                "the smaller of a and b, or the smallest element (inf for an empty vector)", min_max_sig),
            ("mean", &["real mean(vector x)"],
                "sum(x) / size; NaN for an empty vector", reduce_sig),
            ("dot_self", &["real dot_self(vector x)"],
                "sum of x[i]^2", reduce_sig),
            ("norm2", &["real norm2(vector x)"],
                "Euclidean norm sqrt(dot_self(x)), scaled by max |x[i]| so it neither \
                 overflows nor underflows", reduce_sig),
            ("squared_distance", &["real squared_distance(vector a, vector b)"],
                "sum of (a[i] - b[i])^2", two_vectors_sig),
            // ---- vector to vector --------------------------------------------
            ("softmax", &["vector softmax(vector x)"],
                "exp(x) / sum(exp(x)), shifted by max(x) so it never overflows", vector_map_sig),
            ("log_softmax", &["vector log_softmax(vector x)"],
                "x - log_sum_exp(x), finite where softmax would underflow to 0", vector_map_sig),
            // ---- checks --------------------------------------------------------
            ("is_inf", &["int is_inf(real x)"],
                "1 if x is +inf or -inf, else 0", predicate_sig),
            ("is_nan", &["int is_nan(real x)"],
                "1 if x is NaN, else 0. A rejected or failed density evaluation can give NaN: \
                 test before using it", predicate_sig),
            // ---- constants ---------------------------------------------------
            ("pi", &["real pi()"], "π", constant_sig),
            ("e", &["real e()"], "Euler's number, exp(1)", constant_sig),
            ("negative_infinity", &["real negative_infinity()"],
                "-inf. As propose()'s log_correction it rejects the proposal", constant_sig),
            ("positive_infinity", &["real positive_infinity()"], "+inf", constant_sig),
        ];
        for &(name, signatures, doc, sig) in pure {
            entries.push(Entry {
                name,
                signatures,
                doc,
                rng: "none",
                needs_chain: false,
                kind: Kind::Function {
                    sig,
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
        _ => Err(bad_args("f(real x) or f(vector x)", a)),
    }
}

/// Two real-or-vector arguments, elementwise: a real broadcasts.
fn binary_math_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [x, y]
            if [x, y]
                .iter()
                .all(|t| t.is_scalar() || matches!(t, Ty::Vector(_))) =>
        {
            broadcast_type(a)
        }
        _ => Err(bad_args("f(real|vector x, real|vector y)", a)),
    }
}

/// `f(vector x)` -> real.
fn reduce_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(_)] => Ok(Ty::Real),
        _ => Err(bad_args("f(vector x)", a)),
    }
}

/// `f(real a, real b)` or `f(vector x)` -> real.
fn pair_or_reduce_sig(a: &[Ty]) -> Result<Ty, String> {
    match a {
        [Ty::Vector(_)] => Ok(Ty::Real),
        [x, y] if x.is_scalar() && y.is_scalar() => Ok(Ty::Real),
        _ => Err(bad_args("f(real a, real b) or f(vector x)", a)),
    }
}

fn log_sum_exp_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    pair_or_reduce_sig(a)
}

fn min_max_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    pair_or_reduce_sig(a)
}

/// `f(vector a, vector b)` of agreeing sizes -> real.
fn two_vectors_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(x), Ty::Vector(y)] if x.agrees(*y) => Ok(Ty::Real),
        [Ty::Vector(x), Ty::Vector(y)] => Err(format!("vector sizes {x} and {y} differ")),
        _ => Err(bad_args("f(vector a, vector b)", a)),
    }
}

/// `f(vector[n] x)` -> vector[n].
fn vector_map_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(n)] => Ok(Ty::Vector(*n)),
        _ => Err(bad_args("f(vector x)", a)),
    }
}

/// `f(real x)` -> int (0 or 1).
fn predicate_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [t] if t.is_scalar() => Ok(Ty::Int),
        _ => Err(bad_args("f(real x)", a)),
    }
}

/// `f()` -> real.
fn constant_sig(a: &[Ty], _: &[Option<i64>], _: Option<usize>) -> Result<Ty, String> {
    match a {
        [] => Ok(Ty::Real),
        _ => Err(bad_args("f()", a)),
    }
}

const PROPOSE_DOC: &str = "\
Metropolis-Hastings accept/reject of `q_new`. Ends the step.

The chain moves to `q_new` with probability `min(1, exp(log_alpha))`, where

```text
log_alpha = lp(q_new) - lp + log_correction
```

`log_correction` is the log proposal ratio `log g(q | q_new) - log g(q_new | q)`: \
0 for a symmetric proposal (random walk), the reverse minus the forward proposal \
log density otherwise (MALA, Barker). A wrong correction biases the chain without \
any error; `laplace sampler test --sbc` is the check that catches it.

- `lp(q_new)` is evaluated by the runtime, density only. If the step already \
evaluated `q_new` (`log_density`, `log_density_grad`), the cached value is reused \
at no cost.
- `lp(q_new) = -inf` (outside the support) or a NaN `log_alpha` is a rejection. \
`negative_infinity()` as the correction rejects on purpose.
- On rejection the chain stays at `q`, with its `lp` and `grad`.
- `min(1, exp(log_alpha))` is the acceptance statistic that `@adapt(step_size)` \
steers toward `target_accept`.
- Every path through `step { }` ends in exactly one `propose()` or `move()`, \
never inside a loop.";

const MOVE_DOC: &str = "\
Moves the chain to `q_new` unconditionally. Ends the step.

No accept/reject happens here, so the step itself must leave the target \
invariant: an exact conditional (Gibbs) draw, or a step that does its own \
accept/reject with `uniform_rng` and moves to the winner.

- The runtime evaluates `lp(q_new)` (free if the step already did) and **stops \
the run** if it is not finite: a `move()` outside the support is an error, not a \
rejection. Use `propose()` when `q_new` can leave the support.
- `move()` reports no acceptance statistic. `@adapt(step_size)` reads that as \
0 and keeps shrinking the step size, so a kernel that adapts its step size \
should end in `propose()`.
- Every path through `step { }` ends in exactly one `propose()` or `move()`, \
never inside a loop.";

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

#[cfg(test)]
mod tests {
    use super::*;

    /// `real|vector y` -> the types it allows (sizes such as `[dim]` are
    /// dropped: the rule is checked without a dimension).
    fn alternatives(param: &str) -> Vec<Ty> {
        let ty = param.rsplit_once(' ').map_or(param, |(t, _)| t);
        ty.split('|')
            .map(|t| match t.split('[').next().unwrap() {
                "int" => Ty::Int,
                "real" => Ty::Real,
                "vector" => Ty::Vector(Size::Unknown),
                other => panic!("unknown type `{other}` in a signature"),
            })
            .collect()
    }

    /// `vector[dim]` -> `vector`, to compare a declared type with a checked
    /// one whatever the sizes.
    fn without_sizes(t: &str) -> String {
        let mut out = String::new();
        let mut depth = 0;
        for c in t.chars() {
            match c {
                '[' => depth += 1,
                ']' => depth -= 1,
                _ if depth == 0 => out.push(c),
                _ => {}
            }
        }
        out
    }

    /// Every argument list the alternatives allow.
    fn combinations(params: &[Vec<Ty>]) -> Vec<Vec<Ty>> {
        params.iter().fold(vec![vec![]], |acc, alts| {
            acc.iter()
                .flat_map(|prefix| {
                    alts.iter().map(move |t| {
                        let mut v = prefix.clone();
                        v.push(t.clone());
                        v
                    })
                })
                .collect()
        })
    }

    /// The signatures shown in the editor are the ones the checker
    /// enforces: each one's argument types are accepted, and give its
    /// return type.
    #[test]
    fn every_signature_agrees_with_the_type_rule() {
        for e in Registry::standard().entries() {
            assert!(!e.signatures.is_empty(), "{} has no signature", e.name);
            for sig in e.signatures {
                match &e.kind {
                    Kind::Value { ty } => {
                        let (decl, name) = sig.rsplit_once(' ').unwrap();
                        assert_eq!(name, e.name, "{sig}");
                        assert_eq!(
                            without_sizes(&ty(None).to_string()),
                            without_sizes(decl),
                            "{sig}"
                        );
                    }
                    Kind::Function { sig: rule, .. } => {
                        // `<return type> <name>(...)`; the return type may
                        // have parentheses of its own (`tuple(...)`).
                        let at = sig
                            .find(&format!(" {}(", e.name))
                            .unwrap_or_else(|| panic!("`{sig}` is not `<type> {}(...)`", e.name));
                        let ret = &sig[..at];
                        let open = at + e.name.len() + 1;
                        let inner = &sig[open + 1..sig.len() - 1];
                        // In `y | mu, sigma` the outcome bar separates like a
                        // comma; `real|vector` (no spaces) is one parameter.
                        let params: Vec<Vec<Ty>> = inner
                            .split(", ")
                            .flat_map(|p| p.split(" | "))
                            .filter(|p| !p.is_empty())
                            .map(alternatives)
                            .collect();
                        for args in combinations(&params) {
                            let consts = vec![None; args.len()];
                            let got = rule(&args, &consts, None)
                                .unwrap_or_else(|m| panic!("{sig} with {args:?}: {m}"));
                            assert_eq!(
                                without_sizes(&got.to_string()),
                                without_sizes(ret),
                                "{sig} with {args:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
