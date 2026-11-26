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

use nalgebra::{DMatrix, DVector};
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
        let pure: &[(&'static str, CallFn)] = &[
            // ---- elementwise, one argument --------------------------------
            ("exp", |_, a| unary_math(&a[0], f64::exp)),
            ("log", |_, a| unary_math(&a[0], f64::ln)),
            ("sqrt", |_, a| unary_math(&a[0], f64::sqrt)),
            ("square", |_, a| unary_math(&a[0], |x| x * x)),
            ("abs", |_, a| unary_math(&a[0], f64::abs)),
            ("inv", |_, a| unary_math(&a[0], |x| 1.0 / x)),
            ("inv_sqrt", |_, a| unary_math(&a[0], |x| 1.0 / x.sqrt())),
            ("inv_square", |_, a| unary_math(&a[0], |x| 1.0 / (x * x))),
            ("expm1", |_, a| unary_math(&a[0], f64::exp_m1)),
            ("log1p", |_, a| unary_math(&a[0], f64::ln_1p)),
            ("log1m", |_, a| unary_math(&a[0], log1m)),
            ("log1p_exp", |_, a| unary_math(&a[0], log1p_exp)),
            ("log1m_exp", |_, a| unary_math(&a[0], log1m_exp)),
            ("inv_logit", |_, a| unary_math(&a[0], inv_logit)),
            ("logit", |_, a| unary_math(&a[0], |p| p.ln() - log1m(p))),
            ("log_inv_logit", |_, a| {
                unary_math(&a[0], |x| -log1p_exp(-x))
            }),
            ("log1m_inv_logit", |_, a| {
                unary_math(&a[0], |x| -log1p_exp(x))
            }),
            ("tanh", |_, a| unary_math(&a[0], f64::tanh)),
            // ---- elementwise, two arguments --------------------------------
            ("fmin", |_, a| binary_math(a, f64::min)),
            ("fmax", |_, a| binary_math(a, f64::max)),
            ("log_diff_exp", |_, a| binary_math(a, log_diff_exp)),
            // ---- reductions --------------------------------------------------
            ("log_sum_exp", |_, a| match a {
                [x] => Ok(Value::Real(log_sum_exp(vec_arg(x)?.as_slice()))),
                [x, y] => Ok(Value::Real(log_sum_exp(&[real_arg(x)?, real_arg(y)?]))),
                _ => Err("log_sum_exp: expected one vector or two reals".into()),
            }),
            ("max", |_, a| pair_or_fold(a, f64::NEG_INFINITY, f64::max)),
            ("min", |_, a| pair_or_fold(a, f64::INFINITY, f64::min)),
            ("mean", |_, a| {
                let x = vec_arg(&a[0])?;
                Ok(Value::Real(x.iter().sum::<f64>() / x.len() as f64))
            }),
            ("dot_self", |_, a| {
                Ok(Value::Real(vec_arg(&a[0])?.iter().map(|x| x * x).sum()))
            }),
            ("norm2", |_, a| {
                Ok(Value::Real(norm2(vec_arg(&a[0])?.as_slice())))
            }),
            ("squared_distance", |_, a| {
                let (x, y) = (vec_arg(&a[0])?, vec_arg(&a[1])?);
                same_len("squared_distance", x, y)?;
                Ok(Value::Real(
                    x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum(),
                ))
            }),
            // ---- vector to vector --------------------------------------------
            ("softmax", |_, a| {
                // exp(x - max) / sum, not exp(x - log_sum_exp(x)): the
                // rounding of a large log_sum_exp would land in the exponent.
                let x = vec_arg(&a[0])?;
                let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w = x.map(|v| (v - m).exp());
                let total: f64 = w.iter().sum();
                Ok(Value::Vector(w / total))
            }),
            ("log_softmax", |_, a| {
                let x = vec_arg(&a[0])?;
                let lse = log_sum_exp(x.as_slice());
                Ok(Value::Vector(x.map(|v| v - lse)))
            }),
            // ---- checks --------------------------------------------------------
            ("is_inf", |_, a| {
                Ok(Value::Int(real_arg(&a[0])?.is_infinite() as i64))
            }),
            ("is_nan", |_, a| {
                Ok(Value::Int(real_arg(&a[0])?.is_nan() as i64))
            }),
            // ---- matrices --------------------------------------------------------
            ("rep_matrix", |_, a| {
                let x = real_arg(&a[0])?;
                let (r, c) = (size_arg(&a[1])?, size_arg(&a[2])?);
                Ok(Value::Matrix(DMatrix::from_element(r, c, x)))
            }),
            ("rows", |_, a| match &a[0] {
                Value::Matrix(m) => Ok(Value::Int(m.nrows() as i64)),
                other => Ok(Value::Int(vec_arg(other)?.len() as i64)),
            }),
            ("cols", |_, a| match &a[0] {
                Value::Matrix(m) => Ok(Value::Int(m.ncols() as i64)),
                other => vec_arg(other).map(|_| Value::Int(1)),
            }),
            ("col", |_, a| {
                let m = mat_arg(&a[0])?;
                let j = line_index("column", &a[1], m.ncols())?;
                Ok(Value::Vector(m.column(j).into_owned()))
            }),
            ("row", |_, a| {
                let m = mat_arg(&a[0])?;
                let i = line_index("row", &a[1], m.nrows())?;
                Ok(Value::Vector(m.row(i).transpose()))
            }),
            ("set_col", |_, a| {
                let mut m = mat_arg(&a[0])?.clone();
                let j = line_index("column", &a[1], m.ncols())?;
                let v = vec_arg(&a[2])?;
                line_len("column", m.nrows(), v)?;
                m.set_column(j, v);
                Ok(Value::Matrix(m))
            }),
            ("set_row", |_, a| {
                let mut m = mat_arg(&a[0])?.clone();
                let i = line_index("row", &a[1], m.nrows())?;
                let v = vec_arg(&a[2])?;
                line_len("row", m.ncols(), v)?;
                m.set_row(i, &v.transpose());
                Ok(Value::Matrix(m))
            }),
            // ---- constants -----------------------------------------------------
            ("pi", |_, _| Ok(Value::Real(std::f64::consts::PI))),
            ("e", |_, _| Ok(Value::Real(std::f64::consts::E))),
            ("negative_infinity", |_, _| {
                Ok(Value::Real(f64::NEG_INFINITY))
            }),
            ("positive_infinity", |_, _| Ok(Value::Real(f64::INFINITY))),
        ];
        for &(name, call) in pure {
            entries.push((name, Native::Function(call)));
        }

        Natives { entries }
    }
}

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

// ---- numerically safe scalar functions (formulas as in Stan's math library) --

/// log(1 - x).
fn log1m(x: f64) -> f64 {
    (-x).ln_1p()
}

/// log(1 + exp(x)): never overflows, exact to rounding for large |x|.
fn log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// log(1 - exp(x)) for x <= 0 (Mächler 2012: switch at -log 2).
fn log1m_exp(x: f64) -> f64 {
    if x > 0.0 {
        f64::NAN
    } else if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// 1 / (1 + exp(-x)), with exp() only ever of a non-positive number.
fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// log(exp(a) - exp(b)) for a >= b.
fn log_diff_exp(a: f64, b: f64) -> f64 {
    if b > a {
        f64::NAN
    } else if a == b {
        f64::NEG_INFINITY
    } else {
        a + log1m_exp(b - a)
    }
}

/// log(sum(exp(x))), shifted by the maximum.
fn log_sum_exp(x: &[f64]) -> f64 {
    if x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m.is_infinite() {
        // empty or all -inf: -inf; some +inf: +inf
        return m;
    }
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

/// sqrt(sum(x^2)), scaled by max |x| to avoid overflow and underflow.
fn norm2(x: &[f64]) -> f64 {
    if x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let s = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if s == 0.0 || s.is_infinite() {
        return s;
    }
    s * x.iter().map(|v| (v / s) * (v / s)).sum::<f64>().sqrt()
}

/// `f(real|vector, real|vector)` elementwise; a real broadcasts.
fn binary_math(a: &[Value], f: fn(f64, f64) -> f64) -> Result<Value, String> {
    match broadcast_len(a)? {
        None => Ok(Value::Real(f(real_arg(&a[0])?, real_arg(&a[1])?))),
        Some(n) => Ok(Value::Vector(DVector::from_fn(n, |i, _| {
            f(elem(&a[0], i), elem(&a[1], i))
        }))),
    }
}

/// `f(real, real)`, or `f` folded over a vector from `init`. Unlike
/// `fmin`/`fmax`, a NaN anywhere gives NaN.
fn pair_or_fold(a: &[Value], init: f64, f: fn(f64, f64) -> f64) -> Result<Value, String> {
    let xs: Vec<f64> = match a {
        [x] => vec_arg(x)?.iter().copied().collect(),
        [x, y] => vec![real_arg(x)?, real_arg(y)?],
        _ => return Err("expected one vector or two reals".into()),
    };
    if xs.iter().any(|v| v.is_nan()) {
        return Ok(Value::Real(f64::NAN));
    }
    Ok(Value::Real(xs.into_iter().fold(init, f)))
}

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

fn mat_arg(v: &Value) -> Result<&DMatrix<f64>, String> {
    match v {
        Value::Matrix(m) => Ok(m),
        other => Err(format!("expected a matrix, got {}", other.type_name())),
    }
}

/// 1-based row/column index -> 0-based, bounds-checked.
fn line_index(what: &str, v: &Value, count: usize) -> Result<usize, String> {
    let i = v
        .as_int()
        .ok_or_else(|| format!("the {what} index must be an int"))?;
    if i >= 1 && (i as usize) <= count {
        Ok(i as usize - 1)
    } else {
        Err(format!("{what} {i} is out of range 1..{count}"))
    }
}

/// The vector written into a row/column must fill it exactly.
fn line_len(what: &str, want: usize, v: &DVector<f64>) -> Result<(), String> {
    if v.len() == want {
        Ok(())
    } else {
        Err(format!(
            "a {what} of this matrix has {want} elements, the vector has {}",
            v.len()
        ))
    }
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

    fn call(name: &str, args: &[Value]) -> Value {
        let mut c = Ctx {
            dim: 0,
            state: None,
            ev: None,
            rng: None,
            metric: None,
        };
        match Natives::standard().get(name) {
            Some(Native::Function(f)) => f(&mut c, args).unwrap(),
            _ => panic!("`{name}` is not a function"),
        }
    }

    fn real(name: &str, args: &[f64]) -> f64 {
        let args: Vec<Value> = args.iter().map(|&x| Value::Real(x)).collect();
        call(name, &args).as_real().unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        a == b || (a - b).abs() <= 1e-14 * a.abs().max(b.abs())
    }

    /// The safe composites stay finite and accurate where the naive
    /// formula overflows, underflows or cancels.
    #[test]
    fn safe_composites_at_the_extremes() {
        assert_eq!(real("log1p_exp", &[800.0]), 800.0);
        assert!(close(real("log1p_exp", &[-800.0]), (-800.0_f64).exp()));
        assert!(close(real("log1p_exp", &[0.0]), std::f64::consts::LN_2));
        assert_eq!(real("inv_logit", &[800.0]), 1.0);
        assert!(close(real("inv_logit", &[-700.0]), (-700.0_f64).exp()));
        assert_eq!(real("inv_logit", &[0.0]), 0.5);
        assert!(close(real("log_inv_logit", &[-800.0]), -800.0));
        assert!(close(real("log1m_inv_logit", &[800.0]), -800.0));
        assert!(close(real("logit", &[0.75]), 3.0_f64.ln()));
        // log(1 - exp(-1e-20)) = log(1e-20) to rounding; the naive form is -inf.
        assert!(close(real("log1m_exp", &[-1e-20]), (1e-20_f64).ln()));
        assert!(close(real("log1m_exp", &[-50.0]), -(-50.0_f64).exp()));
        assert_eq!(real("log1m_exp", &[0.0]), f64::NEG_INFINITY);
        assert!(real("log1m_exp", &[1.0]).is_nan());
        assert!(close(real("log1m", &[1e-20]), -1e-20));
        assert!(close(
            real("log_sum_exp", &[1000.0, 1000.0]),
            1000.0 + std::f64::consts::LN_2
        ));
        assert_eq!(
            real("log_sum_exp", &[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
        assert!(close(
            real("log_diff_exp", &[1000.0, 1000.0 - std::f64::consts::LN_2]),
            1000.0 - std::f64::consts::LN_2
        ));
        assert_eq!(real("log_diff_exp", &[3.0, 3.0]), f64::NEG_INFINITY);
        assert!(real("log_diff_exp", &[1.0, 2.0]).is_nan());
        assert_eq!(real("fmin", &[f64::NAN, 2.0]), 2.0);
        assert!(real("max", &[f64::NAN, 2.0]).is_nan());
    }

    #[test]
    fn vector_functions() {
        let v = |xs: &[f64]| Value::Vector(DVector::from_column_slice(xs));
        let r = |x: Value| x.as_real().unwrap();
        assert!(close(r(call("norm2", &[v(&[3e200, 4e200])])), 5e200));
        assert!(close(r(call("norm2", &[v(&[3e-200, 4e-200])])), 5e-200));
        assert!(close(
            r(call("log_sum_exp", &[v(&[1000.0, 1000.0, 1000.0])])),
            1000.0 + 3.0_f64.ln()
        ));
        assert_eq!(r(call("log_sum_exp", &[v(&[])])), f64::NEG_INFINITY);
        let Value::Vector(s) = call("softmax", &[v(&[1000.0, 1000.0])]) else {
            panic!()
        };
        assert_eq!(s.as_slice(), &[0.5, 0.5]);
        let Value::Vector(l) = call("log_softmax", &[v(&[0.0, -2000.0])]) else {
            panic!()
        };
        assert!(l[1].is_finite() && close(l[1], -2000.0));
        assert_eq!(r(call("max", &[v(&[1.0, 5.0, 3.0])])), 5.0);
        assert_eq!(r(call("min", &[v(&[])])), f64::INFINITY);
        assert_eq!(r(call("mean", &[v(&[1.0, 2.0, 6.0])])), 3.0);
        assert_eq!(
            r(call("squared_distance", &[v(&[1.0, 2.0]), v(&[4.0, 6.0])])),
            25.0
        );
        // a real broadcasts against a vector
        let Value::Vector(m) = call("fmax", &[v(&[-1.0, 2.0]), Value::Real(0.0)]) else {
            panic!()
        };
        assert_eq!(m.as_slice(), &[0.0, 2.0]);
        let Value::Vector(e) = call("inv_logit", &[v(&[-800.0, 800.0])]) else {
            panic!()
        };
        assert_eq!(e.as_slice(), &[0.0, 1.0]);
        assert_eq!(call("is_nan", &[Value::Real(f64::NAN)]).as_int(), Some(1));
        assert_eq!(call("is_inf", &[Value::Int(3)]).as_int(), Some(0));
    }

    #[test]
    fn matrix_functions() {
        let v = |xs: &[f64]| Value::Vector(DVector::from_column_slice(xs));
        // 2 x 3, column-major: [1 3 5; 2 4 6]
        let m = Value::Matrix(DMatrix::from_column_slice(
            2,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let int = |x: Value| x.as_int().unwrap();
        assert_eq!(int(call("rows", std::slice::from_ref(&m))), 2);
        assert_eq!(int(call("cols", std::slice::from_ref(&m))), 3);
        assert_eq!(int(call("rows", &[v(&[1.0, 2.0, 3.0])])), 3);
        assert_eq!(int(call("cols", &[v(&[1.0, 2.0, 3.0])])), 1);
        assert_eq!(call("col", &[m.clone(), Value::Int(2)]), v(&[3.0, 4.0]));
        assert_eq!(
            call("row", &[m.clone(), Value::Int(2)]),
            v(&[2.0, 4.0, 6.0])
        );
        assert_eq!(
            call("set_col", &[m.clone(), Value::Int(3), v(&[-1.0, -2.0])]),
            Value::Matrix(DMatrix::from_column_slice(
                2,
                3,
                &[1.0, 2.0, 3.0, 4.0, -1.0, -2.0]
            ))
        );
        assert_eq!(
            call("set_row", &[m.clone(), Value::Int(1), v(&[7.0, 8.0, 9.0])]),
            Value::Matrix(DMatrix::from_column_slice(
                2,
                3,
                &[7.0, 2.0, 8.0, 4.0, 9.0, 6.0]
            ))
        );
        assert_eq!(
            call("rep_matrix", &[Value::Int(0), Value::Int(2), Value::Int(1)]),
            Value::Matrix(DMatrix::zeros(2, 1))
        );
    }

    #[test]
    fn matrix_functions_check_indices_and_lengths() {
        let fail = |name: &str, args: &[Value]| {
            let mut c = Ctx {
                dim: 0,
                state: None,
                ev: None,
                rng: None,
                metric: None,
            };
            let Some(Native::Function(f)) = Natives::standard().get(name) else {
                panic!()
            };
            f(&mut c, args).unwrap_err()
        };
        let m = Value::Matrix(DMatrix::zeros(2, 3));
        let v2 = Value::Vector(DVector::zeros(2));
        assert_eq!(
            fail("col", &[m.clone(), Value::Int(4)]),
            "column 4 is out of range 1..3"
        );
        assert_eq!(
            fail("row", &[m.clone(), Value::Int(0)]),
            "row 0 is out of range 1..2"
        );
        assert_eq!(
            fail("set_row", &[m.clone(), Value::Int(1), v2]),
            "a row of this matrix has 3 elements, the vector has 2"
        );
    }

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
