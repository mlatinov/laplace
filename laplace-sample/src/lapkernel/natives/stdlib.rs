//! Implementations of the kernel standard library
//! (`laplace_sampler_syntax::lapkernel::builtins::stdlib`), keyed by the
//! same names. The registry there owns each function's type rule and
//! documentation; this file owns what it computes.
//!
//! Errors are returned without the function's name: the interpreter
//! prefixes it (`file:line:col: gamma_rng: alpha must be positive, got 0`).

use std::f64::consts::{LN_2, PI, SQRT_2};

use nalgebra::{DMatrix, DVector};
use rand::RngExt;
use rand_distr::{Beta, Binomial, Exp1, Gamma, Poisson, StandardNormal, StudentT};

use super::{
    binary_math, broadcast_len, elem, elems, inv_logit, log1m, log1p_exp, log_sum_exp, mat_arg,
    real_arg, same_len, size_arg, unary_math, vec_arg, CallFn, Ctx, Native, Value, HALF_LN_2PI,
};
use crate::kernel::ChainRng;

pub(super) fn push(entries: &mut Vec<(&'static str, Native)>) {
    let rows: &[(&'static str, CallFn)] = &[
        // ---- trigonometric and hyperbolic ------------------------------------
        ("sin", |_, a| unary_math(&a[0], f64::sin)),
        ("cos", |_, a| unary_math(&a[0], f64::cos)),
        ("tan", |_, a| unary_math(&a[0], f64::tan)),
        ("asin", |_, a| unary_math(&a[0], f64::asin)),
        ("acos", |_, a| unary_math(&a[0], f64::acos)),
        ("atan", |_, a| unary_math(&a[0], f64::atan)),
        ("sinh", |_, a| unary_math(&a[0], f64::sinh)),
        ("cosh", |_, a| unary_math(&a[0], f64::cosh)),
        ("asinh", |_, a| unary_math(&a[0], f64::asinh)),
        ("acosh", |_, a| unary_math(&a[0], f64::acosh)),
        ("atanh", |_, a| unary_math(&a[0], f64::atanh)),
        // ---- powers and logarithms -------------------------------------------
        ("log2", |_, a| unary_math(&a[0], f64::log2)),
        ("log10", |_, a| unary_math(&a[0], f64::log10)),
        ("exp2", |_, a| unary_math(&a[0], f64::exp2)),
        ("cbrt", |_, a| unary_math(&a[0], f64::cbrt)),
        // ---- rounding --------------------------------------------------------
        ("floor", |_, a| unary_math(&a[0], f64::floor)),
        ("ceil", |_, a| unary_math(&a[0], f64::ceil)),
        ("round", |_, a| unary_math(&a[0], f64::round)),
        ("trunc", |_, a| unary_math(&a[0], f64::trunc)),
        // ---- special functions -----------------------------------------------
        ("erf", |_, a| unary_math(&a[0], libm::erf)),
        ("erfc", |_, a| unary_math(&a[0], libm::erfc)),
        ("Phi", |_, a| unary_math(&a[0], phi)),
        ("Phi_approx", |_, a| {
            unary_math(&a[0], |x| inv_logit(0.07056 * x * x * x + 1.5976 * x))
        }),
        ("inv_Phi", |_, a| unary_math(&a[0], inv_phi)),
        ("lgamma", |_, a| unary_math(&a[0], libm::lgamma)),
        ("tgamma", |_, a| unary_math(&a[0], libm::tgamma)),
        ("digamma", |_, a| unary_math(&a[0], digamma)),
        ("inv_cloglog", |_, a| {
            unary_math(&a[0], |x| -(-x.exp()).exp_m1())
        }),
        // ---- two arguments ---------------------------------------------------
        ("pow", |_, a| binary_math(a, f64::powf)),
        ("atan2", |_, a| binary_math(a, f64::atan2)),
        ("hypot", |_, a| binary_math(a, f64::hypot)),
        ("fdim", |_, a| {
            binary_math(a, |x, y| {
                if x.is_nan() || y.is_nan() {
                    f64::NAN
                } else {
                    (x - y).max(0.0)
                }
            })
        }),
        ("fmod", |_, a| binary_math(a, |x, y| x % y)),
        ("lbeta", |_, a| binary_math(a, lbeta)),
        ("lchoose", |_, a| binary_math(a, lchoose)),
        ("log_mix", |_, a| {
            let (theta, lp1, lp2) = (real_arg(&a[0])?, real_arg(&a[1])?, real_arg(&a[2])?);
            probability("theta", theta)?;
            Ok(Value::Real(log_sum_exp(&[
                theta.ln() + lp1,
                log1m(theta) + lp2,
            ])))
        }),
        // ---- integers and conversions ----------------------------------------
        ("to_int", |_, a| {
            let x = real_arg(&a[0])?;
            // i64::MAX is not an f64; 2^63 is the first value past it.
            const LIMIT: f64 = 9_223_372_036_854_775_808.0;
            if !x.is_finite() || x.trunc() >= LIMIT || x.trunc() < -LIMIT {
                return Err(format!("{x} is not in the int range"));
            }
            Ok(Value::Int(x.trunc() as i64))
        }),
        ("int_step", |_, a| {
            Ok(Value::Int((real_arg(&a[0])? > 0.0) as i64))
        }),
        ("choose", |_, a| {
            let (n, k) = (int_arg(&a[0])?, int_arg(&a[1])?);
            choose(n, k).map(Value::Int)
        }),
        ("to_vector", |_, a| match &a[0] {
            Value::Matrix(m) => Ok(Value::Vector(DVector::from_column_slice(m.as_slice()))),
            Value::Vector(v) => Ok(Value::Vector(v.clone())),
            other => Ok(Value::Vector(DVector::from_element(1, real_arg(other)?))),
        }),
        ("to_matrix", |_, a| {
            let x = elems(&a[0])?;
            let (r, c) = match a {
                [_] => (x.len(), 1),
                [_, r, c] => (size_arg(r)?, size_arg(c)?),
                _ => unreachable!(),
            };
            if r * c != x.len() {
                return Err(format!(
                    "{r} x {c} is {} elements, the input has {}",
                    r * c,
                    x.len()
                ));
            }
            Ok(Value::Matrix(DMatrix::from_column_slice(r, c, x)))
        }),
        ("size", |_, a| Ok(Value::Int(elems(&a[0])?.len() as i64))),
        ("num_elements", |_, a| {
            Ok(Value::Int(elems(&a[0])?.len() as i64))
        }),
        // ---- reductions --------------------------------------------------------
        ("prod", |_, a| {
            Ok(Value::Real(elems(&a[0])?.iter().product()))
        }),
        ("variance", |_, a| variance(elems(&a[0])?).map(Value::Real)),
        ("sd", |_, a| {
            variance(elems(&a[0])?).map(|v| Value::Real(v.sqrt()))
        }),
        ("norm1", |_, a| {
            Ok(Value::Real(vec_arg(&a[0])?.iter().map(|x| x.abs()).sum()))
        }),
        ("distance", |_, a| {
            let (x, y) = (vec_arg(&a[0])?, vec_arg(&a[1])?);
            same_len("distance", x, y)?;
            Ok(Value::Real((x - y).norm()))
        }),
        // ---- building and reshaping vectors --------------------------------------
        ("zeros_vector", |_, a| {
            Ok(Value::Vector(DVector::zeros(size_arg(&a[0])?)))
        }),
        ("ones_vector", |_, a| {
            Ok(Value::Vector(DVector::from_element(size_arg(&a[0])?, 1.0)))
        }),
        ("linspaced_vector", |_, a| {
            let (n, lo, hi) = (size_arg(&a[0])?, real_arg(&a[1])?, real_arg(&a[2])?);
            Ok(Value::Vector(DVector::from_fn(n, |i, _| {
                if i + 1 == n {
                    hi
                } else {
                    lo + (hi - lo) * i as f64 / (n - 1) as f64
                }
            })))
        }),
        ("one_hot_vector", |_, a| {
            let n = size_arg(&a[0])?;
            let k = position(&a[1], n, "position")?;
            let mut v = DVector::zeros(n);
            v[k] = 1.0;
            Ok(Value::Vector(v))
        }),
        ("head", |_, a| {
            let (v, n) = (vec_arg(&a[0])?, size_arg(&a[1])?);
            fits(n, v.len())?;
            Ok(Value::Vector(v.rows(0, n).into_owned()))
        }),
        ("tail", |_, a| {
            let (v, n) = (vec_arg(&a[0])?, size_arg(&a[1])?);
            fits(n, v.len())?;
            Ok(Value::Vector(v.rows(v.len() - n, n).into_owned()))
        }),
        ("segment", |_, a| {
            let (v, n) = (vec_arg(&a[0])?, size_arg(&a[2])?);
            let i = int_arg(&a[1])?;
            if i < 1 || (i as usize - 1) + n > v.len() {
                return Err(format!(
                    "{n} elements from position {i} do not fit in a vector of {}",
                    v.len()
                ));
            }
            Ok(Value::Vector(v.rows(i as usize - 1, n).into_owned()))
        }),
        ("append_row", |_, a| match (&a[0], &a[1]) {
            (Value::Matrix(x), Value::Matrix(y)) => {
                if x.ncols() != y.ncols() {
                    return Err(format!(
                        "the matrices have {} and {} columns",
                        x.ncols(),
                        y.ncols()
                    ));
                }
                let r = x.nrows();
                Ok(Value::Matrix(DMatrix::from_fn(
                    r + y.nrows(),
                    x.ncols(),
                    |i, j| {
                        if i < r {
                            x[(i, j)]
                        } else {
                            y[(i - r, j)]
                        }
                    },
                )))
            }
            (x, y) => {
                let mut out = column(x)?;
                out.extend(column(y)?);
                Ok(Value::Vector(DVector::from_vec(out)))
            }
        }),
        ("append_col", |_, a| {
            let (x, y) = (as_matrix(&a[0])?, as_matrix(&a[1])?);
            if x.nrows() != y.nrows() {
                return Err(format!("the two have {} and {} rows", x.nrows(), y.nrows()));
            }
            let c = x.ncols();
            Ok(Value::Matrix(DMatrix::from_fn(
                x.nrows(),
                c + y.ncols(),
                |i, j| {
                    if j < c {
                        x[(i, j)]
                    } else {
                        y[(i, j - c)]
                    }
                },
            )))
        }),
        ("reverse", |_, a| {
            let v = vec_arg(&a[0])?;
            Ok(Value::Vector(DVector::from_iterator(
                v.len(),
                v.iter().rev().copied(),
            )))
        }),
        ("sort_asc", |_, a| sorted(vec_arg(&a[0])?, false)),
        ("sort_desc", |_, a| sorted(vec_arg(&a[0])?, true)),
        ("cumulative_sum", |_, a| {
            let mut total = 0.0;
            Ok(Value::Vector(vec_arg(&a[0])?.map(|x| {
                total += x;
                total
            })))
        }),
        // ---- matrices and linear algebra -----------------------------------------
        ("identity_matrix", |_, a| {
            let n = size_arg(&a[0])?;
            Ok(Value::Matrix(DMatrix::identity(n, n)))
        }),
        ("diag_matrix", |_, a| {
            Ok(Value::Matrix(DMatrix::from_diagonal(vec_arg(&a[0])?)))
        }),
        ("diagonal", |_, a| {
            let m = mat_arg(&a[0])?;
            let n = m.nrows().min(m.ncols());
            Ok(Value::Vector(DVector::from_fn(n, |i, _| m[(i, i)])))
        }),
        ("trace", |_, a| {
            let m = mat_arg(&a[0])?;
            Ok(Value::Real(
                (0..m.nrows().min(m.ncols())).map(|i| m[(i, i)]).sum(),
            ))
        }),
        ("determinant", |_, a| {
            Ok(Value::Real(square_arg(&a[0])?.clone().lu().determinant()))
        }),
        ("log_determinant", |_, a| {
            let u = square_arg(&a[0])?.clone().lu().u();
            Ok(Value::Real(
                (0..u.nrows()).map(|i| u[(i, i)].abs().ln()).sum(),
            ))
        }),
        ("inverse", |_, a| {
            square_arg(&a[0])?
                .clone()
                .try_inverse()
                .map(Value::Matrix)
                .ok_or_else(|| "the matrix is singular".to_string())
        }),
        ("cholesky_decompose", |_, a| {
            cholesky(mat_arg(&a[0])?).map(Value::Matrix)
        }),
        ("eigenvalues_sym", |_, a| {
            eigen_sym(mat_arg(&a[0])?).map(|(v, _)| Value::Vector(v))
        }),
        ("eigenvectors_sym", |_, a| {
            eigen_sym(mat_arg(&a[0])?).map(|(_, m)| Value::Matrix(m))
        }),
        ("crossprod", |_, a| {
            let m = mat_arg(&a[0])?;
            Ok(Value::Matrix(m.tr_mul(m)))
        }),
        ("tcrossprod", |_, a| {
            let m = mat_arg(&a[0])?;
            Ok(Value::Matrix(m * m.transpose()))
        }),
        ("multiply_lower_tri_self_transpose", |_, a| {
            let l = mat_arg(&a[0])?.lower_triangle();
            Ok(Value::Matrix(&l * l.transpose()))
        }),
        ("quad_form", |_, a| {
            let (m, v) = (square_arg(&a[0])?, vec_arg(&a[1])?);
            rhs_rows(m, v.len())?;
            Ok(Value::Real(v.dot(&(m * v))))
        }),
        ("mdivide_left", |_, a| {
            let m = square_arg(&a[0])?;
            let lu = m.clone().lu();
            solve(m, &a[1], |b| lu.solve(b)).ok_or_else(|| "the matrix is singular".to_string())?
        }),
        ("mdivide_left_tri_low", |_, a| {
            let m = square_arg(&a[0])?;
            solve(m, &a[1], |b| m.solve_lower_triangular(b))
                .ok_or_else(|| "a diagonal element of L is zero".to_string())?
        }),
        ("mdivide_left_spd", |_, a| {
            let m = mat_arg(&a[0])?;
            cholesky(m)?;
            let chol = m.clone().cholesky().expect("checked by cholesky()");
            solve(m, &a[1], |b| Some(chol.solve(b))).expect("a Cholesky solve cannot fail")
        }),
        // ---- random numbers: continuous, broadcasting ------------------------------
        ("exponential_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("beta", p[0])?;
                let e: f64 = rng.sample(Exp1);
                Ok(e / p[0])
            })
        }),
        ("gamma_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("alpha", p[0])?;
                positive("beta", p[1])?;
                let g = Gamma::new(p[0], 1.0 / p[1]).map_err(|e| e.to_string())?;
                Ok(rng.sample(g))
            })
        }),
        ("beta_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("a", p[0])?;
                positive("b", p[1])?;
                let b = Beta::new(p[0], p[1]).map_err(|e| e.to_string())?;
                Ok(rng.sample(b))
            })
        }),
        ("lognormal_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("sigma", p[1])?;
                let z: f64 = rng.sample(StandardNormal);
                Ok((p[0] + p[1] * z).exp())
            })
        }),
        ("cauchy_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("sigma", p[1])?;
                let u: f64 = rng.random();
                Ok(p[0] + p[1] * (PI * (u - 0.5)).tan())
            })
        }),
        ("double_exponential_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("sigma", p[1])?;
                let w: f64 = rng.random::<f64>() - 0.5;
                Ok(p[0] - p[1] * w.signum() * (1.0 - 2.0 * w.abs()).ln())
            })
        }),
        ("student_t_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("nu", p[0])?;
                positive("sigma", p[2])?;
                let t = StudentT::new(p[0]).map_err(|e| e.to_string())?;
                Ok(p[1] + p[2] * rng.sample(t))
            })
        }),
        ("chi_square_rng", |c, a| {
            draw_each(c, a, |rng, p| {
                positive("nu", p[0])?;
                let g = Gamma::new(p[0] / 2.0, 2.0).map_err(|e| e.to_string())?;
                Ok(rng.sample(g))
            })
        }),
        // ---- random numbers: discrete --------------------------------------------
        ("bernoulli_rng", |c, a| {
            let theta = real_arg(&a[0])?;
            probability("theta", theta)?;
            let u: f64 = c.rng()?.random();
            Ok(Value::Int((u < theta) as i64))
        }),
        ("bernoulli_logit_rng", |c, a| {
            let alpha = real_arg(&a[0])?;
            if alpha.is_nan() {
                return Err("alpha is NaN".into());
            }
            let u: f64 = c.rng()?.random();
            Ok(Value::Int((u < inv_logit(alpha)) as i64))
        }),
        ("binomial_rng", |c, a| {
            let (n, theta) = (int_arg(&a[0])?, real_arg(&a[1])?);
            if n < 0 {
                return Err(format!("N must be non-negative, got {n}"));
            }
            probability("theta", theta)?;
            let b = Binomial::new(n as u64, theta).map_err(|e| e.to_string())?;
            Ok(Value::Int(c.rng()?.sample(b) as i64))
        }),
        ("poisson_rng", |c, a| poisson(c, real_arg(&a[0])?)),
        ("poisson_log_rng", |c, a| poisson(c, real_arg(&a[0])?.exp())),
        ("categorical_rng", |c, a| {
            let theta = vec_arg(&a[0])?;
            simplex(theta)?;
            let u: f64 = c.rng()?.random();
            Ok(Value::Int(pick(theta.as_slice(), u)))
        }),
        ("categorical_logit_rng", |c, a| {
            let w = softmax(vec_arg(&a[0])?)?;
            let u: f64 = c.rng()?.random();
            Ok(Value::Int(pick(w.as_slice(), u)))
        }),
        ("discrete_range_rng", |c, a| {
            let (lo, hi) = (int_arg(&a[0])?, int_arg(&a[1])?);
            if lo > hi {
                return Err(format!("lower {lo} is above upper {hi}"));
            }
            let span = (hi as i128 - lo as i128 + 1) as f64;
            let u: f64 = c.rng()?.random();
            Ok(Value::Int(
                ((lo as i128 + (u * span) as i128).min(hi as i128)) as i64,
            ))
        }),
        // ---- random numbers: multivariate ----------------------------------------
        ("multi_normal_rng", |c, a| {
            let mu = vec_arg(&a[0])?;
            let l = cholesky(mat_arg(&a[1])?)?;
            rhs_rows(&l, mu.len())?;
            Ok(Value::Vector(mu + l * normals(c.rng()?, mu.len())))
        }),
        ("multi_normal_cholesky_rng", |c, a| {
            let mu = vec_arg(&a[0])?;
            let l = square_arg(&a[1])?.lower_triangle();
            rhs_rows(&l, mu.len())?;
            Ok(Value::Vector(mu + l * normals(c.rng()?, mu.len())))
        }),
        ("dirichlet_rng", |c, a| {
            let alpha = vec_arg(&a[0])?;
            let rng = c.rng()?;
            let mut g = DVector::zeros(alpha.len());
            for (i, &k) in alpha.iter().enumerate() {
                positive("alpha", k)?;
                g[i] = rng.sample(Gamma::new(k, 1.0).map_err(|e| e.to_string())?);
            }
            let total = g.sum();
            Ok(Value::Vector(g / total))
        }),
        // ---- densities ----------------------------------------------------------------
        ("exponential_lpdf", |_, a| {
            sum_each(a, |p| {
                let (y, beta) = (p[0], p[1]);
                positive("beta", beta)?;
                Ok(if y < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    beta.ln() - beta * y
                })
            })
        }),
        ("gamma_lpdf", |_, a| {
            sum_each(a, |p| {
                let (y, alpha, beta) = (p[0], p[1], p[2]);
                positive("alpha", alpha)?;
                positive("beta", beta)?;
                if y < 0.0 {
                    return Ok(f64::NEG_INFINITY);
                }
                Ok(alpha * beta.ln() - libm::lgamma(alpha) + xlogy(alpha - 1.0, y) - beta * y)
            })
        }),
        ("beta_lpdf", |_, a| {
            sum_each(a, |p| {
                let (y, x, z) = (p[0], p[1], p[2]);
                positive("a", x)?;
                positive("b", z)?;
                if !(0.0..=1.0).contains(&y) {
                    return Ok(f64::NEG_INFINITY);
                }
                Ok(xlogy(x - 1.0, y) + xlog1my(z - 1.0, y) - lbeta(x, z))
            })
        }),
        ("lognormal_lpdf", |_, a| {
            sum_each(a, |p| {
                let (y, mu, sigma) = (p[0], p[1], p[2]);
                positive("sigma", sigma)?;
                if y <= 0.0 {
                    return Ok(f64::NEG_INFINITY);
                }
                let z = (y.ln() - mu) / sigma;
                Ok(-y.ln() - sigma.ln() - HALF_LN_2PI - 0.5 * z * z)
            })
        }),
        ("cauchy_lpdf", |_, a| {
            sum_each(a, |p| {
                positive("sigma", p[2])?;
                let z = (p[0] - p[1]) / p[2];
                Ok(-PI.ln() - p[2].ln() - (z * z).ln_1p())
            })
        }),
        ("double_exponential_lpdf", |_, a| {
            sum_each(a, |p| {
                positive("sigma", p[2])?;
                Ok(-LN_2 - p[2].ln() - (p[0] - p[1]).abs() / p[2])
            })
        }),
        ("student_t_lpdf", |_, a| {
            sum_each(a, |p| {
                let (y, nu, mu, sigma) = (p[0], p[1], p[2], p[3]);
                positive("nu", nu)?;
                positive("sigma", sigma)?;
                let z = (y - mu) / sigma;
                Ok(libm::lgamma((nu + 1.0) / 2.0)
                    - libm::lgamma(nu / 2.0)
                    - 0.5 * (nu * PI).ln()
                    - sigma.ln()
                    - (nu + 1.0) / 2.0 * (z * z / nu).ln_1p())
            })
        }),
        ("uniform_lpdf", |_, a| {
            sum_each(a, |p| {
                let (y, lo, hi) = (p[0], p[1], p[2]);
                if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                    return Err(format!("needs finite a < b, got a = {lo}, b = {hi}"));
                }
                Ok(if (lo..=hi).contains(&y) {
                    -(hi - lo).ln()
                } else {
                    f64::NEG_INFINITY
                })
            })
        }),
        ("logistic_lpdf", |_, a| {
            sum_each(a, |p| {
                positive("sigma", p[2])?;
                let z = (p[0] - p[1]) / p[2];
                Ok(-z - p[2].ln() - 2.0 * log1p_exp(-z))
            })
        }),
        ("multi_normal_lpdf", |_, a| {
            let l = cholesky(mat_arg(&a[2])?)?;
            mvn_cholesky_lpdf(vec_arg(&a[0])?, vec_arg(&a[1])?, &l)
        }),
        ("multi_normal_cholesky_lpdf", |_, a| {
            let l = square_arg(&a[2])?.lower_triangle();
            mvn_cholesky_lpdf(vec_arg(&a[0])?, vec_arg(&a[1])?, &l)
        }),
        ("normal_lcdf", |_, a| {
            sum_each(a, |p| {
                positive("sigma", p[2])?;
                Ok(log_phi((p[0] - p[1]) / p[2]))
            })
        }),
        ("normal_lccdf", |_, a| {
            sum_each(a, |p| {
                positive("sigma", p[2])?;
                Ok(log_phi(-(p[0] - p[1]) / p[2]))
            })
        }),
        ("std_normal_lcdf", |_, a| sum_each(a, |p| Ok(log_phi(p[0])))),
        ("std_normal_lccdf", |_, a| {
            sum_each(a, |p| Ok(log_phi(-p[0])))
        }),
        ("bernoulli_lpmf", |_, a| {
            let (y, theta) = (int_arg(&a[0])?, real_arg(&a[1])?);
            probability("theta", theta)?;
            Ok(Value::Real(match y {
                1 => theta.ln(),
                0 => log1m(theta),
                _ => f64::NEG_INFINITY,
            }))
        }),
        ("bernoulli_logit_lpmf", |_, a| {
            let (y, alpha) = (int_arg(&a[0])?, real_arg(&a[1])?);
            Ok(Value::Real(match y {
                1 => -log1p_exp(-alpha),
                0 => -log1p_exp(alpha),
                _ => f64::NEG_INFINITY,
            }))
        }),
        ("poisson_lpmf", |_, a| {
            let (n, lambda) = (int_arg(&a[0])?, real_arg(&a[1])?);
            if lambda.is_nan() || lambda < 0.0 {
                return Err(format!("lambda must be non-negative, got {lambda}"));
            }
            Ok(Value::Real(if n < 0 {
                f64::NEG_INFINITY
            } else {
                xlogy(n as f64, lambda) - lambda - libm::lgamma(n as f64 + 1.0)
            }))
        }),
        ("poisson_log_lpmf", |_, a| {
            let (n, alpha) = (int_arg(&a[0])?, real_arg(&a[1])?);
            Ok(Value::Real(if n < 0 {
                f64::NEG_INFINITY
            } else if alpha == f64::NEG_INFINITY {
                if n == 0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                n as f64 * alpha - alpha.exp() - libm::lgamma(n as f64 + 1.0)
            }))
        }),
        ("binomial_lpmf", |_, a| {
            let (n, big_n, theta) = (int_arg(&a[0])?, int_arg(&a[1])?, real_arg(&a[2])?);
            if big_n < 0 {
                return Err(format!("N must be non-negative, got {big_n}"));
            }
            probability("theta", theta)?;
            Ok(Value::Real(if n < 0 || n > big_n {
                f64::NEG_INFINITY
            } else {
                let (n, big_n) = (n as f64, big_n as f64);
                lchoose(big_n, n) + xlogy(n, theta) + xlog1my(big_n - n, theta)
            }))
        }),
        ("categorical_lpmf", |_, a| {
            let theta = vec_arg(&a[1])?;
            simplex(theta)?;
            let k = int_arg(&a[0])?;
            Ok(Value::Real(if k < 1 || k as usize > theta.len() {
                f64::NEG_INFINITY
            } else {
                theta[k as usize - 1].ln()
            }))
        }),
        ("categorical_logit_lpmf", |_, a| {
            let beta = vec_arg(&a[1])?;
            let k = int_arg(&a[0])?;
            Ok(Value::Real(if k < 1 || k as usize > beta.len() {
                f64::NEG_INFINITY
            } else {
                beta[k as usize - 1] - log_sum_exp(beta.as_slice())
            }))
        }),
    ];
    for &(name, call) in rows {
        entries.push((name, Native::Function(call)));
    }
}

// ---- argument checks -----------------------------------------------------------------

fn int_arg(v: &Value) -> Result<i64, String> {
    v.as_int()
        .ok_or_else(|| format!("expected an int, got {}", v.type_name()))
}

fn positive(name: &str, x: f64) -> Result<(), String> {
    if x > 0.0 && x.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be positive and finite, got {x}"))
    }
}

fn probability(name: &str, x: f64) -> Result<(), String> {
    if (0.0..=1.0).contains(&x) {
        Ok(())
    } else {
        Err(format!("{name} must be in [0, 1], got {x}"))
    }
}

/// 1-based position in 1..=n -> 0-based.
fn position(v: &Value, n: usize, what: &str) -> Result<usize, String> {
    let k = int_arg(v)?;
    if k >= 1 && (k as usize) <= n {
        Ok(k as usize - 1)
    } else {
        Err(format!("{what} {k} is out of range 1..{n}"))
    }
}

fn fits(n: usize, len: usize) -> Result<(), String> {
    if n <= len {
        Ok(())
    } else {
        Err(format!("{n} elements asked for, the vector has {len}"))
    }
}

fn square_arg(v: &Value) -> Result<&DMatrix<f64>, String> {
    let m = mat_arg(v)?;
    if m.is_square() {
        Ok(m)
    } else {
        Err(format!(
            "needs a square matrix, got {} x {}",
            m.nrows(),
            m.ncols()
        ))
    }
}

fn rhs_rows(m: &DMatrix<f64>, rows: usize) -> Result<(), String> {
    if m.nrows() == rows {
        Ok(())
    } else {
        Err(format!(
            "a {} x {} matrix with {rows} elements on the other side",
            m.nrows(),
            m.ncols()
        ))
    }
}

/// Symmetric to Stan's tolerance (1e-8, relative to the larger element).
fn symmetric(m: &DMatrix<f64>) -> Result<(), String> {
    if !m.is_square() {
        return Err(format!(
            "needs a square matrix, got {} x {}",
            m.nrows(),
            m.ncols()
        ));
    }
    for i in 0..m.nrows() {
        for j in i + 1..m.ncols() {
            let (x, y) = (m[(i, j)], m[(j, i)]);
            let gap = (x - y).abs();
            if gap.is_nan() || gap > 1e-8 * x.abs().max(y.abs()).max(1.0) {
                return Err(format!(
                    "the matrix is not symmetric: m[{}, {}] = {x}, m[{}, {}] = {y}",
                    i + 1,
                    j + 1,
                    j + 1,
                    i + 1
                ));
            }
        }
    }
    Ok(())
}

fn simplex(theta: &DVector<f64>) -> Result<(), String> {
    if let Some(x) = theta.iter().find(|x| !x.is_finite() || **x < 0.0) {
        return Err(format!("theta must be non-negative and finite, has {x}"));
    }
    let total: f64 = theta.sum();
    if (total - 1.0).abs() > 1e-8 {
        return Err(format!("theta must sum to 1, sums to {total}"));
    }
    Ok(())
}

// ---- shared computations ---------------------------------------------------------------

/// A vector argument as one column, a real as one element.
fn column(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Vector(x) => Ok(x.as_slice().to_vec()),
        other => Ok(vec![real_arg(other)?]),
    }
}

/// A matrix, or a vector as a one-column matrix.
fn as_matrix(v: &Value) -> Result<DMatrix<f64>, String> {
    match v {
        Value::Matrix(m) => Ok(m.clone()),
        Value::Vector(x) => Ok(DMatrix::from_column_slice(x.len(), 1, x.as_slice())),
        other => Err(format!(
            "expected a matrix or vector, got {}",
            other.type_name()
        )),
    }
}

fn variance(x: &[f64]) -> Result<f64, String> {
    if x.len() < 2 {
        return Err(format!("needs at least 2 elements, got {}", x.len()));
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    Ok(x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0))
}

fn sorted(v: &DVector<f64>, descending: bool) -> Result<Value, String> {
    if v.iter().any(|x| x.is_nan()) {
        return Err("cannot sort a vector containing NaN".into());
    }
    let mut xs = v.as_slice().to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    if descending {
        xs.reverse();
    }
    Ok(Value::Vector(DVector::from_vec(xs)))
}

fn choose(n: i64, k: i64) -> Result<i64, String> {
    if n < 0 || k < 0 {
        return Err(format!(
            "n and k must be non-negative, got n = {n}, k = {k}"
        ));
    }
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut r: i128 = 1;
    for i in 1..=k as i128 {
        // r * (n - k + i) / i is exact at every step (a binomial coefficient)
        r = r * (n as i128 - k as i128 + i) / i;
        if r > i64::MAX as i128 {
            return Err(format!("choose({n}, {k}) overflows an int"));
        }
    }
    Ok(r as i64)
}

/// L * L' = m, checking symmetry first (nalgebra reads one triangle only).
fn cholesky(m: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    symmetric(m)?;
    m.clone()
        .cholesky()
        .map(|c| c.l())
        .ok_or_else(|| "the matrix is not positive definite".to_string())
}

/// Eigenvalues ascending, with their eigenvectors as columns in that order.
fn eigen_sym(m: &DMatrix<f64>) -> Result<(DVector<f64>, DMatrix<f64>), String> {
    symmetric(m)?;
    let e = m.clone().symmetric_eigen();
    let mut order: Vec<usize> = (0..e.eigenvalues.len()).collect();
    order.sort_by(|&i, &j| {
        e.eigenvalues[i]
            .partial_cmp(&e.eigenvalues[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values = DVector::from_iterator(order.len(), order.iter().map(|&i| e.eigenvalues[i]));
    let vectors = DMatrix::from_fn(m.nrows(), order.len(), |r, c| e.eigenvectors[(r, order[c])]);
    Ok((values, vectors))
}

/// `m \ b` for a vector or matrix b, with the size checked first.
fn solve(
    m: &DMatrix<f64>,
    b: &Value,
    f: impl Fn(&DMatrix<f64>) -> Option<DMatrix<f64>>,
) -> Option<Result<Value, String>> {
    let rhs = match as_matrix(b) {
        Ok(rhs) => rhs,
        Err(e) => return Some(Err(e)),
    };
    if let Err(e) = rhs_rows(m, rhs.nrows()) {
        return Some(Err(e));
    }
    let x = f(&rhs)?;
    Some(Ok(match b {
        Value::Vector(_) => Value::Vector(x.column(0).into_owned()),
        _ => Value::Matrix(x),
    }))
}

fn softmax(x: &DVector<f64>) -> Result<DVector<f64>, String> {
    if x.iter().any(|v| v.is_nan()) {
        return Err("beta contains NaN".into());
    }
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return Err("every element of beta is -inf".into());
    }
    let w = x.map(|v| (v - m).exp());
    let total = w.sum();
    Ok(w / total)
}

/// The 1-based category a uniform u falls in, through the cumulative sum.
/// Rounding can leave the total just under u: the last category with any
/// weight takes it.
fn pick(w: &[f64], u: f64) -> i64 {
    let mut total = 0.0;
    for (k, &x) in w.iter().enumerate() {
        total += x;
        if u < total {
            return k as i64 + 1;
        }
    }
    w.iter().rposition(|&x| x > 0.0).map_or(w.len(), |k| k + 1) as i64
}

fn poisson(c: &mut Ctx<'_, '_>, lambda: f64) -> Result<Value, String> {
    if lambda.is_nan() || lambda < 0.0 {
        return Err(format!("lambda must be non-negative, got {lambda}"));
    }
    if lambda >= (1u64 << 30) as f64 {
        return Err(format!("lambda must be below 2^30, got {lambda}"));
    }
    if lambda == 0.0 {
        return Ok(Value::Int(0));
    }
    let p = Poisson::new(lambda).map_err(|e| e.to_string())?;
    Ok(Value::Int(c.rng()?.sample(p) as i64))
}

fn normals(rng: &mut ChainRng, n: usize) -> DVector<f64> {
    DVector::from_iterator(n, (0..n).map(|_| rng.sample::<f64, _>(StandardNormal)))
}

/// One draw per broadcast element, in index order: a real if every
/// argument is a real, else a vector.
fn draw_each(
    c: &mut Ctx<'_, '_>,
    a: &[Value],
    mut f: impl FnMut(&mut ChainRng, &[f64]) -> Result<f64, String>,
) -> Result<Value, String> {
    let n = broadcast_len(a)?;
    let rng = c.rng()?;
    let mut p = vec![0.0; a.len()];
    let mut one = |i: usize, p: &mut Vec<f64>| {
        for (k, v) in a.iter().enumerate() {
            p[k] = elem(v, i);
        }
        f(rng, p)
    };
    match n {
        None => one(0, &mut p).map(Value::Real),
        Some(n) => {
            let mut out = DVector::zeros(n);
            for i in 0..n {
                out[i] = one(i, &mut p)?;
            }
            Ok(Value::Vector(out))
        }
    }
}

/// A density summed over broadcast elements.
fn sum_each(a: &[Value], f: fn(&[f64]) -> Result<f64, String>) -> Result<Value, String> {
    let n = broadcast_len(a)?.unwrap_or(1);
    let mut p = vec![0.0; a.len()];
    let mut total = 0.0;
    for i in 0..n {
        for (k, v) in a.iter().enumerate() {
            p[k] = elem(v, i);
        }
        total += f(&p)?;
    }
    Ok(Value::Real(total))
}

fn mvn_cholesky_lpdf(
    y: &DVector<f64>,
    mu: &DVector<f64>,
    l: &DMatrix<f64>,
) -> Result<Value, String> {
    same_len("multi_normal", y, mu)?;
    rhs_rows(l, y.len())?;
    if let Some(i) = (0..l.nrows()).find(|&i| l[(i, i)].is_nan() || l[(i, i)] <= 0.0) {
        return Err(format!(
            "the Cholesky factor needs a positive diagonal, has {} at [{}, {}]",
            l[(i, i)],
            i + 1,
            i + 1
        ));
    }
    let r = l
        .solve_lower_triangular(&(y - mu))
        .ok_or("the Cholesky factor is singular")?;
    let log_det: f64 = (0..l.nrows()).map(|i| l[(i, i)].ln()).sum();
    Ok(Value::Real(
        -(y.len() as f64) * HALF_LN_2PI - log_det - 0.5 * r.norm_squared(),
    ))
}

// ---- special functions ---------------------------------------------------------------------

/// c * log(y), with 0 * log(0) = 0 (a density's boundary term).
fn xlogy(c: f64, y: f64) -> f64 {
    if c == 0.0 && !y.is_nan() {
        0.0
    } else {
        c * y.ln()
    }
}

/// c * log(1 - y), with 0 * log(0) = 0.
fn xlog1my(c: f64, y: f64) -> f64 {
    if c == 0.0 && !y.is_nan() {
        0.0
    } else {
        c * log1m(y)
    }
}

fn lbeta(a: f64, b: f64) -> f64 {
    libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)
}

fn lchoose(n: f64, k: f64) -> f64 {
    libm::lgamma(n + 1.0) - libm::lgamma(k + 1.0) - libm::lgamma(n - k + 1.0)
}

fn phi(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

/// log Phi(z), accurate in both tails: log1p of the upper tail above 0,
/// erfc in the middle, the asymptotic series where erfc underflows.
fn log_phi(z: f64) -> f64 {
    if z.is_nan() {
        f64::NAN
    } else if z > 0.0 {
        (-0.5 * libm::erfc(z / SQRT_2)).ln_1p()
    } else if z > -37.5 {
        (0.5 * libm::erfc(-z / SQRT_2)).ln()
    } else if z == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        let w = 1.0 / (z * z);
        -0.5 * z * z - HALF_LN_2PI - (-z).ln() + (-w + 3.0 * w * w - 15.0 * w * w * w).ln_1p()
    }
}

/// Standard normal quantile: Acklam's rational approximation (relative
/// error 1.15e-9), then one Halley step on Phi, which brings it to about
/// machine precision. Computed on the lower half; the upper by symmetry.
fn inv_phi(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];
    const LOW: f64 = 0.02425;
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        // 1 - p is exact here (Sterbenz), and the refinement below stays
        // accurate only where Phi(x) is not close to 1.
        return -inv_phi(1.0 - p);
    }
    let tail = |q: f64| {
        let q = (-2.0 * q.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let x = if p < LOW {
        tail(p)
    } else if p <= 1.0 - LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail(1.0 - p)
    };
    let e = phi(x) - p;
    let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

/// ψ(x): reflection below 0, the recurrence ψ(x) = ψ(x + 1) - 1/x up to
/// x >= 15 (where the series is exact to rounding), then the asymptotic series.
fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY || (x <= 0.0 && x == x.floor()) {
        return f64::NAN;
    }
    if x < 0.0 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let (mut x, mut r) = (x, 0.0);
    while x < 15.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    r + x.ln()
        - 0.5 / x
        - f * (1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0 - f / 132.0))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        a == b || (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-300)
    }

    #[test]
    fn special_functions_against_known_values() {
        // Phi and its inverse
        assert!(close(phi(1.959963984540054), 0.975, 1e-15));
        for p in [1e-300, 1e-20, 0.001, 0.02425, 0.3, 0.5, 0.9, 0.999] {
            assert!(close(phi(inv_phi(p)), p, 1e-13), "inv_Phi({p})");
        }
        assert_eq!(inv_phi(0.5), 0.0);
        // log Phi in both tails
        assert!(close(log_phi(-40.0), -804.6084420137538, 1e-12));
        assert!(close(log_phi(10.0), -7.619853024160527e-24, 1e-9));
        assert!(close(log_phi(0.0), -LN_2, 1e-15));
        // digamma: ψ(1) = -γ, ψ(1/2) = -γ - 2 ln 2, reflection
        let gamma = 0.5772156649015329;
        assert!(close(digamma(1.0), -gamma, 1e-14));
        assert!(close(digamma(0.5), -gamma - 2.0 * LN_2, 1e-14));
        assert!(close(digamma(-0.5), 0.03648997397857652, 1e-12));
        assert!(digamma(-2.0).is_nan());
        assert!(close(lchoose(10.0, 3.0), 120f64.ln(), 1e-13));
    }

    #[test]
    fn exact_integers() {
        assert_eq!(choose(10, 3), Ok(120));
        assert_eq!(choose(3, 5), Ok(0));
        assert_eq!(choose(66, 33), Ok(7219428434016265740));
        assert!(choose(68, 34).is_err());
        assert!(choose(-1, 0).is_err());
    }

    #[test]
    fn categorical_pick_inverts_the_cumulative_sum() {
        let w = [0.2, 0.0, 0.5, 0.3];
        assert_eq!(pick(&w, 0.0), 1);
        assert_eq!(pick(&w, 0.2), 3);
        assert_eq!(pick(&w, 0.69), 3);
        assert_eq!(pick(&w, 0.7), 4);
        // rounding left the total below u: the last category with weight
        assert_eq!(pick(&[0.5, 0.49999999999, 0.0], 0.999999999999), 2);
    }
}
