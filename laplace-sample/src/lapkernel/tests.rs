//! Phase 2.3 tests: every built-in (values and RNG consumption) and the
//! interpreter's semantics.

use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use super::builtins::Registry;
use super::check::{check, Checked};
use super::interp::{Interp, Outcome};
use super::natives::{Ctx, Native, Natives};
use super::parse::parse;
use super::types::Value;
use crate::kernel::ChainRng;
use crate::state::{ChainState, Evaluator};
use crate::target::analytic::DiagGaussian;
use crate::target::Target;

fn rng(seed: u64) -> ChainRng {
    ChainRng::seed_from_u64(seed)
}

fn v(xs: &[f64]) -> Value {
    Value::Vector(DVector::from_column_slice(xs))
}

fn real(x: &Value) -> f64 {
    x.as_real().unwrap()
}

fn vec_of(x: &Value) -> Vec<f64> {
    x.as_vector().unwrap().iter().copied().collect()
}

fn target() -> DiagGaussian {
    DiagGaussian {
        mu: vec![1.0, -1.0, 0.5],
        sd: vec![1.0, 2.0, 0.5],
    }
}

/// Call a registry function with a full chain context.
fn call(name: &str, args: &[Value], seed: u64) -> (Value, ChainRng, u64) {
    let t = target();
    let mut ev = Evaluator::new(&t);
    let state = ChainState::at(vec![0.2, 0.3, -0.4], &mut ev).unwrap();
    let mut r = rng(seed);
    let metric = DVector::from_column_slice(&[1.0, 4.0, 0.25]);
    let before = ev.n_grad();
    let out = {
        let mut ctx = Ctx {
            dim: 3,
            state: Some(&state),
            ev: Some(&mut ev),
            rng: Some(&mut r),
            metric: Some(&metric),
        };
        match Natives::standard().get(name) {
            Some(Native::Function(f)) => f(&mut ctx, args).unwrap(),
            Some(Native::Value(get)) => get(&ctx).unwrap(),
            None => panic!("{name} is not callable"),
        }
    };
    (out, r, ev.n_grad() - before)
}

// ---- registry ------------------------------------------------------------------

#[test]
fn registry_entries_are_unique_and_documented() {
    let reg = Registry::standard();
    let mut names: Vec<&str> = reg.entries().iter().map(|e| e.name).collect();
    for e in reg.entries() {
        assert!(!e.doc.is_empty() && !e.rng.is_empty(), "{}", e.name);
    }
    let n = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), n, "duplicate registry names");
    for required in [
        "q",
        "lp",
        "grad",
        "dim",
        "metric",
        "log_density",
        "log_density_grad",
        "leapfrog",
        "propose",
        "move",
        "std_normal_rng",
        "normal_rng",
        "uniform_rng",
        "normal_lpdf",
        "std_normal_lpdf",
        "dot_product",
        "sum",
        "exp",
        "log",
        "sqrt",
        "square",
        "rep_vector",
    ] {
        assert!(reg.get(required).is_some(), "missing {required}");
    }
    // Post-v1 candidates are deliberately absent (open-questions.md).
    for later in [
        "hessian",
        "hessian_vector_product",
        "is_warmup",
        "iteration",
    ] {
        assert!(reg.get(later).is_none(), "{later} is post-v1");
    }
}

// ---- context -------------------------------------------------------------------

#[test]
fn context_values() {
    let (q, ..) = call("q", &[], 1);
    assert_eq!(vec_of(&q), vec![0.2, 0.3, -0.4]);
    let t = target();
    let mut g = vec![0.0; 3];
    let lp_true = t.log_density_gradient(&[0.2, 0.3, -0.4], &mut g).unwrap();
    assert_eq!(real(&call("lp", &[], 1).0), lp_true);
    assert_eq!(vec_of(&call("grad", &[], 1).0), g);
    assert_eq!(call("dim", &[], 1).0, Value::Int(3));
    assert_eq!(vec_of(&call("metric", &[], 1).0), vec![1.0, 4.0, 0.25]);
}

#[test]
fn metric_defaults_to_ones_without_an_adapted_one() {
    let Some(Native::Value(get)) = Natives::standard().get("metric") else {
        panic!()
    };
    let ctx = Ctx {
        dim: 2,
        state: None,
        ev: None,
        rng: None,
        metric: None,
    };
    assert_eq!(vec_of(&get(&ctx).unwrap()), vec![1.0, 1.0]);
}

#[test]
fn log_density_and_gradient_go_through_the_cache() {
    let t = target();
    let x = [1.0, 0.0, 0.0];
    let mut g = vec![0.0; 3];
    let lp = t.log_density_gradient(&x, &mut g).unwrap();
    // Density only: no gradient is computed or counted.
    let (out, _, grads) = call("log_density", &[v(&x)], 1);
    assert_eq!((real(&out), grads), (lp, 0));
    let (out, _, _) = call("log_density_grad", &[v(&x)], 1);
    let Value::Tuple(parts) = out else { panic!() };
    assert_eq!(real(&parts[0]), lp);
    assert_eq!(vec_of(&parts[1]), g);
    // The current point is already cached: free.
    let (_, _, grads) = call("log_density", &[v(&[0.2, 0.3, -0.4])], 1);
    assert_eq!(grads, 0);
}

#[test]
fn leapfrog_matches_a_hand_written_integrator() {
    let t = target();
    let (q0, p0, eps, steps, m) = ([0.2, 0.3, -0.4], [0.5, -1.0, 0.1], 0.1, 7, [1.0, 4.0, 0.25]);
    let (out, _, grads) = call(
        "leapfrog",
        &[v(&q0), v(&p0), Value::Real(eps), Value::Int(steps), v(&m)],
        1,
    );
    let (mut q, mut p) = (q0.to_vec(), p0.to_vec());
    let mut g = vec![0.0; 3];
    let mut lp = t.log_density_gradient(&q, &mut g).unwrap();
    for _ in 0..steps {
        for i in 0..3 {
            p[i] += 0.5 * eps * g[i];
            q[i] += eps * m[i] * p[i];
        }
        lp = t.log_density_gradient(&q, &mut g).unwrap();
        for i in 0..3 {
            p[i] += 0.5 * eps * g[i];
        }
    }
    let Value::Tuple(r) = out else { panic!() };
    assert_eq!(vec_of(&r[0]), q);
    assert_eq!(vec_of(&r[1]), p);
    assert_eq!(real(&r[2]), lp);
    assert_eq!(vec_of(&r[3]), g);
    assert_eq!(grads, 7, "start point cached, one gradient per step");
}

// ---- random numbers: values and consumption order -------------------------------

#[test]
fn std_normal_rng_draws_in_index_order() {
    let (out, mut after, _) = call("std_normal_rng", &[Value::Int(4)], 9);
    let mut r = rng(9);
    let expect: Vec<f64> = (0..4).map(|_| r.sample(StandardNormal)).collect();
    assert_eq!(vec_of(&out), expect);
    assert_eq!(
        after.random::<u64>(),
        r.random::<u64>(),
        "stream position after the call"
    );
    let (out, mut after, _) = call("std_normal_rng", &[], 9);
    let mut r = rng(9);
    assert_eq!(real(&out), r.sample::<f64, _>(StandardNormal));
    assert_eq!(after.random::<u64>(), r.random::<u64>());
    let (out, mut after, _) = call("std_normal_rng", &[Value::Int(0)], 9);
    assert_eq!(vec_of(&out), Vec::<f64>::new());
    assert_eq!(after.random::<u64>(), rng(9).random::<u64>(), "zero draws");
}

#[test]
fn normal_rng_is_mu_plus_sigma_z() {
    let (out, mut after, _) = call("normal_rng", &[Value::Real(2.0), Value::Real(3.0)], 4);
    let mut r = rng(4);
    let z: f64 = r.sample(StandardNormal);
    assert_eq!(real(&out), 2.0 + 3.0 * z);
    assert_eq!(after.random::<u64>(), r.random::<u64>());
    let (out, mut after, _) = call("normal_rng", &[v(&[0.0, 10.0]), v(&[1.0, 2.0])], 4);
    let mut r = rng(4);
    let z: Vec<f64> = (0..2).map(|_| r.sample(StandardNormal)).collect();
    assert_eq!(vec_of(&out), vec![z[0], 10.0 + 2.0 * z[1]]);
    assert_eq!(after.random::<u64>(), r.random::<u64>());
}

#[test]
fn uniform_rng_is_one_uniform() {
    let (out, mut after, _) = call("uniform_rng", &[Value::Real(-1.0), Value::Real(3.0)], 5);
    let mut r = rng(5);
    let u: f64 = r.random();
    assert_eq!(real(&out), -1.0 + 4.0 * u);
    assert_eq!(after.random::<u64>(), r.random::<u64>());
}

#[test]
fn deterministic_builtins_draw_nothing() {
    for (name, args) in [
        (
            "normal_lpdf",
            vec![Value::Real(0.1), Value::Real(0.0), Value::Real(1.0)],
        ),
        ("std_normal_lpdf", vec![v(&[0.1, 0.2])]),
        ("dot_product", vec![v(&[1.0, 2.0]), v(&[3.0, 4.0])]),
        ("sum", vec![v(&[1.0, 2.0])]),
        ("exp", vec![Value::Real(1.0)]),
        ("rep_vector", vec![Value::Real(1.0), Value::Int(2)]),
        ("log_density", vec![v(&[0.0, 0.0, 0.0])]),
        (
            "leapfrog",
            vec![
                v(&[0.0; 3]),
                v(&[1.0; 3]),
                Value::Real(0.1),
                Value::Int(2),
                v(&[1.0; 3]),
            ],
        ),
    ] {
        let (_, mut after, _) = call(name, &args, 6);
        assert_eq!(
            after.random::<u64>(),
            rng(6).random::<u64>(),
            "{name} touched the RNG"
        );
    }
}

// ---- densities and math -----------------------------------------------------------

#[test]
fn normal_densities_include_constants() {
    let c = -0.5 * (2.0 * std::f64::consts::PI).ln();
    let (out, ..) = call(
        "normal_lpdf",
        &[Value::Real(1.0), Value::Real(0.5), Value::Real(2.0)],
        1,
    );
    assert!((real(&out) - (c - 2f64.ln() - 0.5 * 0.0625)).abs() < 1e-14);
    // vector y, vector mu, scalar sigma: summed
    let (out, ..) = call(
        "normal_lpdf",
        &[v(&[1.0, 2.0]), v(&[0.0, 0.0]), Value::Real(1.0)],
        1,
    );
    assert!((real(&out) - (2.0 * c - 0.5 - 2.0)).abs() < 1e-14);
    let (out, ..) = call("std_normal_lpdf", &[v(&[1.0, -1.0])], 1);
    assert!((real(&out) - (2.0 * c - 1.0)).abs() < 1e-14);
}

#[test]
fn math_functions() {
    assert_eq!(real(&call("exp", &[Value::Int(0)], 1).0), 1.0);
    assert_eq!(
        vec_of(&call("log", &[v(&[1.0, std::f64::consts::E])], 1).0),
        vec![0.0, 1.0]
    );
    assert_eq!(
        vec_of(&call("sqrt", &[v(&[4.0, 9.0])], 1).0),
        vec![2.0, 3.0]
    );
    assert_eq!(real(&call("square", &[Value::Real(-3.0)], 1).0), 9.0);
    assert_eq!(real(&call("sum", &[v(&[1.0, 2.0, 3.5])], 1).0), 6.5);
    assert_eq!(
        real(&call("dot_product", &[v(&[1.0, 2.0]), v(&[3.0, 4.0])], 1).0),
        11.0
    );
    assert_eq!(
        vec_of(&call("rep_vector", &[Value::Int(2), Value::Int(3)], 1).0),
        vec![2.0; 3]
    );
}

// ---- interpreter -------------------------------------------------------------------

fn compile(src: &str, dim: usize) -> Checked {
    let k = parse("t.lapkernel", src).unwrap_or_else(|e| panic!("{e}"));
    check("t.lapkernel", &k, Some(dim), &Registry::standard()).unwrap_or_else(|e| panic!("{e:#?}"))
}

/// Run `steps` steps on the analytic target at q = (0.2, 0.3, -0.4);
/// returns the state variables after the last and the outcomes.
fn run(src: &str, steps: usize, seed: u64) -> (Vec<Value>, Vec<Outcome>) {
    let checked = compile(src, 3);
    let interp = Interp::new(std::sync::Arc::new(checked));
    let tuning = interp.tuning_defaults(3).unwrap();
    let t = target();
    let mut ev = Evaluator::new(&t);
    let state = ChainState::at(vec![0.2, 0.3, -0.4], &mut ev).unwrap();
    let mut r = rng(seed);
    let mut ctx = Ctx {
        dim: 3,
        state: Some(&state),
        ev: Some(&mut ev),
        rng: Some(&mut r),
        metric: None,
    };
    let mut vars = interp.init_state(&mut ctx, &tuning).unwrap();
    let outcomes = (0..steps)
        .map(|_| interp.step(&mut ctx, &tuning, &mut vars).unwrap())
        .collect();
    (vars, outcomes)
}

fn run_err(src: &str) -> String {
    let checked = compile(src, 3);
    let interp = Interp::new(std::sync::Arc::new(checked));
    let tuning = interp.tuning_defaults(3).unwrap();
    let t = target();
    let mut ev = Evaluator::new(&t);
    let state = ChainState::at(vec![0.0; 3], &mut ev).unwrap();
    let mut r = rng(1);
    let mut ctx = Ctx {
        dim: 3,
        state: Some(&state),
        ev: Some(&mut ev),
        rng: Some(&mut r),
        metric: None,
    };
    let mut vars = interp.init_state(&mut ctx, &tuning).unwrap();
    interp
        .step(&mut ctx, &tuning, &mut vars)
        .unwrap_err()
        .to_string()
}

#[test]
fn arithmetic_and_precedence() {
    let (vars, _) = run(
        "kernel k { state { real a; real b; int c; int d; real e; real f; }
           step { a = 1 + 2 * 3 ^ 2; b = -2 ^ 2; c = 7 / 2; d = 7 % 3; e = 7.0 / 2; f = 0.1 ^ 2; move(q); } }",
        1,
        1,
    );
    assert_eq!(
        vars,
        vec![
            Value::Real(19.0),
            Value::Real(-4.0),
            Value::Int(3),
            Value::Int(1),
            Value::Real(3.5),
            Value::Real(0.1 * 0.1)
        ]
    );
}

#[test]
fn vector_arithmetic_is_elementwise() {
    let (vars, _) = run(
        "kernel k { state { vector[3] a; vector[3] b; real s; }
           step { a = q + 2 * grad; b = (q .* q) ./ (q - 1); s = dot_product(a, b); move(q); } }",
        1,
        1,
    );
    let q = [0.2, 0.3, -0.4];
    let t = target();
    let mut g = vec![0.0; 3];
    t.log_density_gradient(&q, &mut g).unwrap();
    let a: Vec<f64> = (0..3).map(|i| q[i] + 2.0 * g[i]).collect();
    let b: Vec<f64> = (0..3).map(|i| (q[i] * q[i]) / (q[i] - 1.0)).collect();
    assert_eq!(vec_of(&vars[0]), a);
    assert_eq!(vec_of(&vars[1]), b);
    assert_eq!(real(&vars[2]), a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

#[test]
fn control_flow_functions_and_tuples() {
    let (vars, out) = run(
        "kernel k {
           state { real acc; int n; tuple(real, int) t; }
           functions {
             real twice(real x) { return 2 * x; }
             int fact(int k) { if (k <= 1) return 1; else return k * fact(k - 1); }
           }
           step {
             acc = 0;
             for (i in 1:4) { acc += twice(i); }
             n = fact(5);
             while (n > 100) { n -= 7; }
             (real x, int m) = (acc, n);
             t = (x + 1, m);
             if (t.2 > 0 && acc > 1) move(q); else propose(q, 0);
           } }",
        1,
        1,
    );
    assert_eq!(vars[0], Value::Real(20.0));
    assert_eq!(vars[1], Value::Int(99));
    assert_eq!(
        vars[2],
        Value::Tuple(vec![Value::Real(21.0), Value::Int(99)])
    );
    assert!(matches!(out[0], Outcome::Move { .. }));
}

#[test]
fn short_circuit_skips_the_right_hand_side() {
    // The right-hand side would draw a random number; && must not evaluate it.
    let (vars, _) = run(
        "kernel k { state { real x; } step { if (0 && std_normal_rng() > 0) x = 1; x = std_normal_rng(); move(q); } }",
        1,
        3,
    );
    let mut r = rng(3);
    assert_eq!(vars[0], Value::Real(r.sample(StandardNormal)));
}

#[test]
fn state_persists_and_init_runs_once() {
    let (vars, _) = run(
        "kernel k { state { int calls; real first; } init { calls = 100; first = lp; }
           step { calls += 1; move(q); } }",
        5,
        1,
    );
    assert_eq!(vars[0], Value::Int(105));
    let t = target();
    let mut g = vec![0.0; 3];
    assert_eq!(
        vars[1],
        Value::Real(t.log_density_gradient(&[0.2, 0.3, -0.4], &mut g).unwrap())
    );
}

#[test]
fn terminals_return_their_arguments() {
    let (_, out) = run(
        "kernel k { tuning { real eps = 0.5; } step { propose(q + eps, -1.25); } }",
        1,
        1,
    );
    assert_eq!(
        out[0],
        Outcome::Propose {
            q_new: DVector::from_column_slice(&[0.2 + 0.5, 0.3 + 0.5, -0.4 + 0.5]),
            log_correction: -1.25
        }
    );
}

#[test]
fn tuning_defaults_and_bounds() {
    let checked = compile(
        "kernel k { tuning { real<lower=0> eps = 0.1 * dim; vector<lower=0>[dim] m @adapt(metric); int n = dim + 1; }
           step { move(q); } }",
        3,
    );
    let interp = Interp::new(std::sync::Arc::new(checked));
    let t = interp.tuning_defaults(3).unwrap();
    assert!((real(&t[0]) - 0.3).abs() < 1e-15);
    assert_eq!(vec_of(&t[1]), vec![1.0; 3]);
    assert_eq!(t[2], Value::Int(4));
    assert!(interp
        .check_tuning(0, &Value::Real(-1.0), 3)
        .unwrap_err()
        .contains("`eps` must be >= 0"));
    assert_eq!(
        interp.check_tuning(0, &Value::Int(2), 3).unwrap(),
        Value::Real(2.0)
    );
    assert!(interp
        .check_tuning(1, &v(&[1.0, 2.0]), 3)
        .unwrap_err()
        .contains("vector[3]"));
}

#[test]
fn runtime_errors_point_at_the_expression() {
    let e = run_err("kernel k { step {\n  real x = q[4];\n  move(q);\n} }");
    assert_eq!(e, "t.lapkernel:2:12: index 4 out of range 1..3");
    let e = run_err("kernel k { step {\n  int n = 4;\n  vector[n] v = q;\n  move(q);\n} }");
    assert_eq!(
        e,
        "t.lapkernel:3:17: `v` is a vector[4] but the value has 3 elements"
    );
    let e = run_err("kernel k { step {\n  real s = normal_lpdf(q | q, -1);\n  move(q);\n} }");
    assert_eq!(
        e,
        "t.lapkernel:2:12: normal_lpdf: sigma must be positive, got -1"
    );
}

#[test]
fn matrix_columns_without_an_inner_loop() {
    let (vars, _) = run(
        "kernel k {
           state { matrix[dim, 2] y; vector[dim] c2; vector[2] r1; int n; }
           step {
             for (j in 1:cols(y)) {
               y = set_col(y, j, q + j);
             }
             c2 = col(y, 2);
             r1 = row(y, 1);
             n = rows(y) * 10 + cols(rep_matrix(0, 1, 4));
             move(q);
           } }",
        1,
        1,
    );
    let q = [0.2, 0.3, -0.4];
    let col = |j: f64| q.iter().map(|x| x + j).collect::<Vec<_>>();
    let Value::Matrix(y) = &vars[0] else { panic!() };
    assert_eq!(y.column(0).iter().copied().collect::<Vec<_>>(), col(1.0));
    assert_eq!(y.column(1).iter().copied().collect::<Vec<_>>(), col(2.0));
    assert_eq!(vec_of(&vars[1]), col(2.0));
    assert_eq!(vec_of(&vars[2]), vec![q[0] + 1.0, q[0] + 2.0]);
    assert_eq!(vars[3], Value::Int(34));
}

#[test]
fn matrix_function_mistakes() {
    // a size known when checking is checked then
    let k = parse(
        "t.lapkernel",
        "kernel k { step {\n  matrix[dim, 2] y;\n  y = set_col(y, 1, rep_vector(0, 2));\n  move(q);\n} }",
    )
    .unwrap();
    let e = format!(
        "{:?}",
        check("t.lapkernel", &k, Some(3), &Registry::standard()).unwrap_err()
    );
    assert!(
        e.contains("a column of this matrix has 3 elements, the vector has 2"),
        "{e}"
    );
    // an index known only at run time is checked then
    let e = run_err(
        "kernel k { step {\n  matrix[dim, 2] y;\n  int j = 3;\n  vector[dim] c = col(y, j);\n  move(q);\n} }",
    );
    assert_eq!(e, "t.lapkernel:4:19: col: column 3 is out of range 1..2");
}

// ---- standard library ---------------------------------------------------------------

/// Reference values for the deterministic functions, from SciPy and NumPy
/// (the generating script is in the PR that added them). Each must agree
/// to 1e-12 relative, and none may touch the random stream.
#[test]
#[allow(clippy::approx_constant)] // reference data that happens to equal ln 2, log2(10), ...
fn stdlib_matches_scipy() {
    // Generated by SciPy 1.18.1 / NumPy 2.5.5 (scratch script, values pasted).
    let cases: Vec<(&str, Vec<Value>, f64)> = vec![
        ("sin", vec![Value::Real(0.7)], 0.644217687237691),
        ("cos", vec![Value::Real(0.7)], 0.7648421872844885),
        ("tan", vec![Value::Real(0.7)], 0.8422883804630794),
        ("asin", vec![Value::Real(0.3)], 0.3046926540153975),
        ("acos", vec![Value::Real(0.3)], 1.2661036727794992),
        ("atan", vec![Value::Real(2.0)], 1.1071487177940904),
        ("sinh", vec![Value::Real(1.3)], 1.698382437292616),
        ("cosh", vec![Value::Real(1.3)], 1.9709142303266285),
        ("asinh", vec![Value::Real(-2.0)], -1.4436354751788103),
        ("acosh", vec![Value::Real(3.0)], 1.762747174039086),
        ("atanh", vec![Value::Real(0.4)], 0.42364893019360184),
        ("log2", vec![Value::Real(10.0)], 3.321928094887362),
        ("log10", vec![Value::Real(7.0)], 0.8450980400142568),
        ("exp2", vec![Value::Real(3.3)], 9.849155306759329),
        ("cbrt", vec![Value::Real(-27.5)], -3.018405368398843),
        ("floor", vec![Value::Real(-2.5)], -3.0),
        ("ceil", vec![Value::Real(-2.5)], -2.0),
        ("round", vec![Value::Real(2.5)], 3.0),
        ("round", vec![Value::Real(-2.5)], -3.0),
        ("trunc", vec![Value::Real(-2.7)], -2.0),
        ("erf", vec![Value::Real(0.3)], 0.3286267594591274),
        ("erfc", vec![Value::Real(5.0)], 1.5374597944280347e-12),
        ("Phi", vec![Value::Real(-3.0)], 0.0013498980316300933),
        ("Phi", vec![Value::Real(-30.0)], 4.906713927147908e-198),
        ("Phi_approx", vec![Value::Real(0.5)], 0.6916021089764008),
        ("inv_Phi", vec![Value::Real(0.01)], -2.3263478740408408),
        ("inv_Phi", vec![Value::Real(0.999999)], 4.753424308817087),
        ("inv_Phi", vec![Value::Real(1e-250)], -33.79958617269484),
        ("lgamma", vec![Value::Real(0.3)], 1.0957979948180756),
        ("lgamma", vec![Value::Real(100.0)], 359.1342053695754),
        ("tgamma", vec![Value::Real(4.5)], 11.63172839656745),
        ("digamma", vec![Value::Real(3.7)], 1.1671535393615113),
        ("digamma", vec![Value::Real(0.001)], -1000.5755719318103),
        ("digamma", vec![Value::Real(-1.3)], 2.8825405488661677),
        ("inv_cloglog", vec![Value::Real(-1.0)], 0.3077993724446536),
        (
            "pow",
            vec![Value::Real(2.5), Value::Real(1.7)],
            4.747861205827337,
        ),
        (
            "atan2",
            vec![Value::Real(1.0), Value::Real(-1.0)],
            2.356194490192345,
        ),
        (
            "hypot",
            vec![Value::Real(3e+200), Value::Real(4e+200)],
            5e+200,
        ),
        ("fdim", vec![Value::Real(3.0), Value::Real(5.0)], 0.0),
        ("fdim", vec![Value::Real(5.0), Value::Real(3.0)], 2.0),
        ("fmod", vec![Value::Real(-7.5), Value::Real(2.0)], -1.5),
        (
            "lbeta",
            vec![Value::Real(2.5), Value::Real(3.5)],
            -3.3018352699620523,
        ),
        (
            "lchoose",
            vec![Value::Real(10.5), Value::Real(3.2)],
            5.107677968430057,
        ),
        (
            "log_mix",
            vec![Value::Real(0.3), Value::Real(-1.0), Value::Real(-2.0)],
            -1.5842647781563712,
        ),
        (
            "log_mix",
            vec![Value::Real(0.5), Value::Real(-1000.0), Value::Real(-1001.0)],
            -1000.3798854930417,
        ),
        (
            "exponential_lpdf",
            vec![Value::Real(1.3), Value::Real(2.0)],
            -1.9068528194400547,
        ),
        (
            "gamma_lpdf",
            vec![Value::Real(2.5), Value::Real(3.0), Value::Real(1.5)],
            -1.394170392487142,
        ),
        (
            "gamma_lpdf",
            vec![v(&[1.0, 2.0, 3.0]), Value::Real(2.0), Value::Real(1.5)],
            -4.775449882122958,
        ),
        (
            "beta_lpdf",
            vec![Value::Real(0.3), Value::Real(2.0), Value::Real(5.0)],
            0.7705248015812898,
        ),
        (
            "beta_lpdf",
            vec![Value::Real(0.0), Value::Real(1.0), Value::Real(3.0)],
            1.0986122886681098,
        ),
        (
            "lognormal_lpdf",
            vec![Value::Real(1.7), Value::Real(0.2), Value::Real(0.8)],
            -1.311825608265469,
        ),
        (
            "cauchy_lpdf",
            vec![Value::Real(1.0), Value::Real(0.5), Value::Real(2.0)],
            -1.8985016882257804,
        ),
        (
            "double_exponential_lpdf",
            vec![Value::Real(1.5), Value::Real(0.5), Value::Real(2.0)],
            -1.8862943611198908,
        ),
        (
            "student_t_lpdf",
            vec![
                Value::Real(1.2),
                Value::Real(3.5),
                Value::Real(0.5),
                Value::Real(2.0),
            ],
            -1.7600239819334451,
        ),
        (
            "uniform_lpdf",
            vec![Value::Real(0.5), Value::Real(0.0), Value::Real(2.0)],
            -0.6931471805599453,
        ),
        (
            "logistic_lpdf",
            vec![Value::Real(1.0), Value::Real(0.3), Value::Real(1.5)],
            -1.8457169368838429,
        ),
        (
            "logistic_lpdf",
            vec![Value::Real(-800.0), Value::Real(0.0), Value::Real(1.0)],
            -800.0,
        ),
        (
            "normal_lcdf",
            vec![Value::Real(1.2), Value::Real(0.5), Value::Real(2.0)],
            -0.45125151248278306,
        ),
        (
            "normal_lcdf",
            vec![Value::Real(-40.0), Value::Real(0.0), Value::Real(1.0)],
            -804.6084420137539,
        ),
        (
            "normal_lccdf",
            vec![Value::Real(1.2), Value::Real(0.5), Value::Real(2.0)],
            -1.0128860278195608,
        ),
        (
            "normal_lccdf",
            vec![Value::Real(40.0), Value::Real(0.0), Value::Real(1.0)],
            -804.6084420137539,
        ),
        (
            "std_normal_lcdf",
            vec![Value::Real(-1.0)],
            -1.8410216450092634,
        ),
        (
            "std_normal_lccdf",
            vec![Value::Real(3.0)],
            -6.60772622151035,
        ),
        (
            "std_normal_lcdf",
            vec![Value::Real(8.0)],
            -6.220960574271742e-16,
        ),
        (
            "bernoulli_lpmf",
            vec![Value::Int(1), Value::Real(0.3)],
            -1.2039728043259361,
        ),
        (
            "bernoulli_logit_lpmf",
            vec![Value::Int(0), Value::Real(1.2)],
            -1.4632824673380307,
        ),
        (
            "poisson_lpmf",
            vec![Value::Int(4), Value::Real(2.5)],
            -2.0128909028513253,
        ),
        (
            "poisson_log_lpmf",
            vec![Value::Int(4), Value::Real(0.9162907318741551)],
            -2.0128909028513253,
        ),
        (
            "binomial_lpmf",
            vec![Value::Int(3), Value::Int(10), Value::Real(0.25)],
            -1.3851658477400912,
        ),
        (
            "binomial_lpmf",
            vec![Value::Int(0), Value::Int(10), Value::Real(0.0)],
            0.0,
        ),
        (
            "categorical_lpmf",
            vec![Value::Int(2), v(&[0.2, 0.5, 0.3])],
            -0.6931471805599453,
        ),
        (
            "categorical_logit_lpmf",
            vec![Value::Int(3), v(&[1.0, 2.0, 0.5])],
            -1.9643687841079447,
        ),
        (
            "multi_normal_lpdf",
            vec![
                v(&[0.3, -1.2]),
                v(&[0.1, 0.4]),
                Value::Matrix(DMatrix::from_column_slice(2, 2, &[2.0, 0.6, 0.6, 1.0])),
            ],
            -3.7754690897664247,
        ),
        (
            "multi_normal_cholesky_lpdf",
            vec![
                v(&[0.3, -1.2]),
                v(&[0.1, 0.4]),
                Value::Matrix(DMatrix::from_column_slice(
                    2,
                    2,
                    &[
                        1.4142135623730951,
                        0.42426406871192845,
                        0.0,
                        0.9055385138137417,
                    ],
                )),
            ],
            -3.7754690897664247,
        ),
        (
            "determinant",
            vec![Value::Matrix(DMatrix::from_column_slice(
                3,
                3,
                &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
            ))],
            21.290000000000006,
        ),
        (
            "log_determinant",
            vec![Value::Matrix(DMatrix::from_column_slice(
                3,
                3,
                &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
            ))],
            3.0582374789053883,
        ),
        (
            "trace",
            vec![Value::Matrix(DMatrix::from_column_slice(
                3,
                3,
                &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
            ))],
            9.0,
        ),
        (
            "quad_form",
            vec![
                Value::Matrix(DMatrix::from_column_slice(
                    3,
                    3,
                    &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
                )),
                v(&[1.0, -2.0, 0.5]),
            ],
            12.600000000000001,
        ),
        ("variance", vec![v(&[1.0, 2.0, 4.0, 7.0])], 7.0),
        ("sd", vec![v(&[1.0, 2.0, 4.0, 7.0])], 2.6457513110645907),
        ("prod", vec![v(&[1.5, -2.0, 4.0])], -12.0),
        ("norm1", vec![v(&[1.5, -2.0, 4.0])], 7.5),
        ("distance", vec![v(&[1.0, 2.0]), v(&[4.0, 6.0])], 5.0),
    ];
    let mut wrong = vec![];
    for (name, args, want) in &cases {
        let (got, mut after, grads) = call(name, args, 3);
        let got = real(&got);
        let ok = got == *want || (got - want).abs() <= 1e-12 * want.abs().max(1e-300);
        if !ok {
            wrong.push(format!("{name}{args:?}: got {got:e}, want {want:e}"));
        }
        assert_eq!(grads, 0, "{name} evaluated the target");
        assert_eq!(
            after.random::<u64>(),
            rng(3).random::<u64>(),
            "{name} touched the RNG"
        );
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Draw `n` values from one stream.
fn draws(name: &str, args: &[Value], n: usize) -> Vec<f64> {
    let t = target();
    let mut ev = Evaluator::new(&t);
    let state = ChainState::at(vec![0.2, 0.3, -0.4], &mut ev).unwrap();
    let mut r = rng(11);
    let mut ctx = Ctx {
        dim: 3,
        state: Some(&state),
        ev: Some(&mut ev),
        rng: Some(&mut r),
        metric: None,
    };
    let Some(Native::Function(f)) = Natives::standard().get(name) else {
        panic!("{name}")
    };
    (0..n)
        .map(|_| match f(&mut ctx, args).unwrap() {
            Value::Int(i) => i as f64,
            Value::Real(x) => x,
            other => panic!("{name}: {other:?}"),
        })
        .collect()
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    (
        m,
        xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0),
    )
}

/// Each random-number function has the right mean and variance: 200,000
/// draws, the mean within 5 standard errors, the variance within 3%.
#[test]
fn stdlib_rngs_have_the_right_moments() {
    let r = Value::Real;
    let p = 1.0 / (1.0 + (-0.4f64).exp());
    let cases: Vec<(&str, Vec<Value>, f64, f64)> = vec![
        ("exponential_rng", vec![r(2.0)], 0.5, 0.25),
        ("gamma_rng", vec![r(3.0), r(1.5)], 2.0, 3.0 / 2.25),
        (
            "beta_rng",
            vec![r(2.0), r(5.0)],
            2.0 / 7.0,
            10.0 / (49.0 * 8.0),
        ),
        (
            "lognormal_rng",
            vec![r(0.2), r(0.5)],
            (0.2f64 + 0.125).exp(),
            (0.25f64.exp() - 1.0) * (0.4f64 + 0.25).exp(),
        ),
        ("double_exponential_rng", vec![r(0.5), r(2.0)], 0.5, 8.0),
        (
            "student_t_rng",
            vec![r(7.0), r(1.0), r(2.0)],
            1.0,
            4.0 * 7.0 / 5.0,
        ),
        ("chi_square_rng", vec![r(4.0)], 4.0, 8.0),
        ("bernoulli_rng", vec![r(0.3)], 0.3, 0.21),
        ("bernoulli_logit_rng", vec![r(0.4)], p, p * (1.0 - p)),
        ("binomial_rng", vec![Value::Int(10), r(0.25)], 2.5, 1.875),
        ("poisson_rng", vec![r(3.5)], 3.5, 3.5),
        ("poisson_log_rng", vec![r(3.5f64.ln())], 3.5, 3.5),
        ("categorical_rng", vec![v(&[0.2, 0.5, 0.3])], 2.1, 0.49),
        (
            "categorical_logit_rng",
            vec![v(&[0.2f64.ln(), 0.5f64.ln(), 0.3f64.ln()])],
            2.1,
            0.49,
        ),
        (
            "discrete_range_rng",
            vec![Value::Int(-2), Value::Int(3)],
            0.5,
            35.0 / 12.0,
        ),
        ("uniform_rng", vec![r(-1.0), r(3.0)], 1.0, 16.0 / 12.0),
    ];
    let n = 200_000;
    for (name, args, mean, var) in cases {
        let (m, s2) = mean_var(&draws(name, &args, n));
        let se = (var / n as f64).sqrt();
        assert!((m - mean).abs() < 5.0 * se, "{name}: mean {m}, want {mean}");
        assert!(
            (s2 / var - 1.0).abs() < 0.03,
            "{name}: variance {s2}, want {var}"
        );
    }
    // Cauchy has no moments: its quartiles are mu -/+ sigma.
    let xs = draws("cauchy_rng", &[r(0.5), r(2.0)], n);
    let below = |t: f64| xs.iter().filter(|&&x| x < t).count() as f64 / n as f64;
    assert!((below(0.5) - 0.5).abs() < 0.005 && (below(2.5) - 0.75).abs() < 0.005);
}

#[test]
fn stdlib_vector_rngs_and_multivariate() {
    // a vector argument gives one draw per element
    let (out, _, _) = call("gamma_rng", &[v(&[1.0, 2.0, 3.0]), Value::Real(1.0)], 5);
    assert_eq!(vec_of(&out).len(), 3);
    // two reals: one uniform, the same stream as before vectors were allowed
    let (out, _, _) = call("uniform_rng", &[Value::Real(0.0), Value::Real(1.0)], 5);
    assert_eq!(real(&out), rng(5).random::<f64>());
    // multi_normal_rng: the sample covariance is Sigma
    let sigma = DMatrix::from_column_slice(2, 2, &[2.0, 0.6, 0.6, 1.0]);
    let t = target();
    let mut ev = Evaluator::new(&t);
    let state = ChainState::at(vec![0.2, 0.3, -0.4], &mut ev).unwrap();
    let mut r = rng(17);
    let mut ctx = Ctx {
        dim: 3,
        state: Some(&state),
        ev: Some(&mut ev),
        rng: Some(&mut r),
        metric: None,
    };
    let Some(Native::Function(f)) = Natives::standard().get("multi_normal_rng") else {
        panic!()
    };
    let n = 100_000;
    let mut s = DMatrix::<f64>::zeros(2, 2);
    let mut m = DVector::<f64>::zeros(2);
    for _ in 0..n {
        let x = f(&mut ctx, &[v(&[1.0, -1.0]), Value::Matrix(sigma.clone())]).unwrap();
        let x = x.as_vector().unwrap().clone();
        m += &x;
        s += &x * x.transpose();
    }
    m /= n as f64;
    let cov = s / n as f64 - &m * m.transpose();
    assert!(
        (m - DVector::from_column_slice(&[1.0, -1.0])).norm() < 0.02,
        "mean"
    );
    assert!((cov - sigma).norm() < 0.04, "covariance");
    // dirichlet_rng is a simplex with mean alpha / sum(alpha)
    let mut total = DVector::<f64>::zeros(3);
    for seed in 0..20_000 {
        let (x, _, _) = call("dirichlet_rng", &[v(&[1.0, 2.0, 3.0])], seed);
        let x = x.as_vector().unwrap().clone();
        assert!((x.sum() - 1.0).abs() < 1e-12);
        total += x;
    }
    let mean = total / 20_000.0;
    assert!((mean - DVector::from_column_slice(&[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])).norm() < 0.01);
}

#[test]
fn stdlib_linear_algebra() {
    let a = DMatrix::from_column_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]);
    let m = |x: Value| match x {
        Value::Matrix(m) => m,
        other => panic!("{other:?}"),
    };
    let am = Value::Matrix(a.clone());
    let l = m(call("cholesky_decompose", std::slice::from_ref(&am), 1).0);
    assert!((&l * l.transpose() - &a).norm() < 1e-12 && l[(0, 1)] == 0.0);
    let inv = m(call("inverse", std::slice::from_ref(&am), 1).0);
    assert!((&a * inv - DMatrix::identity(3, 3)).norm() < 1e-12);
    let b = v(&[1.0, 2.0, 3.0]);
    for f in ["mdivide_left", "mdivide_left_spd"] {
        let x = call(f, &[am.clone(), b.clone()], 1).0;
        assert!(
            (&a * x.as_vector().unwrap() - DVector::from_column_slice(&[1.0, 2.0, 3.0])).norm()
                < 1e-12,
            "{f}"
        );
    }
    let x = call(
        "mdivide_left_tri_low",
        &[Value::Matrix(l.clone()), b.clone()],
        1,
    )
    .0;
    assert!(
        (&l * x.as_vector().unwrap() - DVector::from_column_slice(&[1.0, 2.0, 3.0])).norm() < 1e-12
    );
    let vals = call("eigenvalues_sym", std::slice::from_ref(&am), 1).0;
    let vecs = m(call("eigenvectors_sym", std::slice::from_ref(&am), 1).0);
    let vals = vals.as_vector().unwrap().clone();
    assert!(vals[0] <= vals[1] && vals[1] <= vals[2], "ascending");
    assert!((&a * &vecs - &vecs * DMatrix::from_diagonal(&vals)).norm() < 1e-12);
    // shapes and conversions, column-major as Stan
    let mm = Value::Matrix(DMatrix::from_column_slice(
        2,
        3,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    assert_eq!(
        vec_of(&call("to_vector", std::slice::from_ref(&mm), 1).0),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        m(call(
            "to_matrix",
            &[
                v(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Value::Int(3),
                Value::Int(2)
            ],
            1
        )
        .0),
        DMatrix::from_column_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
    assert_eq!(
        m(call("crossprod", std::slice::from_ref(&mm), 1).0).shape(),
        (3, 3)
    );
    assert_eq!(
        m(call("tcrossprod", std::slice::from_ref(&mm), 1).0).shape(),
        (2, 2)
    );
    assert_eq!(
        m(call("append_col", &[mm.clone(), v(&[7.0, 8.0])], 1).0)
            .column(3)
            .as_slice(),
        &[7.0, 8.0]
    );
    assert_eq!(
        vec_of(&call("append_row", &[Value::Real(0.0), v(&[1.0, 2.0])], 1).0),
        vec![0.0, 1.0, 2.0]
    );
    assert_eq!(
        vec_of(
            &call(
                "segment",
                &[v(&[1.0, 2.0, 3.0, 4.0]), Value::Int(2), Value::Int(2)],
                1
            )
            .0
        ),
        vec![2.0, 3.0]
    );
    assert_eq!(
        vec_of(
            &call(
                "linspaced_vector",
                &[Value::Int(5), Value::Real(0.0), Value::Real(1.0)],
                1
            )
            .0
        ),
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
    assert_eq!(
        vec_of(&call("sort_desc", &[v(&[2.0, -1.0, 3.0])], 1).0),
        vec![3.0, 2.0, -1.0]
    );
    assert_eq!(
        vec_of(&call("cumulative_sum", &[v(&[1.0, 2.0, 3.0])], 1).0),
        vec![1.0, 3.0, 6.0]
    );
    // elementwise functions and reductions now take matrices too
    assert_eq!(
        m(call(
            "floor",
            &[Value::Matrix(DMatrix::from_element(2, 2, 1.5))],
            1
        )
        .0),
        DMatrix::from_element(2, 2, 1.0)
    );
    assert_eq!(real(&call("sum", std::slice::from_ref(&mm), 1).0), 21.0);
    assert_eq!(real(&call("max", &[mm], 1).0), 6.0);
}

#[test]
fn stdlib_conversions_and_errors() {
    assert_eq!(call("to_int", &[Value::Real(-2.9)], 1).0, Value::Int(-2));
    assert_eq!(
        call("choose", &[Value::Int(10), Value::Int(3)], 1).0,
        Value::Int(120)
    );
    assert_eq!(call("int_step", &[Value::Real(0.0)], 1).0, Value::Int(0));
    let e = run_err("kernel k { step {\n  int n = to_int(positive_infinity());\n  move(q);\n} }");
    assert_eq!(e, "t.lapkernel:2:11: to_int: inf is not in the int range");
    let e = run_err("kernel k { step {\n  real x = gamma_rng(0, 1);\n  move(q);\n} }");
    assert_eq!(
        e,
        "t.lapkernel:2:12: gamma_rng: alpha must be positive and finite, got 0"
    );
    let e = run_err("kernel k { step {\n  int k = categorical_rng(append_row(0.5, rep_vector(0.6, 1)));\n  move(q);\n} }");
    assert_eq!(
        e,
        "t.lapkernel:2:11: categorical_rng: theta must sum to 1, sums to 1.1"
    );
    let e = run_err("kernel k { step {\n  matrix[2, 2] s = rep_matrix(1, 2, 2);\n  matrix[2, 2] l = cholesky_decompose(s);\n  move(q);\n} }");
    assert_eq!(
        e,
        "t.lapkernel:3:20: cholesky_decompose: the matrix is not positive definite"
    );
}

/// The whole library from a kernel: densities take `|` (also `_lpmf`,
/// `_lcdf`, `_lccdf`), int results flow into int variables and indices.
#[test]
fn stdlib_from_a_kernel() {
    let (vars, out) = run(
        "kernel k {
           state { int k; real lpm; real lc; vector[3] z; real s; }
           step {
             k = categorical_logit_rng(rep_vector(0, 3));
             lpm = poisson_lpmf(to_int(floor(2.7)) | 2.0) + bernoulli_lpmf(1 | 0.25);
             lc = normal_lcdf(0 | 0, 1) + std_normal_lccdf(0);
             matrix[3, 3] L = cholesky_decompose(diag_matrix(ones_vector(3)));
             z = multi_normal_cholesky_rng(q, L);
             s = sin(pi() / 2) + cos(0) + head(append_row(1, z), 1)[1];
             propose(z, gamma_lpdf(2 | 2, 1) - gamma_lpdf(2 | 2, 1));
           } }",
        1,
        1,
    );
    let Value::Int(k) = vars[0] else { panic!() };
    assert!((1..=3).contains(&k));
    assert!(
        (real(&vars[1]) - ((-2.0f64 + 2.0 * 2.0f64.ln() - 2.0f64.ln()) + 0.25f64.ln())).abs()
            < 1e-12
    );
    assert!((real(&vars[2]) - 2.0 * 0.5f64.ln()).abs() < 1e-15);
    assert_eq!(vec_of(&vars[3]).len(), 3);
    assert_eq!(real(&vars[4]), 3.0);
    assert!(matches!(out[0], Outcome::Propose { .. }));
    // a density without `|` is an error, as for _lpdf
    let k = parse(
        "t.lapkernel",
        "kernel k { step { real x = poisson_lpmf(2, 1.0); move(q); } }",
    )
    .unwrap();
    let e = format!(
        "{:?}",
        check("t.lapkernel", &k, Some(3), &Registry::standard()).unwrap_err()
    );
    assert!(e.contains("write poisson_lpmf(y | ...)"), "{e}");
}
