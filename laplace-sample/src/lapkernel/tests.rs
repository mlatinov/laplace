//! Phase 2.3 tests: every built-in (values and RNG consumption) and the
//! interpreter's semantics.

use nalgebra::DVector;
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
        "t.lapkernel:2:12: normal_lpdf: normal_lpdf: sigma must be positive, got -1"
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
