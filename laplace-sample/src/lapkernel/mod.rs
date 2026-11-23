//! `.lapkernel`: kernels written in a small Stan-like language (Level 3).
//!
//! - [`parse`]: grammar (pest) and syntax tree, with line/column spans;
//! - static checks (2.2), the interpreter and its built-in registry (2.3),
//!   and the `Kernel` implementation (2.4) build on it.

pub mod ast;
pub mod builtins;
pub mod check;
pub mod parse;
pub mod types;

use std::fmt;

use ast::Span;

/// An error in a `.lapkernel` file, pointing at `file:line:column`.
#[derive(Clone, Debug, PartialEq)]
pub struct LapError {
    pub file: String,
    pub span: Span,
    pub message: String,
}

impl LapError {
    pub fn new(file: &str, span: Span, message: impl Into<String>) -> Self {
        Self {
            file: file.to_string(),
            span,
            message: message.into(),
        }
    }
}

impl fmt::Display for LapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.file, self.span, self.message)
    }
}

impl std::error::Error for LapError {}

#[cfg(test)]
mod parse_tests {
    use super::ast::*;
    use super::parse::parse;

    /// The plan's example, verbatim (docs/sampler/laplace_kernels.md, 2.1).
    const PLAN_MALA: &str = r#"//@laplace
//@brief Metropolis-adjusted Langevin algorithm
//@param eps Step size
kernel mala {
  tuning {
    real<lower=0> eps = 0.1 @adapt(step_size);
  }
  state { }          // optional: variables kept across iterations
  init  { }          // optional: sets state
  functions { }      // optional: helpers, Stan syntax
  step {
    vector[dim] mu_fwd = q + 0.5 * eps^2 * grad;
    vector[dim] q_new  = mu_fwd + eps * std_normal_rng(dim);
    (real lp_new, vector[dim] g_new) = log_density_grad(q_new);
    vector[dim] mu_bwd = q_new + 0.5 * eps^2 * g_new;
    propose(q_new, normal_lpdf(q | mu_bwd, eps) - normal_lpdf(q_new | mu_fwd, eps));
  }
}
"#;

    fn ok(src: &str) -> KernelFile {
        parse("t.lapkernel", src).unwrap_or_else(|e| panic!("{e}"))
    }

    fn err(src: &str) -> String {
        parse("t.lapkernel", src).unwrap_err().to_string()
    }

    #[test]
    fn parses_the_plan_example() {
        let k = ok(PLAN_MALA);
        assert_eq!(k.name, "mala");
        assert!(k.docs.laplace);
        assert_eq!(
            k.docs.brief.as_deref(),
            Some("Metropolis-adjusted Langevin algorithm")
        );
        assert_eq!(k.docs.params, vec![("eps".into(), "Step size".into())]);
        assert_eq!(k.tuning.len(), 1);
        let eps = &k.tuning[0];
        assert_eq!(eps.name, "eps");
        assert_eq!(eps.ty, TypeExpr::Real);
        assert!(eps.bounds.lower.is_some());
        assert_eq!(eps.adapt.as_ref().unwrap().name, "step_size");
        assert_eq!(k.init.as_ref().unwrap().stmts.len(), 0);
        let step = k.step.as_ref().unwrap();
        assert_eq!(step.stmts.len(), 5);
        assert!(matches!(&step.stmts[2], Stmt::Destructure { targets, .. } if targets.len() == 2));
        let Stmt::Expr(Expr {
            kind: ExprKind::Call { name, args, .. },
            ..
        }) = &step.stmts[4]
        else {
            panic!("propose call")
        };
        assert_eq!(name, "propose");
        let ExprKind::Binary(BinOp::Sub, l, _) = &args[1].kind else {
            panic!("difference")
        };
        assert!(
            matches!(&l.kind, ExprKind::Call { conditional: true, args, .. } if args.len() == 3)
        );
        assert_eq!(
            k.sections.iter().map(|s| s.0).collect::<Vec<_>>(),
            vec![
                Section::Tuning,
                Section::State,
                Section::Init,
                Section::Functions,
                Section::Step
            ]
        );
    }

    #[test]
    fn precedence_and_associativity() {
        let k = ok("kernel k { step { real x = -a ^ 2 ^ 3 * b + c .* d; move(q); } }");
        let Stmt::Decl(d) = &k.step.as_ref().unwrap().stmts[0] else {
            panic!()
        };
        // ((-(a ^ (2 ^ 3))) * b) + (c .* d)
        let ExprKind::Binary(BinOp::Add, l, r) = &d.init.as_ref().unwrap().kind else {
            panic!("+ at the top")
        };
        assert!(matches!(&r.kind, ExprKind::Binary(BinOp::ElMul, _, _)));
        let ExprKind::Binary(BinOp::Mul, neg, _) = &l.kind else {
            panic!("* under +")
        };
        let ExprKind::Unary(UnOp::Neg, pow) = &neg.kind else {
            panic!("unary minus outside ^")
        };
        let ExprKind::Binary(BinOp::Pow, a, rest) = &pow.kind else {
            panic!("^")
        };
        assert_eq!(a.kind, ExprKind::Var("a".into()));
        assert!(
            matches!(&rest.kind, ExprKind::Binary(BinOp::Pow, _, _)),
            "^ is right-associative"
        );
    }

    #[test]
    fn every_construct() {
        let k = ok(r#"
kernel everything {
  tuning {
    real<lower=0, upper=10> eps = 1.5e-1 @adapt(step_size, target_accept = 0.574);
    vector<lower=0>[dim] m = rep_vector(1, dim) @adapt(metric);
    int<lower=1> steps = 10;
  }
  state {
    int n_seen = 0;
    matrix[dim, 2] buf;
    tuple(real, vector[dim]) last;
  }
  init {
    n_seen = 0;
  }
  functions {
    real half(real x) { return 0.5 * x; }
    tuple(real, vector) pair(vector v) { return (sum(v), v); }
    void nothing() { return; }
  }
  step {
    vector[dim] p = std_normal_rng(dim);
    real acc = 0;
    for (i in 1:dim) {
      acc += p[i] ^ 2;
    }
    while (acc > 100 && !(acc != acc) || acc < 0) { acc /= 2; }
    if (acc > 1) {
      p[1] = half(acc);
    } else if (acc == 0) {
      p = p * 2;
    } else {
      p = p ./ 2;
    }
    buf[1, 2] = .5;
    (acc, p) = (1.0, p);
    n_seen = n_seen + 1 % 3;
    move(q + eps * p);
  }
}
"#);
        assert_eq!(k.tuning.len(), 3);
        assert_eq!(
            k.tuning[0].adapt.as_ref().unwrap().args[0].0,
            "target_accept"
        );
        assert_eq!(k.state.len(), 3);
        assert!(matches!(&k.state[2].ty, TypeExpr::Tuple(t) if t.len() == 2));
        assert_eq!(k.functions.len(), 3);
        assert_eq!(
            k.functions[1].ret,
            TypeExpr::Tuple(vec![TypeExpr::Real, TypeExpr::Vector(None)])
        );
        assert_eq!(k.functions[2].ret, TypeExpr::Void);
        assert_eq!(k.step.as_ref().unwrap().stmts.len(), 9);
    }

    #[test]
    fn spans_point_at_the_construct() {
        let k = ok("kernel k {\n  step {\n    real x = 1;\n      move(q);\n  }\n}\n");
        let s = &k.step.as_ref().unwrap().stmts;
        assert_eq!(s[0].span(), Span { line: 3, col: 5 });
        assert_eq!(s[1].span(), Span { line: 4, col: 7 });
    }

    #[test]
    fn missing_semicolon_points_at_the_next_token() {
        let e = err("kernel k {\n  step {\n    real x = 1\n    move(q);\n  }\n}\n");
        assert!(e.starts_with("t.lapkernel:4:5: syntax error"), "{e}");
    }

    #[test]
    fn unknown_section_is_reported_where_it_starts() {
        let e =
            err("kernel k {\n  tuning { }\n  params { real eps = 1; }\n  step { move(q); }\n}\n");
        assert!(e.starts_with("t.lapkernel:3:3: syntax error"), "{e}");
    }

    #[test]
    fn unclosed_block_is_reported_at_end_of_file() {
        let e = err("kernel k {\n  step {\n    move(q);\n");
        assert!(e.starts_with("t.lapkernel:4:1: syntax error"), "{e}");
    }

    #[test]
    fn keyword_as_a_name_is_rejected() {
        let e = err("kernel k {\n  step {\n    real for = 1;\n  }\n}\n");
        assert!(e.starts_with("t.lapkernel:3:10: syntax error"), "{e}");
        assert!(e.contains("a name"), "{e}");
    }

    #[test]
    fn expected_lists_do_not_repeat_names() {
        let e = err("kernel k {\n  step {\n    real x = f(1) g;\n  }\n}\n");
        assert!(
            e.starts_with("t.lapkernel:3:19: syntax error: expected "),
            "{e}"
        );
        assert_eq!(e.matches("an operator").count(), 1, "{e}");
    }

    #[test]
    fn missing_kernel_keyword() {
        let e = err("//@laplace\nkernl k { step { move(q); } }\n");
        assert!(
            e.starts_with("t.lapkernel:2:1: syntax error: expected `kernel`"),
            "{e}"
        );
    }

    #[test]
    fn bad_operator_points_at_the_column() {
        let e = err("kernel k {\n  step {\n    real x = 1 +* 2;\n  }\n}\n");
        assert!(
            e.starts_with("t.lapkernel:3:17: syntax error: expected an expression"),
            "{e}"
        );
    }
}
