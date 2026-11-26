//! Tree-walking interpreter for checked kernels (Phase 2.3).
//!
//! Values are [`Value`]s (nalgebra vectors and matrices). Scopes, outermost
//! first: tuning values (read-only), state variables (kept across
//! iterations), then the locals of the block being run. Built-ins come from
//! the [`Registry`] (which are terminal) and [`Natives`] (what the others
//! do). Runtime errors point at the expression that failed.
//!
//! Arithmetic is written to match the Rust kernels operation for operation
//! (Phase 2.5 equivalence): vector arithmetic is elementwise in index order,
//! scalar-vector arithmetic applies the scalar to each element, and `x ^ 2`
//! with a literal integer 2 is `x * x` (other exponents use `powf`).

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::{DMatrix, DVector};

use super::ast::*;
use super::builtins::{Kind, Registry};
use super::check::{AdaptKind, Checked};
use super::natives::{Ctx, Native, Natives};
use super::types::Value;
use super::LapError;

/// How a step ended (the terminal statement and its arguments).
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// `propose(q_new, log_correction)`
    Propose {
        q_new: DVector<f64>,
        log_correction: f64,
    },
    /// `move(q_new)`
    Move { q_new: DVector<f64> },
}

enum Flow {
    Normal,
    Return(Value),
    End(Outcome),
    /// `break;`, travelling out to the innermost loop, which consumes it.
    Break,
    /// `continue;`, likewise.
    Continue,
}

type Scope = HashMap<String, Value>;

pub struct Interp {
    pub checked: Arc<Checked>,
    reg: Registry,
    natives: Natives,
    /// Function name -> index in `checked.kernel.functions`.
    funs: HashMap<String, usize>,
}

impl Interp {
    pub fn new(checked: Arc<Checked>) -> Self {
        let funs = checked
            .kernel
            .functions
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        Self {
            checked,
            reg: Registry::standard(),
            natives: Natives::standard(),
            funs,
        }
    }

    fn err(&self, span: Span, msg: impl Into<String>) -> LapError {
        LapError::new(&self.checked.file, span, msg)
    }

    /// Tuning defaults, in declaration order: initializers evaluated with no
    /// chain (the checker allows only chain-free expressions there); an
    /// `@adapt(metric)` value without one is all ones.
    pub fn tuning_defaults(&self, dim: usize) -> Result<Vec<Value>, LapError> {
        let mut ctx = Ctx {
            dim,
            state: None,
            ev: None,
            rng: None,
            metric: None,
        };
        let mut scopes: Vec<Scope> = vec![Scope::new()];
        let mut out = vec![];
        for (d, info) in self.checked.kernel.tuning.iter().zip(&self.checked.tuning) {
            let v = match &d.init {
                Some(e) => {
                    let v = self.eval(e, &mut scopes, &mut ctx)?;
                    self.conform(&d.ty, v, &mut scopes, &mut ctx, d.span, &d.name)?
                }
                None if info.adapt == Some(AdaptKind::Metric) => {
                    Value::Vector(DVector::from_element(dim, 1.0))
                }
                None => {
                    return Err(
                        self.err(d.span, format!("tuning value `{}` has no default", d.name))
                    )
                }
            };
            scopes[0].insert(d.name.clone(), v.clone());
            out.push(v);
        }
        Ok(out)
    }

    /// Check a tuning value against its declared type and bounds (for
    /// `--set` and for adaptation).
    pub fn check_tuning(&self, index: usize, v: &Value, dim: usize) -> Result<Value, String> {
        let d = &self.checked.kernel.tuning[index];
        let mut ctx = Ctx {
            dim,
            state: None,
            ev: None,
            rng: None,
            metric: None,
        };
        let mut scopes = vec![Scope::new()];
        let v = self
            .conform(&d.ty, v.clone(), &mut scopes, &mut ctx, d.span, &d.name)
            .map_err(|e| e.message)?;
        self.check_bounds(&d.bounds, &v, &mut scopes, &mut ctx, &d.name)
            .map_err(|e| e.message)?;
        Ok(v)
    }

    fn base_scopes(&self, tuning: &[Value], state: &[Value]) -> Vec<Scope> {
        let k = &self.checked.kernel;
        let t: Scope = k
            .tuning
            .iter()
            .zip(tuning)
            .map(|(d, v)| (d.name.clone(), v.clone()))
            .collect();
        let s: Scope = k
            .state
            .iter()
            .zip(state)
            .map(|(d, v)| (d.name.clone(), v.clone()))
            .collect();
        vec![t, s]
    }

    fn save_state(&self, scopes: &[Scope], state: &mut [Value]) {
        for (d, slot) in self.checked.kernel.state.iter().zip(state.iter_mut()) {
            if let Some(v) = scopes[1].get(&d.name) {
                *slot = v.clone();
            }
        }
    }

    /// Initial state variables (their initializers, then `init { }`).
    /// Uninitialised state variables start at zero (NaN for reals, as in
    /// Stan, would make an `init`-less kernel fail on first use).
    pub fn init_state(
        &self,
        ctx: &mut Ctx<'_, '_>,
        tuning: &[Value],
    ) -> Result<Vec<Value>, LapError> {
        let k = &self.checked.kernel;
        let mut scopes = self.base_scopes(tuning, &[]);
        for d in &k.state {
            let v = match &d.init {
                Some(e) => {
                    let v = self.eval(e, &mut scopes, ctx)?;
                    self.conform(&d.ty, v, &mut scopes, ctx, d.span, &d.name)?
                }
                None => self.zero(&d.ty, &mut scopes, ctx, d.span)?,
            };
            scopes[1].insert(d.name.clone(), v);
        }
        let mut state: Vec<Value> = k.state.iter().map(|d| scopes[1][&d.name].clone()).collect();
        if let Some(b) = &k.init {
            scopes = self.base_scopes(tuning, &state);
            match self.block(b, &mut scopes, ctx)? {
                Flow::Normal => {}
                _ => unreachable!("checked: no return/terminal in init"),
            }
            self.save_state(&scopes, &mut state);
        }
        Ok(state)
    }

    /// Run `step { }` once. State variables are updated in place.
    pub fn step(
        &self,
        ctx: &mut Ctx<'_, '_>,
        tuning: &[Value],
        state: &mut [Value],
    ) -> Result<Outcome, LapError> {
        let k = &self.checked.kernel;
        let step = k.step.as_ref().expect("checked: has step");
        let mut scopes = self.base_scopes(tuning, state);
        let flow = self.block(step, &mut scopes, ctx)?;
        self.save_state(&scopes, state);
        match flow {
            Flow::End(o) => Ok(o),
            _ => Err(self.err(step.span, "step finished without propose() or move()")),
        }
    }

    // ---- statements ------------------------------------------------------------

    fn block(
        &self,
        b: &Block,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<Flow, LapError> {
        scopes.push(Scope::new());
        let mut out = Flow::Normal;
        for s in &b.stmts {
            match self.stmt(s, scopes, ctx) {
                Ok(Flow::Normal) => {}
                Ok(f) => {
                    out = f;
                    break;
                }
                Err(e) => {
                    scopes.pop();
                    return Err(e);
                }
            }
        }
        scopes.pop();
        Ok(out)
    }

    fn scoped(
        &self,
        s: &Stmt,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<Flow, LapError> {
        scopes.push(Scope::new());
        let r = self.stmt(s, scopes, ctx);
        scopes.pop();
        r
    }

    fn stmt(
        &self,
        s: &Stmt,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<Flow, LapError> {
        match s {
            Stmt::Decl(d) => {
                let v = match &d.init {
                    Some(e) => {
                        let v = self.eval(e, scopes, ctx)?;
                        self.conform(&d.ty, v, scopes, ctx, e.span, &d.name)?
                    }
                    None => self.zero(&d.ty, scopes, ctx, d.span)?,
                };
                if d.init.is_some() {
                    self.check_bounds(&d.bounds, &v, scopes, ctx, &d.name)?;
                }
                scopes.last_mut().unwrap().insert(d.name.clone(), v);
                Ok(Flow::Normal)
            }
            Stmt::Block(b) => self.block(b, scopes, ctx),
            Stmt::Expr(e) => {
                if let ExprKind::Call { name, args, .. } = &e.kind {
                    if let Some(entry) = self.reg.get(name) {
                        if matches!(entry.kind, Kind::Function { terminal: true, .. }) {
                            let vals = self.args(args, scopes, ctx)?;
                            return Ok(Flow::End(self.terminal(name, vals, e.span, ctx.dim)?));
                        }
                    }
                }
                self.eval(e, scopes, ctx)?;
                Ok(Flow::Normal)
            }
            Stmt::Assign {
                target, op, value, ..
            } => {
                let v = self.eval(value, scopes, ctx)?;
                let v = match op {
                    AssignOp::Set => v,
                    _ => {
                        let cur = self.lookup(&target.name, scopes, ctx, target.span)?;
                        let cur = if target.index.is_empty() {
                            cur
                        } else {
                            let idx = self.indices(&target.index, scopes, ctx)?;
                            self.index(&cur, &idx, target.span)?
                        };
                        let bop = match op {
                            AssignOp::Add => BinOp::Add,
                            AssignOp::Sub => BinOp::Sub,
                            AssignOp::Mul => BinOp::Mul,
                            _ => BinOp::Div,
                        };
                        self.binop(bop, cur, v, value.span)?
                    }
                };
                self.assign(target, v, scopes, ctx)?;
                Ok(Flow::Normal)
            }
            Stmt::Destructure { targets, value, .. } => {
                let Value::Tuple(vals) = self.eval(value, scopes, ctx)? else {
                    return Err(self.err(value.span, "destructuring needs a tuple"));
                };
                if vals.len() != targets.len() {
                    return Err(self.err(value.span, "tuple size does not match"));
                }
                for (t, v) in targets.iter().zip(vals) {
                    match t {
                        DestructTarget::Declare(ty, name, sp) => {
                            let v = self.conform(ty, v, scopes, ctx, *sp, name)?;
                            scopes.last_mut().unwrap().insert(name.clone(), v);
                        }
                        DestructTarget::Existing(name, sp) => {
                            let lv = LValue {
                                name: name.clone(),
                                index: vec![],
                                span: *sp,
                            };
                            self.assign(&lv, v, scopes, ctx)?;
                        }
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::If {
                cond, then, els, ..
            } => {
                if self.truthy(cond, scopes, ctx)? {
                    self.scoped(then, scopes, ctx)
                } else if let Some(e) = els {
                    self.scoped(e, scopes, ctx)
                } else {
                    Ok(Flow::Normal)
                }
            }
            Stmt::For {
                var, lo, hi, body, ..
            } => {
                let lo = self.int(lo, scopes, ctx)?;
                let hi = self.int(hi, scopes, ctx)?;
                for i in lo..=hi {
                    scopes.push(Scope::from([(var.clone(), Value::Int(i))]));
                    let f = self.stmt(body, scopes, ctx);
                    scopes.pop();
                    match f? {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => break,
                        other => return Ok(other),
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::While { cond, body, .. } => {
                while self.truthy(cond, scopes, ctx)? {
                    match self.scoped(body, scopes, ctx)? {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => break,
                        other => return Ok(other),
                    }
                }
                Ok(Flow::Normal)
            }
            // The checker allows these only inside a loop, which consumes them.
            Stmt::Break(_) => Ok(Flow::Break),
            Stmt::Continue(_) => Ok(Flow::Continue),
            Stmt::Return(e, _) => Ok(Flow::Return(match e {
                Some(e) => self.eval(e, scopes, ctx)?,
                None => Value::Void,
            })),
        }
    }

    fn terminal(
        &self,
        name: &str,
        vals: Vec<Value>,
        span: Span,
        dim: usize,
    ) -> Result<Outcome, LapError> {
        let q_new = match vals.first() {
            Some(Value::Vector(v)) if v.len() == dim => v.clone(),
            Some(Value::Vector(v)) => {
                return Err(self.err(
                    span,
                    format!(
                        "{name}(): q_new has {} elements, the block has {dim}",
                        v.len()
                    ),
                ))
            }
            _ => return Err(self.err(span, format!("{name}(): q_new must be a vector"))),
        };
        Ok(match name {
            "propose" => Outcome::Propose {
                q_new,
                log_correction: vals[1]
                    .as_real()
                    .ok_or_else(|| self.err(span, "propose(): log_correction must be a number"))?,
            },
            _ => Outcome::Move { q_new },
        })
    }

    fn assign(
        &self,
        lv: &LValue,
        v: Value,
        scopes: &mut [Scope],
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<(), LapError> {
        let idx = if lv.index.is_empty() {
            vec![]
        } else {
            let mut sc = scopes.to_vec();
            self.indices(&lv.index, &mut sc, ctx)?
        };
        let slot = scopes
            .iter_mut()
            .rev()
            .find_map(|s| s.get_mut(&lv.name))
            .ok_or_else(|| self.err(lv.span, format!("unknown variable `{}`", lv.name)))?;
        if idx.is_empty() {
            let same_shape = match (&*slot, &v) {
                (Value::Vector(a), Value::Vector(b)) => a.len() == b.len(),
                (Value::Matrix(a), Value::Matrix(b)) => a.shape() == b.shape(),
                _ => true,
            };
            if !same_shape {
                return Err(self.err(lv.span, format!("size mismatch assigning to `{}`", lv.name)));
            }
            *slot = match (&*slot, v) {
                (Value::Real(_), Value::Int(i)) => Value::Real(i as f64),
                (_, v) => v,
            };
            return Ok(());
        }
        let x = v
            .as_real()
            .ok_or_else(|| self.err(lv.span, "an indexed element must be assigned a number"))?;
        match (slot, idx.as_slice()) {
            (Value::Vector(vec), [i]) => {
                let i = self.pos(*i, vec.len(), lv.span)?;
                vec[i] = x;
            }
            (Value::Matrix(m), [i, j]) => {
                let (r, c) = (
                    self.pos(*i, m.nrows(), lv.span)?,
                    self.pos(*j, m.ncols(), lv.span)?,
                );
                m[(r, c)] = x;
            }
            _ => return Err(self.err(lv.span, "bad index")),
        }
        Ok(())
    }

    // ---- values ------------------------------------------------------------------

    fn size_of(
        &self,
        e: &Expr,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<usize, LapError> {
        let n = self.int(e, scopes, ctx)?;
        usize::try_from(n).map_err(|_| self.err(e.span, format!("size {n} is negative")))
    }

    /// Promote and size-check a value for a declared type.
    fn conform(
        &self,
        ty: &TypeExpr,
        v: Value,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
        span: Span,
        name: &str,
    ) -> Result<Value, LapError> {
        match (ty, v) {
            (TypeExpr::Real, Value::Int(i)) => Ok(Value::Real(i as f64)),
            (TypeExpr::Vector(Some(n)), Value::Vector(x)) => {
                let n = self.size_of(n, scopes, ctx)?;
                if x.len() != n {
                    return Err(self.err(
                        span,
                        format!(
                            "`{name}` is a vector[{n}] but the value has {} elements",
                            x.len()
                        ),
                    ));
                }
                Ok(Value::Vector(x))
            }
            (TypeExpr::Matrix(Some((r, c))), Value::Matrix(m)) => {
                let (r, c) = (self.size_of(r, scopes, ctx)?, self.size_of(c, scopes, ctx)?);
                if m.shape() != (r, c) {
                    return Err(self.err(
                        span,
                        format!(
                            "`{name}` is a matrix[{r}, {c}] but the value is {:?}",
                            m.shape()
                        ),
                    ));
                }
                Ok(Value::Matrix(m))
            }
            (TypeExpr::Tuple(ts), Value::Tuple(vs)) => Ok(Value::Tuple(
                ts.iter()
                    .zip(vs)
                    .map(|(t, v)| self.conform(t, v, scopes, ctx, span, name))
                    .collect::<Result<_, _>>()?,
            )),
            (_, v) => Ok(v),
        }
    }

    fn zero(
        &self,
        ty: &TypeExpr,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
        span: Span,
    ) -> Result<Value, LapError> {
        Ok(match ty {
            TypeExpr::Int => Value::Int(0),
            TypeExpr::Real => Value::Real(0.0),
            TypeExpr::Vector(Some(n)) => {
                Value::Vector(DVector::zeros(self.size_of(n, scopes, ctx)?))
            }
            TypeExpr::Matrix(Some((r, c))) => {
                let (r, c) = (self.size_of(r, scopes, ctx)?, self.size_of(c, scopes, ctx)?);
                Value::Matrix(DMatrix::zeros(r, c))
            }
            TypeExpr::Tuple(ts) => Value::Tuple(
                ts.iter()
                    .map(|t| self.zero(t, scopes, ctx, span))
                    .collect::<Result<_, _>>()?,
            ),
            _ => return Err(self.err(span, "cannot create a value of this type")),
        })
    }

    fn check_bounds(
        &self,
        b: &Bounds,
        v: &Value,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
        name: &str,
    ) -> Result<(), LapError> {
        let elems: Vec<f64> = match v {
            Value::Int(i) => vec![*i as f64],
            Value::Real(x) => vec![*x],
            Value::Vector(x) => x.iter().copied().collect(),
            Value::Matrix(m) => m.iter().copied().collect(),
            _ => return Ok(()),
        };
        if let Some(lo) = &b.lower {
            let lo_v = self.real(lo, scopes, ctx)?;
            if let Some(x) = elems.iter().find(|x| x.is_nan() || **x < lo_v) {
                return Err(self.err(lo.span, format!("`{name}` must be >= {lo_v}, got {x}")));
            }
        }
        if let Some(hi) = &b.upper {
            let hi_v = self.real(hi, scopes, ctx)?;
            if let Some(x) = elems.iter().find(|x| x.is_nan() || **x > hi_v) {
                return Err(self.err(hi.span, format!("`{name}` must be <= {hi_v}, got {x}")));
            }
        }
        Ok(())
    }

    fn lookup(
        &self,
        name: &str,
        scopes: &[Scope],
        ctx: &mut Ctx<'_, '_>,
        span: Span,
    ) -> Result<Value, LapError> {
        if let Some(v) = scopes.iter().rev().find_map(|s| s.get(name)) {
            return Ok(v.clone());
        }
        match self.natives.get(name) {
            Some(Native::Value(get)) => {
                get(ctx).map_err(|m| self.err(span, format!("`{name}`: {m}")))
            }
            _ => Err(self.err(span, format!("unknown name `{name}`"))),
        }
    }

    fn truthy(
        &self,
        e: &Expr,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<bool, LapError> {
        Ok(self.real(e, scopes, ctx)? != 0.0)
    }

    fn real(
        &self,
        e: &Expr,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<f64, LapError> {
        let v = self.eval(e, scopes, ctx)?;
        v.as_real()
            .ok_or_else(|| self.err(e.span, format!("expected a number, got {}", v.type_name())))
    }

    fn int(
        &self,
        e: &Expr,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<i64, LapError> {
        let v = self.eval(e, scopes, ctx)?;
        v.as_int()
            .ok_or_else(|| self.err(e.span, format!("expected an int, got {}", v.type_name())))
    }

    fn indices(
        &self,
        idx: &[Expr],
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<Vec<i64>, LapError> {
        idx.iter().map(|e| self.int(e, scopes, ctx)).collect()
    }

    /// 1-based index -> 0-based position, bounds-checked.
    fn pos(&self, i: i64, len: usize, span: Span) -> Result<usize, LapError> {
        if i >= 1 && (i as usize) <= len {
            Ok(i as usize - 1)
        } else {
            Err(self.err(span, format!("index {i} out of range 1..{len}")))
        }
    }

    fn index(&self, base: &Value, idx: &[i64], span: Span) -> Result<Value, LapError> {
        match (base, idx) {
            (Value::Vector(v), [i]) => Ok(Value::Real(v[self.pos(*i, v.len(), span)?])),
            (Value::Matrix(m), [i, j]) => Ok(Value::Real(
                m[(
                    self.pos(*i, m.nrows(), span)?,
                    self.pos(*j, m.ncols(), span)?,
                )],
            )),
            _ => Err(self.err(span, "bad index")),
        }
    }

    fn args(
        &self,
        args: &[Expr],
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<Vec<Value>, LapError> {
        args.iter().map(|a| self.eval(a, scopes, ctx)).collect()
    }

    pub(crate) fn eval(
        &self,
        e: &Expr,
        scopes: &mut Vec<Scope>,
        ctx: &mut Ctx<'_, '_>,
    ) -> Result<Value, LapError> {
        match &e.kind {
            ExprKind::Int(i) => Ok(Value::Int(*i)),
            ExprKind::Real(x) => Ok(Value::Real(*x)),
            ExprKind::Var(name) => self.lookup(name, scopes, ctx, e.span),
            ExprKind::Tuple(es) => Ok(Value::Tuple(self.args(es, scopes, ctx)?)),
            ExprKind::TupleIndex(b, n) => match self.eval(b, scopes, ctx)? {
                Value::Tuple(mut vs) if *n >= 1 && *n <= vs.len() => Ok(vs.swap_remove(n - 1)),
                _ => Err(self.err(e.span, format!(".{n} is out of range"))),
            },
            ExprKind::Index(b, idx) => {
                let base = self.eval(b, scopes, ctx)?;
                let idx = self.indices(idx, scopes, ctx)?;
                self.index(&base, &idx, e.span)
            }
            ExprKind::Unary(op, x) => {
                let v = self.eval(x, scopes, ctx)?;
                match (op, v) {
                    (UnOp::Neg, Value::Int(i)) => Ok(Value::Int(-i)),
                    (UnOp::Neg, Value::Real(x)) => Ok(Value::Real(-x)),
                    (UnOp::Neg, Value::Vector(v)) => Ok(Value::Vector(-v)),
                    (UnOp::Neg, Value::Matrix(m)) => Ok(Value::Matrix(-m)),
                    (UnOp::Not, v) => {
                        let x = v
                            .as_real()
                            .ok_or_else(|| self.err(e.span, "`!` needs a number"))?;
                        Ok(Value::Int((x == 0.0) as i64))
                    }
                    (_, v) => Err(self.err(e.span, format!("cannot negate a {}", v.type_name()))),
                }
            }
            ExprKind::Binary(BinOp::And, a, b) => Ok(Value::Int(
                (self.truthy(a, scopes, ctx)? && self.truthy(b, scopes, ctx)?) as i64,
            )),
            ExprKind::Binary(BinOp::Or, a, b) => Ok(Value::Int(
                (self.truthy(a, scopes, ctx)? || self.truthy(b, scopes, ctx)?) as i64,
            )),
            ExprKind::Binary(BinOp::Pow, a, b) if b.kind == ExprKind::Int(2) => {
                // x ^ 2 is x * x, exactly as the Rust kernels write it.
                let x = self.real(a, scopes, ctx)?;
                Ok(Value::Real(x * x))
            }
            ExprKind::Binary(op, a, b) => {
                let va = self.eval(a, scopes, ctx)?;
                let vb = self.eval(b, scopes, ctx)?;
                self.binop(*op, va, vb, e.span)
            }
            ExprKind::Call { name, args, .. } => {
                let vals = self.args(args, scopes, ctx)?;
                if let Some(&i) = self.funs.get(name.as_str()) {
                    return self.call_user(&self.checked.kernel.functions[i], vals, ctx, e.span);
                }
                match self.natives.get(name) {
                    Some(Native::Function(call)) => {
                        call(ctx, &vals).map_err(|m| self.err(e.span, format!("{name}: {m}")))
                    }
                    _ => Err(self.err(e.span, format!("`{name}` cannot be called here"))),
                }
            }
        }
    }

    fn call_user(
        &self,
        f: &FunDef,
        vals: Vec<Value>,
        ctx: &mut Ctx<'_, '_>,
        span: Span,
    ) -> Result<Value, LapError> {
        // A function sees its parameters, other functions and built-ins.
        let scope: Scope = f
            .params
            .iter()
            .zip(vals)
            .map(|((t, n), v)| {
                let v = match (t, v) {
                    (TypeExpr::Real, Value::Int(i)) => Value::Real(i as f64),
                    (_, v) => v,
                };
                (n.clone(), v)
            })
            .collect();
        let mut scopes = vec![scope];
        match self.block(&f.body, &mut scopes, ctx)? {
            Flow::Return(v) => Ok(match (&f.ret, v) {
                (TypeExpr::Real, Value::Int(i)) => Value::Real(i as f64),
                (_, v) => v,
            }),
            Flow::Normal if f.ret == TypeExpr::Void => Ok(Value::Void),
            _ => Err(self.err(
                span,
                format!("function `{}` ended without returning", f.name),
            )),
        }
    }

    fn binop(&self, op: BinOp, a: Value, b: Value, span: Span) -> Result<Value, LapError> {
        use Value::*;
        let sym = op.symbol();
        let mismatch =
            |n: usize, m: usize| self.err(span, format!("size mismatch in `{sym}`: {n} and {m}"));
        let r = match (op, a, b) {
            // integers stay integers (Stan: int / int truncates)
            (BinOp::Add, Int(x), Int(y)) => Int(x.wrapping_add(y)),
            (BinOp::Sub, Int(x), Int(y)) => Int(x.wrapping_sub(y)),
            (BinOp::Mul, Int(x), Int(y)) => Int(x.wrapping_mul(y)),
            (BinOp::Div | BinOp::Mod, Int(_), Int(0)) => {
                return Err(self.err(span, "integer division by zero"))
            }
            (BinOp::Div, Int(x), Int(y)) => Int(x / y),
            (BinOp::Mod, Int(x), Int(y)) => Int(x % y),
            (BinOp::Pow, a, b) => Real(scalar(&a).powf(scalar(&b))),
            (BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge, a, b) => {
                let (x, y) = (scalar(&a), scalar(&b));
                Int(match op {
                    BinOp::Eq => x == y,
                    BinOp::Ne => x != y,
                    BinOp::Lt => x < y,
                    BinOp::Le => x <= y,
                    BinOp::Gt => x > y,
                    _ => x >= y,
                } as i64)
            }
            // scalar op scalar
            (op, a @ (Int(_) | Real(_)), b @ (Int(_) | Real(_))) => {
                Real(arith(op, scalar(&a), scalar(&b)))
            }
            // vector/matrix with vector/matrix, elementwise in index order
            (BinOp::Add | BinOp::Sub | BinOp::ElMul | BinOp::ElDiv, Vector(x), Vector(y)) => {
                if x.len() != y.len() {
                    return Err(mismatch(x.len(), y.len()));
                }
                Vector(x.zip_map(&y, |p, q| arith(op, p, q)))
            }
            (BinOp::Add | BinOp::Sub | BinOp::ElMul | BinOp::ElDiv, Matrix(x), Matrix(y)) => {
                if x.shape() != y.shape() {
                    return Err(self.err(span, format!("size mismatch in `{sym}`")));
                }
                Matrix(x.zip_map(&y, |p, q| arith(op, p, q)))
            }
            // scalar with vector/matrix: applied to each element
            (op, s @ (Int(_) | Real(_)), Vector(x)) => {
                let s = scalar(&s);
                Vector(x.map(|v| arith(op, s, v)))
            }
            (op, Vector(x), s @ (Int(_) | Real(_))) => {
                let s = scalar(&s);
                Vector(x.map(|v| arith(op, v, s)))
            }
            (op, s @ (Int(_) | Real(_)), Matrix(x)) => {
                let s = scalar(&s);
                Matrix(x.map(|v| arith(op, s, v)))
            }
            (op, Matrix(x), s @ (Int(_) | Real(_))) => {
                let s = scalar(&s);
                Matrix(x.map(|v| arith(op, v, s)))
            }
            (BinOp::Mul, Matrix(m), Vector(v)) => {
                if m.ncols() != v.len() {
                    return Err(mismatch(m.ncols(), v.len()));
                }
                Vector(m * v)
            }
            (BinOp::Mul, Matrix(a), Matrix(b)) => {
                if a.ncols() != b.nrows() {
                    return Err(mismatch(a.ncols(), b.nrows()));
                }
                Matrix(a * b)
            }
            (_, a, b) => {
                return Err(self.err(
                    span,
                    format!(
                        "`{sym}` is not defined for {} and {}",
                        a.type_name(),
                        b.type_name()
                    ),
                ))
            }
        };
        Ok(r)
    }
}

fn scalar(v: &Value) -> f64 {
    v.as_real().unwrap_or(f64::NAN)
}

/// The scalar operation behind an arithmetic operator (`.*` is `*`).
fn arith(op: BinOp, x: f64, y: f64) -> f64 {
    match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul | BinOp::ElMul => x * y,
        BinOp::Div | BinOp::ElDiv => x / y,
        BinOp::Mod => x % y,
        BinOp::Pow => x.powf(y),
        _ => f64::NAN,
    }
}
