//! Static checks (Phase 2.2). Every error points at `file:line:column`.
//!
//! - **Types and sizes.** `int`, `real`, `vector[n]`, `matrix[r, c]` and
//!   tuples; `int` promotes to `real`. With `dim` bound to the block size,
//!   sizes made of literals, `dim` and integer arithmetic are checked here;
//!   other sizes are checked at run time.
//! - **Terminals.** Every path through `step { }` ends in exactly one
//!   `propose()` or `move()`: none may be missing, none may follow another,
//!   no statement may run after one, and none may sit inside a loop (the
//!   count would depend on the data). They are only allowed in `step`.
//! - **`@adapt`.** Only on tuning declarations; the tag is `step_size` (on a
//!   `real`, with the author's `target_accept = <real in (0, 1)>`, which
//!   is required: the right target depends on the kernel) or `metric` (on a
//!   `vector[dim]`); at most one of each.
//! - **Scopes.** Tuning values are read-only to the kernel (adaptation and
//!   `--set` change them); context values (`q`, `lp`, ...) are read-only;
//!   names may not shadow a variable, a function or a built-in. Tuning
//!   defaults may use only literals, `dim`, earlier tuning values and
//!   built-ins that need no running chain.

use std::collections::{BTreeSet, HashMap};

use super::ast::*;
use super::builtins::{Kind, Registry};
use super::types::{Size, Ty};
use super::LapError;

/// What the runtime needs to know about one tuning value.
#[derive(Clone, Debug, PartialEq)]
pub struct TuningInfo {
    pub name: String,
    pub ty: Ty,
    pub adapt: Option<AdaptKind>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdaptKind {
    StepSize,
    Metric,
}

/// A kernel that passed every check.
#[derive(Clone, Debug)]
pub struct Checked {
    pub file: String,
    pub kernel: KernelFile,
    pub tuning: Vec<TuningInfo>,
    /// Index into `tuning` of the `@adapt(step_size)` value.
    pub step_size: Option<usize>,
    /// Index into `tuning` of the `@adapt(metric)` value.
    pub metric: Option<usize>,
    /// `@adapt(step_size, target_accept = ...)`, if given.
    pub target_accept: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ctx {
    TuningDefault,
    StateDefault,
    Init,
    Step,
    Function,
}

#[derive(Clone, Debug)]
struct Var {
    ty: Ty,
    writable: bool,
    what: &'static str,
    span: Span,
}

#[derive(Clone, Debug)]
struct FunSig {
    ret: Ty,
    params: Vec<Ty>,
}

struct Checker<'r> {
    file: String,
    reg: &'r Registry,
    dim: Option<usize>,
    errors: Vec<LapError>,
    funs: HashMap<String, FunSig>,
    scopes: Vec<HashMap<String, Var>>,
    ctx: Ctx,
    fun_ret: Option<Ty>,
}

/// Check a parsed kernel. `dim` is the size of the block it will update,
/// when known (Level 1: the model's unconstrained dimension).
pub fn check(
    file: &str,
    k: &KernelFile,
    dim: Option<usize>,
    reg: &Registry,
) -> Result<Checked, Vec<LapError>> {
    let mut c = Checker {
        file: file.to_string(),
        reg,
        dim,
        errors: vec![],
        funs: HashMap::new(),
        scopes: vec![HashMap::new()],
        ctx: Ctx::TuningDefault,
        fun_ret: None,
    };
    let checked = c.kernel(k);
    if c.errors.is_empty() {
        Ok(checked)
    } else {
        c.errors.sort_by_key(|e| (e.span.line, e.span.col));
        Err(c.errors)
    }
}

const TERMINALS: &str = "propose()/move()";

impl<'r> Checker<'r> {
    fn err(&mut self, span: Span, msg: impl Into<String>) {
        self.errors.push(LapError::new(&self.file, span, msg));
    }

    // ---- kernel ------------------------------------------------------------

    fn kernel(&mut self, k: &KernelFile) -> Checked {
        let mut seen: Vec<Section> = vec![];
        for (s, sp) in &k.sections {
            if seen.contains(s) {
                self.err(*sp, format!("duplicate `{}` block", s.name()));
            }
            seen.push(*s);
        }
        if k.step.is_none() {
            self.err(
                k.span,
                format!("kernel `{}` has no `step {{ }}` block", k.name),
            );
        }

        // Function signatures first, so any block can call any function.
        for f in &k.functions {
            if self.reg.get(&f.name).is_some() {
                self.err(
                    f.span,
                    format!("function `{}` has the name of a built-in", f.name),
                );
            } else if self.funs.contains_key(&f.name) {
                self.err(f.span, format!("function `{}` is defined twice", f.name));
            }
            let sig = FunSig {
                ret: self.unsized_ty(&f.ret),
                params: f.params.iter().map(|(t, _)| self.unsized_ty(t)).collect(),
            };
            self.funs.insert(f.name.clone(), sig);
        }

        // Tuning.
        let mut tuning = vec![];
        let (mut step_size, mut metric, mut target_accept) = (None, None, None);
        for d in &k.tuning {
            self.ctx = Ctx::TuningDefault;
            let ty = self.declared_ty(&d.ty, d.span);
            self.bounds(&d.bounds);
            let adapt = d.adapt.as_ref().and_then(|t| self.adapt_tag(t, d, &ty));
            match adapt {
                Some(AdaptKind::StepSize) if step_size.is_some() => self.err(
                    d.adapt.as_ref().unwrap().span,
                    "only one tuning value may be tagged @adapt(step_size)",
                ),
                Some(AdaptKind::Metric) if metric.is_some() => self.err(
                    d.adapt.as_ref().unwrap().span,
                    "only one tuning value may be tagged @adapt(metric)",
                ),
                Some(AdaptKind::StepSize) => {
                    step_size = Some(tuning.len());
                    target_accept = self.target_accept(d.adapt.as_ref().unwrap());
                }
                Some(AdaptKind::Metric) => metric = Some(tuning.len()),
                None => {}
            }
            match &d.init {
                Some(e) => {
                    if let Some(t) = self.expr(e) {
                        self.assignable(&ty, &t, e.span, &format!("tuning value `{}`", d.name));
                    }
                }
                None if adapt == Some(AdaptKind::Metric) => {} // defaults to all ones
                None => self.err(
                    d.span,
                    format!(
                        "tuning value `{}` needs a default (`= ...`); --set can override it",
                        d.name
                    ),
                ),
            }
            self.declare(&d.name, ty.clone(), false, "tuning value", d.span);
            tuning.push(TuningInfo {
                name: d.name.clone(),
                ty,
                adapt,
                span: d.span,
            });
        }

        // State.
        self.ctx = Ctx::StateDefault;
        for d in &k.state {
            self.decl(d, true, "state variable");
        }

        // Function bodies: parameters, other functions and built-ins only.
        for f in &k.functions {
            self.ctx = Ctx::Function;
            let saved = std::mem::replace(&mut self.scopes, vec![HashMap::new()]);
            let sig = self.funs[&f.name].clone();
            for ((_, name), t) in f.params.iter().zip(&sig.params) {
                self.declare(name, t.clone(), false, "parameter", f.span);
            }
            self.fun_ret = Some(sig.ret.clone());
            self.block(&f.body, true);
            if sig.ret != Ty::Void && !always_returns(&f.body.stmts) {
                self.err(
                    f.span,
                    format!(
                        "function `{}` can end without returning a {}",
                        f.name, sig.ret
                    ),
                );
            }
            self.fun_ret = None;
            self.scopes = saved;
        }

        if let Some(b) = &k.init {
            self.ctx = Ctx::Init;
            self.block(b, true);
        }
        if let Some(b) = &k.step {
            self.ctx = Ctx::Step;
            self.block(b, true);
            let out = self.terminals(&b.stmts, BTreeSet::from([0u8]));
            if out.contains(&0) {
                let msg = if out.len() == 1 {
                    format!(
                        "step {{ }} never calls {TERMINALS}: every step must end in exactly one"
                    )
                } else {
                    format!("some path through step {{ }} ends without {TERMINALS}")
                };
                self.err(b.span, msg);
            }
        }

        Checked {
            file: self.file.clone(),
            kernel: k.clone(),
            tuning,
            step_size,
            metric,
            target_accept,
        }
    }

    fn adapt_tag(&mut self, t: &AdaptTag, d: &Decl, ty: &Ty) -> Option<AdaptKind> {
        let kind = match t.name.as_str() {
            "step_size" => AdaptKind::StepSize,
            "metric" => AdaptKind::Metric,
            other => {
                self.err(
                    t.span,
                    format!("unknown @adapt tag `{other}`: expected step_size or metric"),
                );
                return None;
            }
        };
        match kind {
            AdaptKind::StepSize => {
                if *ty != Ty::Real {
                    self.err(
                        t.span,
                        format!("@adapt(step_size) needs a real, `{}` is {ty}", d.name),
                    );
                }
                for (n, _) in &t.args {
                    if n != "target_accept" {
                        self.err(
                            t.span,
                            format!("@adapt(step_size) takes only target_accept, not `{n}`"),
                        );
                    }
                }
                if !t.args.iter().any(|(n, _)| n == "target_accept") {
                    self.err(
                        t.span,
                        "@adapt(step_size) needs the kernel's default acceptance target, e.g. \
                         @adapt(step_size, target_accept = 0.574); adapt(k, target_accept = ...) \
                         in a .lapsampler can override it",
                    );
                }
            }
            AdaptKind::Metric => {
                let ok = match ty {
                    Ty::Vector(Size::Known(n)) => Some(*n) == self.dim || self.dim.is_none(),
                    Ty::Vector(Size::Unknown) => {
                        matches!(&d.ty, TypeExpr::Vector(Some(e)) if e.kind == ExprKind::Var("dim".into()))
                    }
                    _ => false,
                };
                if !ok {
                    self.err(
                        t.span,
                        format!("@adapt(metric) needs a vector[dim], `{}` is {ty}", d.name),
                    );
                }
                if !t.args.is_empty() {
                    self.err(t.span, "@adapt(metric) takes no arguments");
                }
            }
        }
        Some(kind)
    }

    fn target_accept(&mut self, t: &AdaptTag) -> Option<f64> {
        let (_, e) = t.args.iter().find(|(n, _)| n == "target_accept")?;
        let v = match &e.kind {
            ExprKind::Real(x) => Some(*x),
            ExprKind::Int(i) => Some(*i as f64),
            _ => None,
        };
        match v {
            Some(x) if x > 0.0 && x < 1.0 => Some(x),
            _ => {
                self.err(
                    e.span,
                    "target_accept must be a number strictly between 0 and 1",
                );
                None
            }
        }
    }

    // ---- scopes --------------------------------------------------------------

    fn lookup(&self, name: &str) -> Option<&Var> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    fn declare(&mut self, name: &str, ty: Ty, writable: bool, what: &'static str, span: Span) {
        if let Some(v) = self.lookup(name) {
            let (vw, vs) = (v.what, v.span);
            self.err(
                span,
                format!("`{name}` is already declared as a {vw} at {vs}"),
            );
            return;
        }
        if self.reg.get(name).is_some() {
            self.err(
                span,
                format!("`{name}` is a built-in and cannot be redeclared"),
            );
            return;
        }
        if self.funs.contains_key(name) {
            self.err(
                span,
                format!("`{name}` is a function and cannot be redeclared"),
            );
            return;
        }
        self.scopes.last_mut().unwrap().insert(
            name.to_string(),
            Var {
                ty,
                writable,
                what,
                span,
            },
        );
    }

    // ---- types ---------------------------------------------------------------

    fn const_int(&self, e: &Expr) -> Option<i64> {
        match &e.kind {
            ExprKind::Int(i) => Some(*i),
            ExprKind::Var(n) if n == "dim" && self.lookup(n).is_none() => {
                self.dim.map(|d| d as i64)
            }
            ExprKind::Unary(UnOp::Neg, x) => self.const_int(x).map(|v| -v),
            ExprKind::Binary(op, a, b) => {
                let (a, b) = (self.const_int(a)?, self.const_int(b)?);
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div if b != 0 => Some(a / b),
                    BinOp::Mod if b != 0 => Some(a % b),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn size(&mut self, e: &Expr) -> Size {
        if let Some(t) = self.expr(e) {
            if t != Ty::Int {
                self.err(e.span, format!("a size must be an int, this is {t}"));
            }
        }
        match self.const_int(e) {
            Some(n) if n < 0 => {
                self.err(e.span, format!("size {n} is negative"));
                Size::Unknown
            }
            Some(n) => Size::Known(n as usize),
            None => Size::Unknown,
        }
    }

    fn declared_ty(&mut self, t: &TypeExpr, span: Span) -> Ty {
        match t {
            TypeExpr::Void => {
                self.err(span, "a variable cannot be void");
                Ty::Void
            }
            TypeExpr::Int => Ty::Int,
            TypeExpr::Real => Ty::Real,
            TypeExpr::Vector(Some(n)) => Ty::Vector(self.size(n)),
            TypeExpr::Vector(None) => Ty::Vector(Size::Unknown),
            TypeExpr::Matrix(Some((r, c))) => Ty::Matrix(self.size(r), self.size(c)),
            TypeExpr::Matrix(None) => Ty::Matrix(Size::Unknown, Size::Unknown),
            TypeExpr::Tuple(ts) => {
                Ty::Tuple(ts.iter().map(|t| self.declared_ty(t, span)).collect())
            }
        }
    }

    fn unsized_ty(&self, t: &TypeExpr) -> Ty {
        match t {
            TypeExpr::Void => Ty::Void,
            TypeExpr::Int => Ty::Int,
            TypeExpr::Real => Ty::Real,
            TypeExpr::Vector(_) => Ty::Vector(Size::Unknown),
            TypeExpr::Matrix(_) => Ty::Matrix(Size::Unknown, Size::Unknown),
            TypeExpr::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.unsized_ty(t)).collect()),
        }
    }

    fn bounds(&mut self, b: &Bounds) {
        for e in [&b.lower, &b.upper].into_iter().flatten() {
            if let Some(t) = self.expr(e) {
                if !t.is_scalar() {
                    self.err(e.span, format!("a bound must be a number, this is {t}"));
                }
            }
        }
    }

    fn assignable(&mut self, target: &Ty, value: &Ty, span: Span, what: &str) {
        if !target.accepts(value) {
            self.err(span, format!("{what} is {target}, but this is {value}"));
        }
    }

    // ---- statements ------------------------------------------------------------

    fn decl(&mut self, d: &Decl, writable: bool, what: &'static str) {
        if let Some(t) = &d.adapt {
            self.err(t.span, "@adapt(...) is only allowed on tuning declarations");
        }
        if d.bounds.lower.is_some() || d.bounds.upper.is_some() {
            self.bounds(&d.bounds);
        }
        let ty = self.declared_ty(&d.ty, d.span);
        if let Some(e) = &d.init {
            if let Some(t) = self.expr(e) {
                self.assignable(&ty, &t, e.span, &format!("`{}`", d.name));
            }
        }
        self.declare(&d.name, ty, writable, what, d.span);
    }

    fn block(&mut self, b: &Block, new_scope: bool) {
        if new_scope {
            self.scopes.push(HashMap::new());
        }
        for s in &b.stmts {
            self.stmt(s);
        }
        if new_scope {
            self.scopes.pop();
        }
    }

    fn stmt_in_scope(&mut self, s: &Stmt) {
        self.scopes.push(HashMap::new());
        self.stmt(s);
        self.scopes.pop();
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Decl(d) => self.decl(d, true, "local variable"),
            Stmt::Block(b) => self.block(b, true),
            Stmt::Expr(e) => {
                let ExprKind::Call { name, .. } = &e.kind else {
                    self.err(e.span, "an expression on its own does nothing; only a function call can be a statement");
                    self.expr(e);
                    return;
                };
                if let Some(t) = self.expr(e) {
                    if t != Ty::Void {
                        self.err(
                            e.span,
                            format!("the {t} returned by `{name}(...)` is not used"),
                        );
                    }
                }
            }
            Stmt::Assign {
                target, op, value, ..
            } => {
                let tt = self.lvalue(target);
                let vt = self.expr(value);
                if let (Some(tt), Some(vt)) = (tt, vt) {
                    match op {
                        AssignOp::Set => {
                            self.assignable(&tt, &vt, value.span, &format!("`{}`", target.name))
                        }
                        _ => {
                            let bin = match op {
                                AssignOp::Add => BinOp::Add,
                                AssignOp::Sub => BinOp::Sub,
                                AssignOp::Mul => BinOp::Mul,
                                _ => BinOp::Div,
                            };
                            if let Some(r) = self.binop_ty(bin, &tt, &vt, value.span) {
                                self.assignable(&tt, &r, value.span, &format!("`{}`", target.name));
                            }
                        }
                    }
                }
            }
            Stmt::Destructure { targets, value, .. } => {
                let vt = self.expr(value);
                let elems = match &vt {
                    Some(Ty::Tuple(ts)) => Some(ts.clone()),
                    Some(t) => {
                        self.err(
                            value.span,
                            format!("destructuring needs a tuple, this is {t}"),
                        );
                        None
                    }
                    None => None,
                };
                if let Some(ts) = &elems {
                    if ts.len() != targets.len() {
                        self.err(
                            value.span,
                            format!(
                                "{} names on the left, {} values on the right",
                                targets.len(),
                                ts.len()
                            ),
                        );
                    }
                }
                for (i, t) in targets.iter().enumerate() {
                    let et = elems.as_ref().and_then(|ts| ts.get(i)).cloned();
                    match t {
                        DestructTarget::Declare(te, name, sp) => {
                            let ty = self.declared_ty(te, *sp);
                            if let Some(et) = &et {
                                self.assignable(&ty, et, *sp, &format!("`{name}`"));
                            }
                            self.declare(name, ty, true, "local variable", *sp);
                        }
                        DestructTarget::Existing(name, sp) => {
                            let lv = LValue {
                                name: name.clone(),
                                index: vec![],
                                span: *sp,
                            };
                            if let (Some(tt), Some(et)) = (self.lvalue(&lv), &et) {
                                self.assignable(&tt, et, *sp, &format!("`{name}`"));
                            }
                        }
                    }
                }
            }
            Stmt::If {
                cond, then, els, ..
            } => {
                self.condition(cond);
                self.stmt_in_scope(then);
                if let Some(e) = els {
                    self.stmt_in_scope(e);
                }
            }
            Stmt::For {
                var,
                lo,
                hi,
                body,
                span,
            } => {
                for e in [lo, hi] {
                    if let Some(t) = self.expr(e) {
                        if t != Ty::Int {
                            self.err(e.span, format!("a loop bound must be an int, this is {t}"));
                        }
                    }
                }
                self.scopes.push(HashMap::new());
                self.declare(var, Ty::Int, false, "loop variable", *span);
                self.stmt(body);
                self.scopes.pop();
            }
            Stmt::While { cond, body, .. } => {
                self.condition(cond);
                self.stmt_in_scope(body);
            }
            Stmt::Return(e, span) => match (self.ctx, self.fun_ret.clone()) {
                (Ctx::Function, Some(ret)) => match (e, &ret) {
                    (None, Ty::Void) => {}
                    (None, r) => self.err(
                        *span,
                        format!("this function returns a {r}; `return` needs a value"),
                    ),
                    (Some(e), Ty::Void) => {
                        self.err(e.span, "a void function cannot return a value")
                    }
                    (Some(e), r) => {
                        if let Some(t) = self.expr(e) {
                            self.assignable(r, &t, e.span, "the return value");
                        }
                    }
                },
                _ => self.err(
                    *span,
                    format!("`return` is only allowed in functions; a step ends with {TERMINALS}"),
                ),
            },
        }
    }

    fn condition(&mut self, e: &Expr) {
        if let Some(t) = self.expr(e) {
            if !t.is_scalar() {
                self.err(
                    e.span,
                    format!("a condition must be an int or real, this is {t}"),
                );
            }
        }
    }

    fn lvalue(&mut self, lv: &LValue) -> Option<Ty> {
        let Some(v) = self.lookup(&lv.name).cloned() else {
            if self.reg.get(&lv.name).is_some() {
                self.err(
                    lv.span,
                    format!("`{}` is a built-in and cannot be assigned", lv.name),
                );
            } else {
                self.err(lv.span, format!("unknown variable `{}`", lv.name));
            }
            return None;
        };
        if !v.writable {
            let why = match v.what {
                "tuning value" => "; adaptation and --set change tuning values",
                _ => "",
            };
            self.err(
                lv.span,
                format!("`{}` is a {} and cannot be assigned{why}", lv.name, v.what),
            );
            return None;
        }
        if lv.index.is_empty() {
            return Some(v.ty);
        }
        self.index_ty(&v.ty, &lv.index, lv.span)
    }

    fn index_ty(&mut self, base: &Ty, idx: &[Expr], span: Span) -> Option<Ty> {
        for e in idx {
            if let Some(t) = self.expr(e) {
                if t != Ty::Int {
                    self.err(e.span, format!("an index must be an int, this is {t}"));
                }
            }
        }
        match (base, idx.len()) {
            (Ty::Vector(_), 1) | (Ty::Matrix(_, _), 2) => Some(Ty::Real),
            (t, n) => {
                self.err(
                    span,
                    format!("cannot index a {t} with {n} index(es); use v[i] or m[i, j]"),
                );
                None
            }
        }
    }

    /// Possible terminal counts (0 or 1) on the paths through `stmts`.
    fn terminals(&mut self, stmts: &[Stmt], mut paths: BTreeSet<u8>) -> BTreeSet<u8> {
        for s in stmts {
            if paths.contains(&1) {
                let msg = if is_terminal_stmt(s, self.reg) {
                    format!("a second {TERMINALS} on this path: every path through step {{ }} must end in exactly one")
                } else {
                    format!("this statement can run after {TERMINALS}, which must be the last thing a step does")
                };
                self.err(s.span(), msg);
                // Reported; treat the path as ended so the missing-terminal
                // check does not report it again.
                return BTreeSet::from([1u8]);
            }
            paths = match s {
                s if is_terminal_stmt(s, self.reg) => paths.iter().map(|c| c + 1).collect(),
                Stmt::If { then, els, .. } => {
                    let mut a = self.terminals(std::slice::from_ref(then.as_ref()), paths.clone());
                    let b = match els {
                        Some(e) => self.terminals(std::slice::from_ref(e.as_ref()), paths.clone()),
                        None => paths.clone(),
                    };
                    a.extend(b);
                    a
                }
                Stmt::Block(b) => self.terminals(&b.stmts, paths),
                Stmt::For { body, .. } | Stmt::While { body, .. } => {
                    if let Some(sp) = find_terminal(body, self.reg) {
                        self.err(sp, format!("{TERMINALS} inside a loop: a step must end in exactly one, on every path"));
                    }
                    paths
                }
                _ => paths,
            };
        }
        paths
    }

    // ---- expressions -----------------------------------------------------------

    fn expr(&mut self, e: &Expr) -> Option<Ty> {
        match &e.kind {
            ExprKind::Int(_) => Some(Ty::Int),
            ExprKind::Real(_) => Some(Ty::Real),
            ExprKind::Var(name) => {
                if let Some(v) = self.lookup(name) {
                    return Some(v.ty.clone());
                }
                match self.reg.get(name).map(|en| (en.needs_chain, &en.kind)) {
                    Some((needs_chain, Kind::Value { ty, .. })) => {
                        let ty = ty(self.dim);
                        self.chain_allowed(name, needs_chain, e.span);
                        Some(ty)
                    }
                    Some((_, Kind::Function { .. })) => {
                        self.err(
                            e.span,
                            format!("`{name}` is a function; call it as {name}(...)"),
                        );
                        None
                    }
                    None => {
                        self.err(e.span, format!("unknown name `{name}`"));
                        None
                    }
                }
            }
            ExprKind::Tuple(es) => {
                let ts: Vec<Option<Ty>> = es.iter().map(|x| self.expr(x)).collect();
                ts.into_iter().collect::<Option<Vec<_>>>().map(Ty::Tuple)
            }
            ExprKind::TupleIndex(b, n) => match self.expr(b)? {
                Ty::Tuple(ts) if *n >= 1 && *n <= ts.len() => Some(ts[n - 1].clone()),
                Ty::Tuple(ts) => {
                    self.err(
                        e.span,
                        format!("tuple has {} elements; .{n} is out of range", ts.len()),
                    );
                    None
                }
                t => {
                    self.err(e.span, format!(".{n} needs a tuple, this is {t}"));
                    None
                }
            },
            ExprKind::Index(b, idx) => {
                let bt = self.expr(b)?;
                self.index_ty(&bt, idx, e.span)
            }
            ExprKind::Unary(op, x) => {
                let t = self.expr(x)?;
                match op {
                    UnOp::Neg
                        if matches!(t, Ty::Int | Ty::Real | Ty::Vector(_) | Ty::Matrix(_, _)) =>
                    {
                        Some(t)
                    }
                    UnOp::Not if t.is_scalar() => Some(Ty::Int),
                    _ => {
                        let sym = if *op == UnOp::Neg { "-" } else { "!" };
                        self.err(e.span, format!("cannot apply `{sym}` to a {t}"));
                        None
                    }
                }
            }
            ExprKind::Binary(op, a, b) => {
                let (ta, tb) = (self.expr(a), self.expr(b));
                self.binop_ty(*op, &ta?, &tb?, e.span)
            }
            ExprKind::Call {
                name,
                args,
                conditional,
            } => self.call(name, args, *conditional, e.span),
        }
    }

    fn chain_allowed(&mut self, name: &str, needs_chain: bool, span: Span) {
        if needs_chain && self.ctx == Ctx::TuningDefault {
            self.err(
                span,
                format!(
                    "`{name}` needs a running chain; a tuning default may use only literals, dim, \
                     earlier tuning values and functions such as rep_vector"
                ),
            );
        }
    }

    fn call(&mut self, name: &str, args: &[Expr], conditional: bool, span: Span) -> Option<Ty> {
        let tys: Vec<Option<Ty>> = args.iter().map(|a| self.expr(a)).collect();
        let consts: Vec<Option<i64>> = args.iter().map(|a| self.const_int(a)).collect();
        let is_lpdf = name.ends_with("_lpdf");
        if conditional && !is_lpdf {
            self.err(
                span,
                format!("`|` separates the outcome in a density call; `{name}` is not one"),
            );
        }
        if is_lpdf && args.len() > 1 && !conditional {
            self.err(
                span,
                format!("write {name}(y | ...): the outcome is separated by `|`, as in Stan"),
            );
        }
        let tys: Vec<Ty> = tys.into_iter().collect::<Option<_>>()?;
        if let Some(f) = self.funs.get(name).cloned() {
            if f.params.len() != tys.len() || f.params.iter().zip(&tys).any(|(p, t)| !p.accepts(t))
            {
                let want: Vec<String> = f.params.iter().map(|t| t.to_string()).collect();
                let got: Vec<String> = tys.iter().map(|t| t.to_string()).collect();
                self.err(
                    span,
                    format!(
                        "`{name}` takes ({}), got ({})",
                        want.join(", "),
                        got.join(", ")
                    ),
                );
                return None;
            }
            return Some(f.ret);
        }
        let Some(entry) = self.reg.get(name) else {
            self.err(span, format!("unknown function `{name}`"));
            return None;
        };
        let (sig, terminal) = match &entry.kind {
            Kind::Function { sig, terminal, .. } => (*sig, *terminal),
            Kind::Value { .. } => {
                self.err(span, format!("`{name}` is a value, not a function"));
                return None;
            }
        };
        let needs_chain = entry.needs_chain;
        if terminal && self.ctx != Ctx::Step {
            self.err(
                span,
                format!("`{name}()` ends a step and is only allowed in step {{ }}"),
            );
        }
        self.chain_allowed(name, needs_chain, span);
        match sig(&tys, &consts, self.dim) {
            Ok(t) => Some(t),
            Err(m) => {
                self.err(span, format!("{name}: {m}"));
                None
            }
        }
    }

    fn binop_ty(&mut self, op: BinOp, a: &Ty, b: &Ty, span: Span) -> Option<Ty> {
        let sym = op.symbol();
        let errors_before = self.errors.len();
        let num = |t: &Ty| matches!(t, Ty::Int | Ty::Real);
        let scal = |x: &Ty, y: &Ty| {
            if *x == Ty::Int && *y == Ty::Int {
                Ty::Int
            } else {
                Ty::Real
            }
        };
        let r = match op {
            BinOp::Or
            | BinOp::And
            | BinOp::Eq
            | BinOp::Ne
            | BinOp::Lt
            | BinOp::Le
            | BinOp::Gt
            | BinOp::Ge => (num(a) && num(b)).then_some(Ty::Int),
            BinOp::Mod => (*a == Ty::Int && *b == Ty::Int).then_some(Ty::Int),
            BinOp::Pow => (num(a) && num(b)).then_some(Ty::Real),
            BinOp::Add | BinOp::Sub => match (a, b) {
                (x, y) if num(x) && num(y) => Some(scal(x, y)),
                (Ty::Vector(n), Ty::Vector(m)) => self.same(*n, *m, sym, span).map(Ty::Vector),
                (Ty::Vector(n), s) | (s, Ty::Vector(n)) if num(s) => Some(Ty::Vector(*n)),
                (Ty::Matrix(r1, c1), Ty::Matrix(r2, c2)) => {
                    let r = self.same(*r1, *r2, sym, span)?;
                    Some(Ty::Matrix(r, self.same(*c1, *c2, sym, span)?))
                }
                (Ty::Matrix(r, c), s) | (s, Ty::Matrix(r, c)) if num(s) => Some(Ty::Matrix(*r, *c)),
                _ => None,
            },
            BinOp::Mul => {
                match (a, b) {
                    (x, y) if num(x) && num(y) => Some(scal(x, y)),
                    (s, t @ (Ty::Vector(_) | Ty::Matrix(_, _)))
                    | (t @ (Ty::Vector(_) | Ty::Matrix(_, _)), s)
                        if num(s) =>
                    {
                        Some(t.clone())
                    }
                    (Ty::Matrix(r, c), Ty::Vector(n)) => {
                        self.same(*c, *n, sym, span)?;
                        Some(Ty::Vector(*r))
                    }
                    (Ty::Matrix(r, c), Ty::Matrix(r2, c2)) => {
                        self.same(*c, *r2, sym, span)?;
                        Some(Ty::Matrix(*r, *c2))
                    }
                    (Ty::Vector(_), Ty::Vector(_)) => {
                        self.err(span, "vector * vector is not defined; use .* (elementwise) or dot_product(a, b)");
                        return None;
                    }
                    _ => None,
                }
            }
            BinOp::Div => match (a, b) {
                (x, y) if num(x) && num(y) => Some(scal(x, y)),
                (t @ (Ty::Vector(_) | Ty::Matrix(_, _)), s) if num(s) => Some(t.clone()),
                _ => None,
            },
            BinOp::ElMul | BinOp::ElDiv => match (a, b) {
                (Ty::Vector(n), Ty::Vector(m)) => self.same(*n, *m, sym, span).map(Ty::Vector),
                (Ty::Matrix(r1, c1), Ty::Matrix(r2, c2)) => {
                    let r = self.same(*r1, *r2, sym, span)?;
                    Some(Ty::Matrix(r, self.same(*c1, *c2, sym, span)?))
                }
                (Ty::Vector(n), s) | (s, Ty::Vector(n)) if num(s) => Some(Ty::Vector(*n)),
                _ => None,
            },
        };
        if r.is_none() && self.errors.len() == errors_before {
            self.err(span, format!("`{sym}` is not defined for {a} and {b}"));
        }
        r
    }

    fn same(&mut self, a: Size, b: Size, sym: &str, span: Span) -> Option<Size> {
        if a.agrees(b) {
            Some(a.meet(b))
        } else {
            self.err(span, format!("size mismatch in `{sym}`: {a} and {b}"));
            None
        }
    }
}

fn is_terminal_stmt(s: &Stmt, reg: &Registry) -> bool {
    matches!(s, Stmt::Expr(Expr { kind: ExprKind::Call { name, .. }, .. })
        if matches!(reg.get(name).map(|e| &e.kind), Some(Kind::Function { terminal: true, .. })))
}

fn find_terminal(s: &Stmt, reg: &Registry) -> Option<Span> {
    if is_terminal_stmt(s, reg) {
        return Some(s.span());
    }
    match s {
        Stmt::Block(b) => b.stmts.iter().find_map(|x| find_terminal(x, reg)),
        Stmt::If { then, els, .. } => {
            find_terminal(then, reg).or_else(|| els.as_ref().and_then(|e| find_terminal(e, reg)))
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => find_terminal(body, reg),
        _ => None,
    }
}

fn always_returns(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Return(..) => true,
        Stmt::Block(b) => always_returns(&b.stmts),
        Stmt::If {
            then, els: Some(e), ..
        } => {
            always_returns(std::slice::from_ref(then.as_ref()))
                && always_returns(std::slice::from_ref(e.as_ref()))
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse;
    use super::*;

    fn run(src: &str, dim: Option<usize>) -> Result<Checked, Vec<String>> {
        let k = parse("k.lapkernel", src).unwrap_or_else(|e| panic!("parse: {e}"));
        check("k.lapkernel", &k, dim, &Registry::standard())
            .map_err(|es| es.iter().map(|e| e.to_string()).collect())
    }

    /// The single error a file produces, as "line:col: message".
    fn one_error(src: &str, dim: Option<usize>) -> String {
        let es = run(src, dim).expect_err("should fail");
        assert_eq!(es.len(), 1, "expected one error, got {es:#?}");
        es[0].trim_start_matches("k.lapkernel:").to_string()
    }

    fn kernel(step: &str) -> String {
        format!("kernel k {{\n  tuning {{\n    real<lower=0> eps = 0.5 @adapt(step_size, target_accept = 0.5);\n  }}\n  step {{\n{step}\n  }}\n}}\n")
    }

    const MALA: &str = r#"kernel mala {
  tuning {
    real<lower=0> eps = 1 @adapt(step_size, target_accept = 0.574);
    vector<lower=0>[dim] m = rep_vector(1, dim) @adapt(metric);
  }
  step {
    vector[dim] mu_fwd = q + 0.5 * eps^2 * m .* grad;
    vector[dim] q_new = mu_fwd + eps * sqrt(m) .* std_normal_rng(dim);
    (real lp_new, vector[dim] g_new) = log_density_grad(q_new);
    vector[dim] mu_bwd = q_new + 0.5 * eps^2 * m .* g_new;
    propose(q_new, normal_lpdf(q | mu_bwd, eps * sqrt(m)) - normal_lpdf(q_new | mu_fwd, eps * sqrt(m)));
  }
}
"#;

    #[test]
    fn a_valid_kernel_checks_with_and_without_dim() {
        for dim in [None, Some(10)] {
            let c = run(MALA, dim).unwrap_or_else(|e| panic!("{e:#?}"));
            assert_eq!(c.step_size, Some(0));
            assert_eq!(c.metric, Some(1));
            assert_eq!(c.target_accept, Some(0.574));
            assert_eq!(
                c.tuning[1].ty,
                Ty::Vector(dim.map_or(Size::Unknown, Size::Known))
            );
        }
    }

    #[test]
    fn branches_that_each_propose_are_fine() {
        let src = kernel("    if (lp > 0) {\n      propose(q, 0);\n    } else if (lp < -5) {\n      move(q);\n    } else {\n      { propose(q + eps, 1); }\n    }");
        run(&src, Some(3)).unwrap_or_else(|e| panic!("{e:#?}"));
    }

    // ---- types and sizes ---------------------------------------------------

    #[test]
    fn type_mismatch() {
        let e = one_error(&kernel("    real x = q;\n    move(q);"), None);
        assert_eq!(e, "6:14: `x` is real, but this is vector");
    }

    #[test]
    fn size_mismatch_with_dim_bound() {
        let src = kernel("    vector[3] v = rep_vector(0, 3);\n    move(q + v);");
        let e = one_error(&src, Some(10));
        assert_eq!(e, "7:10: size mismatch in `+`: 10 and 3");
        // Without dim the sizes are only known at run time: no error.
        run(&src, None).unwrap();
    }

    #[test]
    fn vector_times_vector_is_explained() {
        let e = one_error(&kernel("    move(q * grad);"), None);
        assert_eq!(
            e,
            "6:10: vector * vector is not defined; use .* (elementwise) or dot_product(a, b)"
        );
    }

    #[test]
    fn unknown_names_and_bad_builtin_arguments() {
        assert_eq!(
            one_error(&kernel("    move(q + foo);"), None),
            "6:14: unknown name `foo`"
        );
        assert_eq!(
            one_error(&kernel("    move(bar(q));"), None),
            "6:10: unknown function `bar`"
        );
        assert_eq!(
            one_error(&kernel("    real s = sum(1.5);\n    move(q);"), None),
            "6:14: sum: expected sum(vector v), got (real)"
        );
    }

    #[test]
    fn density_calls_use_the_bar() {
        assert_eq!(
            one_error(
                &kernel("    real a = normal_lpdf(q, q, 1);\n    move(q);"),
                None
            ),
            "6:14: write normal_lpdf(y | ...): the outcome is separated by `|`, as in Stan"
        );
        assert_eq!(
            one_error(
                &kernel("    real a = dot_product(q | q);\n    move(q);"),
                None
            ),
            "6:14: `|` separates the outcome in a density call; `dot_product` is not one"
        );
    }

    #[test]
    fn tuning_values_are_read_only() {
        assert_eq!(
            one_error(&kernel("    eps = 1;\n    move(q);"), None),
            "6:5: `eps` is a tuning value and cannot be assigned; adaptation and --set change tuning values"
        );
        assert_eq!(
            one_error(&kernel("    q = q;\n    move(q);"), None),
            "6:5: `q` is a built-in and cannot be assigned"
        );
    }

    #[test]
    fn no_shadowing() {
        assert_eq!(
            one_error(&kernel("    real eps = 1;\n    move(q);"), None),
            "6:5: `eps` is already declared as a tuning value at 3:5"
        );
        assert_eq!(
            one_error(&kernel("    real lp = 1;\n    move(q);"), None),
            "6:5: `lp` is a built-in and cannot be redeclared"
        );
    }

    // ---- terminals -----------------------------------------------------------

    #[test]
    fn step_without_a_terminal() {
        assert_eq!(
            one_error(&kernel("    real x = 1;"), None),
            "5:8: step { } never calls propose()/move(): every step must end in exactly one"
        );
    }

    #[test]
    fn a_path_without_a_terminal() {
        assert_eq!(
            one_error(
                &kernel("    if (lp > 0) {\n      propose(q, 0);\n    }"),
                None
            ),
            "5:8: some path through step { } ends without propose()/move()"
        );
    }

    #[test]
    fn two_terminals_on_one_path() {
        assert_eq!(
            one_error(&kernel("    propose(q, 0);\n    move(q);"), None),
            "7:5: a second propose()/move() on this path: every path through step { } must end in exactly one"
        );
        // ... including after an if whose branch already proposed.
        assert_eq!(
            one_error(&kernel("    if (lp > 0) move(q);\n    propose(q, 0);"), None),
            "7:5: a second propose()/move() on this path: every path through step { } must end in exactly one"
        );
    }

    #[test]
    fn statement_after_a_terminal() {
        assert_eq!(
            one_error(&kernel("    move(q);\n    real x = 1;"), None),
            "7:5: this statement can run after propose()/move(), which must be the last thing a step does"
        );
    }

    #[test]
    fn terminal_inside_a_loop() {
        let es = run(&kernel("    for (i in 1:3) {\n      move(q);\n    }"), None).unwrap_err();
        assert!(
            es.iter()
                .any(|e| e.contains(":7:7: propose()/move() inside a loop")),
            "{es:#?}"
        );
    }

    #[test]
    fn terminal_outside_step() {
        let src = "kernel k {\n  init {\n    move(q);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:5: `move()` ends a step and is only allowed in step { }"
        );
        let src = "kernel k {\n  functions {\n    void f(vector x) { propose(x, 0); }\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:24: `propose()` ends a step and is only allowed in step { }"
        );
    }

    #[test]
    fn return_in_step() {
        assert_eq!(
            one_error(&kernel("    return;\n    move(q);"), None),
            "6:5: `return` is only allowed in functions; a step ends with propose()/move()"
        );
    }

    // ---- @adapt --------------------------------------------------------------

    #[test]
    fn adapt_only_on_tuning() {
        let src = "kernel k {\n  state {\n    real s = 1 @adapt(step_size);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:16: @adapt(...) is only allowed on tuning declarations"
        );
        assert_eq!(
            one_error(
                &kernel("    real x = 1 @adapt(step_size);\n    move(q);"),
                None
            ),
            "6:16: @adapt(...) is only allowed on tuning declarations"
        );
    }

    #[test]
    fn step_size_tag_needs_a_target() {
        let src = "kernel k {\n  tuning {\n    real eps = 1 @adapt(step_size);\n  }\n  step { move(q); }\n}\n";
        let e = one_error(src, None);
        assert!(
            e.starts_with("3:18: @adapt(step_size) needs the kernel's default acceptance target"),
            "{e}"
        );
    }

    #[test]
    fn adapt_tag_must_be_known() {
        let src = "kernel k {\n  tuning {\n    real eps = 1 @adapt(stepsize);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:18: unknown @adapt tag `stepsize`: expected step_size or metric"
        );
    }

    #[test]
    fn adapt_tag_types() {
        let src = "kernel k {\n  tuning {\n    vector[dim] eps = rep_vector(1, dim) @adapt(step_size, target_accept = 0.5);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, Some(4)),
            "3:42: @adapt(step_size) needs a real, `eps` is vector[4]"
        );
        let src = "kernel k {\n  tuning {\n    vector[3] m = rep_vector(1, 3) @adapt(metric);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, Some(4)),
            "3:36: @adapt(metric) needs a vector[dim], `m` is vector[3]"
        );
        let src = "kernel k {\n  tuning {\n    real a = 1 @adapt(step_size, target_accept = 1.5);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:50: target_accept must be a number strictly between 0 and 1"
        );
        let src = "kernel k {\n  tuning {\n    real a = 1 @adapt(step_size, target_accept = 0.5);\n    real b = 1 @adapt(step_size, target_accept = 0.5);\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "4:16: only one tuning value may be tagged @adapt(step_size)"
        );
    }

    #[test]
    fn tuning_defaults_cannot_use_the_chain() {
        let src = "kernel k {\n  tuning {\n    real eps = lp;\n  }\n  step { move(q); }\n}\n";
        assert!(one_error(src, None).starts_with("3:16: `lp` needs a running chain"));
        let src = "kernel k {\n  tuning {\n    real eps = std_normal_rng();\n  }\n  step { move(q); }\n}\n";
        assert!(one_error(src, None).starts_with("3:16: `std_normal_rng` needs a running chain"));
        let src = "kernel k {\n  tuning {\n    real eps;\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:5: tuning value `eps` needs a default (`= ...`); --set can override it"
        );
    }

    // ---- structure -------------------------------------------------------------

    #[test]
    fn missing_and_duplicate_sections() {
        assert_eq!(
            one_error("kernel k {\n  tuning { }\n}\n", None),
            "1:1: kernel `k` has no `step { }` block"
        );
        assert_eq!(
            one_error(
                "kernel k {\n  step { move(q); }\n  step { move(q); }\n}\n",
                None
            ),
            "3:3: duplicate `step` block"
        );
    }

    #[test]
    fn functions_are_checked() {
        let src = "kernel k {\n  functions {\n    real f(real x) { if (x > 0) return x; }\n  }\n  step { move(q); }\n}\n";
        assert_eq!(
            one_error(src, None),
            "3:5: function `f` can end without returning a real"
        );
        let src = "kernel k {\n  functions {\n    real f(real x) { return x; }\n  }\n  step { move(q * f(q)); }\n}\n";
        assert_eq!(one_error(src, None), "5:19: `f` takes (real), got (vector)");
    }

    #[test]
    fn every_error_is_reported() {
        let es = run(
            &kernel("    real x = q;\n    eps = 2;\n    move(foo);"),
            None,
        )
        .unwrap_err();
        assert_eq!(es.len(), 3, "{es:#?}");
    }
}
