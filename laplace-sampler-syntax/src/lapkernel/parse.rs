//! Source text -> [`KernelFile`]. Parse errors carry the line and column
//! pest reports; everything later (2.2 checks, runtime) uses node spans.

use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;
use pest_derive::Parser;

use super::ast::*;
use super::LapError;

#[derive(Parser)]
#[grammar = "lapkernel/grammar.pest"]
struct KernelParser;

fn span(p: &Pair<Rule>) -> Span {
    let (line, col) = p.as_span().start_pos().line_col();
    Span { line, col }
}

/// Human names for the rules pest lists in "expected ..." messages.
fn rule_name(r: &Rule) -> String {
    match r {
        Rule::EOI => "end of file".into(),
        Rule::kw_kernel => "`kernel`".into(),
        Rule::ident => "a name".into(),
        Rule::expr | Rule::operand | Rule::arith_expr => "an expression".into(),
        Rule::stmt => "a statement".into(),
        Rule::decl => "a declaration".into(),
        Rule::sized_type => "a type".into(),
        Rule::fundef => "a function definition".into(),
        Rule::tuning_block => "`tuning {`".into(),
        Rule::state_block => "`state {`".into(),
        Rule::init_block => "`init {`".into(),
        Rule::functions_block => "`functions {`".into(),
        Rule::step_block => "`step {`".into(),
        Rule::block => "`{`".into(),
        Rule::adapt_tag => "`@adapt(...)`".into(),
        Rule::assign_op => "`=`".into(),
        Rule::index => "`[`".into(),
        Rule::tuple_idx => "`.N`".into(),
        Rule::args => "arguments".into(),
        Rule::cond_bar => "`|`".into(),
        Rule::bounds => "`<lower=..., upper=...>`".into(),
        Rule::or
        | Rule::and
        | Rule::eq
        | Rule::ne
        | Rule::le
        | Rule::ge
        | Rule::lt
        | Rule::gt
        | Rule::add
        | Rule::sub
        | Rule::elmul
        | Rule::eldiv
        | Rule::mul
        | Rule::div
        | Rule::modulo
        | Rule::pow => "an operator".into(),
        other => format!("{other:?}"),
    }
}

/// pest lists every alternative rule; several map to the same human name
/// ("an operator"), so collapse repeats: "expected A, A, or B" -> "expected
/// A or B".
fn dedupe(msg: &str) -> String {
    let Some(rest) = msg.strip_prefix("expected ") else {
        return msg.to_string();
    };
    let mut items: Vec<&str> = vec![];
    for part in rest.split(", ") {
        let part = part.strip_prefix("or ").unwrap_or(part);
        for item in part.split(" or ") {
            if !items.contains(&item) {
                items.push(item);
            }
        }
    }
    match items.len() {
        0 => msg.to_string(),
        1 => format!("expected {}", items[0]),
        n => format!("expected {} or {}", items[..n - 1].join(", "), items[n - 1]),
    }
}

/// Parse a whole file. `file` is only used in messages.
pub fn parse(file: &str, src: &str) -> Result<KernelFile, LapError> {
    let mut pairs = KernelParser::parse(Rule::file, src).map_err(|e| {
        let (line, col) = match e.line_col {
            pest::error::LineColLocation::Pos(p) => p,
            pest::error::LineColLocation::Span(p, _) => p,
        };
        let e = e.renamed_rules(rule_name);
        let msg = match &e.variant {
            pest::error::ErrorVariant::ParsingError { .. } => dedupe(&e.variant.message()),
            pest::error::ErrorVariant::CustomError { message } => dedupe(message),
        };
        LapError::new(file, Span { line, col }, format!("syntax error: {msg}"))
    })?;
    let file_pair = pairs.next().expect("file rule");
    let kernel = file_pair.into_inner().next().expect("kernel rule");
    let mut k = build_kernel(kernel);
    k.docs = doc_comments(src);
    Ok(k)
}

/// `//@laplace`, `//@brief text`, `//@param name text` lines before the
/// `kernel` keyword.
pub fn doc_comments(src: &str) -> Docs {
    let mut d = Docs::default();
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("kernel") {
            break;
        }
        let Some(rest) = t.strip_prefix("//@") else {
            continue;
        };
        let (tag, text) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let text = text.trim();
        match tag {
            "laplace" => d.laplace = true,
            "brief" => d.brief = Some(text.to_string()),
            "param" => {
                let (name, desc) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
                d.params.push((name.to_string(), desc.trim().to_string()));
            }
            _ => {}
        }
    }
    d
}

fn build_kernel(p: Pair<Rule>) -> KernelFile {
    let sp = span(&p);
    let mut it = p.into_inner();
    it.next(); // `kernel`
    let name = it.next().expect("name").as_str().to_string();
    let mut k = KernelFile {
        docs: Docs::default(),
        name,
        span: sp,
        tuning: vec![],
        state: vec![],
        init: None,
        functions: vec![],
        step: None,
        sections: vec![],
    };
    for s in it {
        let sec_span = span(&s);
        match s.as_rule() {
            Rule::tuning_block => {
                k.sections.push((Section::Tuning, sec_span));
                k.tuning.extend(s.into_inner().skip(1).map(build_decl));
            }
            Rule::state_block => {
                k.sections.push((Section::State, sec_span));
                k.state.extend(s.into_inner().skip(1).map(build_decl));
            }
            Rule::init_block => {
                k.sections.push((Section::Init, sec_span));
                k.init = Some(build_block(s.into_inner().nth(1).expect("block")));
            }
            Rule::functions_block => {
                k.sections.push((Section::Functions, sec_span));
                k.functions.extend(s.into_inner().skip(1).map(build_fundef));
            }
            Rule::step_block => {
                k.sections.push((Section::Step, sec_span));
                k.step = Some(build_block(s.into_inner().nth(1).expect("block")));
            }
            r => unreachable!("section {r:?}"),
        }
    }
    k
}

fn build_bounds(p: Pair<Rule>) -> Bounds {
    let mut b = Bounds::default();
    for bound in p.into_inner() {
        let mut it = bound.into_inner();
        let kw = it.next().expect("kw");
        let e = build_expr(it.next().expect("expr"));
        match kw.as_rule() {
            Rule::kw_lower => b.lower = Some(e),
            _ => b.upper = Some(e),
        }
    }
    b
}

/// A sized type plus its bounds.
fn build_sized_type(p: Pair<Rule>) -> (TypeExpr, Bounds) {
    let inner = p.into_inner().next().expect("type");
    let rule = inner.as_rule();
    let mut bounds = Bounds::default();
    let mut exprs = vec![];
    let mut tuple = vec![];
    for c in inner.into_inner() {
        match c.as_rule() {
            Rule::bounds => bounds = build_bounds(c),
            Rule::expr => exprs.push(build_expr(c)),
            Rule::sized_type => tuple.push(build_sized_type(c).0),
            _ => {}
        }
    }
    let ty = match rule {
        Rule::int_type => TypeExpr::Int,
        Rule::real_type => TypeExpr::Real,
        Rule::vector_type => TypeExpr::Vector(Some(Box::new(exprs.remove(0)))),
        Rule::matrix_type => {
            let c = exprs.pop().expect("cols");
            let r = exprs.pop().expect("rows");
            TypeExpr::Matrix(Some((Box::new(r), Box::new(c))))
        }
        Rule::tuple_type => TypeExpr::Tuple(tuple),
        r => unreachable!("type {r:?}"),
    };
    (ty, bounds)
}

fn build_fun_type(p: Pair<Rule>) -> TypeExpr {
    let inner = p.into_inner().next().expect("fun type");
    match inner.as_rule() {
        Rule::kw_void => TypeExpr::Void,
        Rule::kw_int => TypeExpr::Int,
        Rule::kw_real => TypeExpr::Real,
        Rule::kw_vector => TypeExpr::Vector(None),
        Rule::kw_matrix => TypeExpr::Matrix(None),
        Rule::fun_tuple_type => TypeExpr::Tuple(
            inner
                .into_inner()
                .filter(|c| c.as_rule() == Rule::fun_type)
                .map(build_fun_type)
                .collect(),
        ),
        r => unreachable!("fun type {r:?}"),
    }
}

fn build_decl(p: Pair<Rule>) -> Decl {
    let sp = span(&p);
    let mut it = p.into_inner();
    let (ty, bounds) = build_sized_type(it.next().expect("type"));
    let name = it.next().expect("name").as_str().to_string();
    let mut init = None;
    let mut adapt = None;
    for c in it {
        match c.as_rule() {
            Rule::expr => init = Some(build_expr(c)),
            Rule::adapt_tag => {
                let tsp = span(&c);
                let mut t = c.into_inner();
                let name = t.next().expect("tag").as_str().to_string();
                let args = t
                    .map(|a| {
                        let mut ai = a.into_inner();
                        let n = ai.next().expect("arg name").as_str().to_string();
                        (n, build_expr(ai.next().expect("arg value")))
                    })
                    .collect();
                adapt = Some(AdaptTag {
                    name,
                    args,
                    span: tsp,
                });
            }
            _ => {}
        }
    }
    Decl {
        ty,
        bounds,
        name,
        init,
        adapt,
        span: sp,
    }
}

fn build_fundef(p: Pair<Rule>) -> FunDef {
    let sp = span(&p);
    let mut it = p.into_inner();
    let ret = build_fun_type(it.next().expect("ret"));
    let name = it.next().expect("name").as_str().to_string();
    let mut params = vec![];
    let mut body = None;
    for c in it {
        match c.as_rule() {
            Rule::param => {
                let mut pi = c.into_inner();
                let t = build_fun_type(pi.next().expect("type"));
                params.push((t, pi.next().expect("name").as_str().to_string()));
            }
            Rule::block => body = Some(build_block(c)),
            _ => {}
        }
    }
    FunDef {
        ret,
        name,
        params,
        body: body.expect("body"),
        span: sp,
    }
}

fn build_block(p: Pair<Rule>) -> Block {
    let sp = span(&p);
    Block {
        stmts: p.into_inner().map(build_stmt).collect(),
        span: sp,
    }
}

fn build_stmt(p: Pair<Rule>) -> Stmt {
    let inner = p.into_inner().next().expect("stmt");
    let sp = span(&inner);
    match inner.as_rule() {
        Rule::decl => Stmt::Decl(build_decl(inner)),
        Rule::block => Stmt::Block(build_block(inner)),
        Rule::expr_stmt => Stmt::Expr(build_expr(inner.into_inner().next().expect("expr"))),
        Rule::return_stmt => Stmt::Return(
            inner
                .into_inner()
                .find(|c| c.as_rule() == Rule::expr)
                .map(build_expr),
            sp,
        ),
        Rule::assign => {
            let mut it = inner.into_inner();
            let lv = it.next().expect("lvalue");
            let lsp = span(&lv);
            let mut li = lv.into_inner();
            let name = li.next().expect("name").as_str().to_string();
            let index = li
                .next()
                .map(|ix| ix.into_inner().map(build_expr).collect())
                .unwrap_or_default();
            let op = match it.next().expect("op").as_str() {
                "+=" => AssignOp::Add,
                "-=" => AssignOp::Sub,
                "*=" => AssignOp::Mul,
                "/=" => AssignOp::Div,
                _ => AssignOp::Set,
            };
            Stmt::Assign {
                target: LValue {
                    name,
                    index,
                    span: lsp,
                },
                op,
                value: build_expr(it.next().expect("value")),
                span: sp,
            }
        }
        Rule::destructure => {
            let mut targets = vec![];
            let mut value = None;
            for c in inner.into_inner() {
                match c.as_rule() {
                    Rule::dtarget => {
                        let tsp = span(&c);
                        let mut ti = c.into_inner();
                        let first = ti.next().expect("target");
                        if first.as_rule() == Rule::sized_type {
                            let (t, _) = build_sized_type(first);
                            let n = ti.next().expect("name").as_str().to_string();
                            targets.push(DestructTarget::Declare(t, n, tsp));
                        } else {
                            targets.push(DestructTarget::Existing(first.as_str().to_string(), tsp));
                        }
                    }
                    Rule::expr => value = Some(build_expr(c)),
                    _ => {}
                }
            }
            Stmt::Destructure {
                targets,
                value: value.expect("value"),
                span: sp,
            }
        }
        Rule::if_stmt => {
            let mut cond = None;
            let mut branches = vec![];
            for c in inner.into_inner() {
                match c.as_rule() {
                    Rule::expr => cond = Some(build_expr(c)),
                    Rule::stmt => branches.push(Box::new(build_stmt(c))),
                    _ => {}
                }
            }
            let mut b = branches.into_iter();
            Stmt::If {
                cond: cond.expect("cond"),
                then: b.next().expect("then"),
                els: b.next(),
                span: sp,
            }
        }
        Rule::for_stmt => {
            let mut var = String::new();
            let mut ex = vec![];
            let mut body = None;
            for c in inner.into_inner() {
                match c.as_rule() {
                    Rule::ident => var = c.as_str().to_string(),
                    Rule::expr => ex.push(build_expr(c)),
                    Rule::stmt => body = Some(Box::new(build_stmt(c))),
                    _ => {}
                }
            }
            let hi = ex.pop().expect("hi");
            let lo = ex.pop().expect("lo");
            Stmt::For {
                var,
                lo,
                hi,
                body: body.expect("body"),
                span: sp,
            }
        }
        Rule::while_stmt => {
            let mut cond = None;
            let mut body = None;
            for c in inner.into_inner() {
                match c.as_rule() {
                    Rule::expr => cond = Some(build_expr(c)),
                    Rule::stmt => body = Some(Box::new(build_stmt(c))),
                    _ => {}
                }
            }
            Stmt::While {
                cond: cond.expect("cond"),
                body: body.expect("body"),
                span: sp,
            }
        }
        r => unreachable!("stmt {r:?}"),
    }
}

fn pratt() -> PrattParser<Rule> {
    // Lowest to highest, as in Stan.
    PrattParser::new()
        .op(Op::infix(Rule::or, Assoc::Left))
        .op(Op::infix(Rule::and, Assoc::Left))
        .op(Op::infix(Rule::eq, Assoc::Left) | Op::infix(Rule::ne, Assoc::Left))
        .op(Op::infix(Rule::lt, Assoc::Left)
            | Op::infix(Rule::le, Assoc::Left)
            | Op::infix(Rule::gt, Assoc::Left)
            | Op::infix(Rule::ge, Assoc::Left))
        .op(Op::infix(Rule::add, Assoc::Left) | Op::infix(Rule::sub, Assoc::Left))
        .op(Op::infix(Rule::mul, Assoc::Left)
            | Op::infix(Rule::div, Assoc::Left)
            | Op::infix(Rule::modulo, Assoc::Left)
            | Op::infix(Rule::elmul, Assoc::Left)
            | Op::infix(Rule::eldiv, Assoc::Left))
        .op(Op::prefix(Rule::neg) | Op::prefix(Rule::not))
        .op(Op::infix(Rule::pow, Assoc::Right))
        .op(Op::postfix(Rule::index) | Op::postfix(Rule::tuple_idx))
}

fn binop(r: Rule) -> BinOp {
    match r {
        Rule::or => BinOp::Or,
        Rule::and => BinOp::And,
        Rule::eq => BinOp::Eq,
        Rule::ne => BinOp::Ne,
        Rule::lt => BinOp::Lt,
        Rule::le => BinOp::Le,
        Rule::gt => BinOp::Gt,
        Rule::ge => BinOp::Ge,
        Rule::add => BinOp::Add,
        Rule::sub => BinOp::Sub,
        Rule::mul => BinOp::Mul,
        Rule::div => BinOp::Div,
        Rule::modulo => BinOp::Mod,
        Rule::elmul => BinOp::ElMul,
        Rule::eldiv => BinOp::ElDiv,
        Rule::pow => BinOp::Pow,
        r => unreachable!("op {r:?}"),
    }
}

/// `expr` / `arith_expr`: a flat `operand (op operand)*` list. Each operand
/// is flattened to `prefix* primary postfix*` so the Pratt parser sees
/// prefix operators and binds `^` tighter than unary minus (`-x^2` is
/// `-(x^2)`, as in Stan).
fn build_expr(p: Pair<Rule>) -> Expr {
    let mut flat: Vec<Pair<Rule>> = vec![];
    for c in p.into_inner() {
        if c.as_rule() == Rule::operand {
            flat.extend(c.into_inner());
        } else {
            flat.push(c);
        }
    }
    pratt()
        .map_primary(build_primary)
        .map_prefix(|op, rhs| {
            let sp = span(&op);
            let o = if op.as_rule() == Rule::neg {
                UnOp::Neg
            } else {
                UnOp::Not
            };
            Expr {
                kind: ExprKind::Unary(o, Box::new(rhs)),
                span: sp,
            }
        })
        .map_postfix(|lhs, op| {
            let sp = lhs.span;
            match op.as_rule() {
                Rule::index => Expr {
                    kind: ExprKind::Index(Box::new(lhs), op.into_inner().map(build_expr).collect()),
                    span: sp,
                },
                _ => Expr {
                    kind: ExprKind::TupleIndex(
                        Box::new(lhs),
                        op.as_str()[1..].parse().expect("digits"),
                    ),
                    span: sp,
                },
            }
        })
        .map_infix(|lhs, op, rhs| {
            let sp = lhs.span;
            Expr {
                kind: ExprKind::Binary(binop(op.as_rule()), Box::new(lhs), Box::new(rhs)),
                span: sp,
            }
        })
        .parse(flat.into_iter())
}

fn build_primary(p: Pair<Rule>) -> Expr {
    let sp = span(&p);
    match p.as_rule() {
        Rule::int_lit => Expr {
            kind: ExprKind::Int(p.as_str().parse().expect("int")),
            span: sp,
        },
        Rule::real_lit => Expr {
            kind: ExprKind::Real(p.as_str().parse().expect("real")),
            span: sp,
        },
        Rule::ident => Expr {
            kind: ExprKind::Var(p.as_str().to_string()),
            span: sp,
        },
        Rule::call => {
            let mut it = p.into_inner();
            let name = it.next().expect("name").as_str().to_string();
            let mut args = vec![];
            let mut conditional = false;
            if let Some(a) = it.next() {
                for c in a.into_inner() {
                    match c.as_rule() {
                        Rule::cond_bar => conditional = true,
                        _ => args.push(build_expr(c)),
                    }
                }
            }
            Expr {
                kind: ExprKind::Call {
                    name,
                    args,
                    conditional,
                },
                span: sp,
            }
        }
        Rule::tuple_lit => Expr {
            kind: ExprKind::Tuple(p.into_inner().map(build_expr).collect()),
            span: sp,
        },
        Rule::expr => build_expr(p),
        r => unreachable!("primary {r:?}"),
    }
}
