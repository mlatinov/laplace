//! `.lapsampler` text -> [`SamplerAst`]. Spans are line:column.

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use crate::lapkernel::ast::Span;
use crate::lapkernel::LapError;

#[derive(Parser)]
#[grammar = "lapsampler/grammar.pest"]
struct SamplerParser;

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerAst {
    pub name: String,
    pub span: Span,
    pub imports: Vec<Import>,
    pub lines: Vec<Line>,
    pub kernel_block: Option<Span>,
    pub warmup: Vec<Setting>,
    pub draws: Vec<Setting>,
    /// Sections in source order (for duplicate errors).
    pub sections: Vec<(&'static str, Span)>,
}

/// `import langevin` (a version, `import langevin@0.1`, parses so that it
/// can be refused with a reason: versions come from laplace.toml/lock).
#[derive(Clone, Debug, PartialEq)]
pub struct Import {
    pub name: String,
    pub version: Option<String>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Setting {
    pub name: String,
    pub value: f64,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LineTarget {
    Block {
        params: Vec<String>,
        label: Option<String>,
    },
    Rest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub target: LineTarget,
    pub kernel: KExpr,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ArgValue {
    Number(f64),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub name: String,
    pub value: ArgValue,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KExpr {
    /// A built-in kernel: `mala(step_size = 0.1)`.
    BuiltIn {
        name: String,
        args: Vec<Arg>,
        span: Span,
    },
    /// A local `.lapkernel`: `./kernels/mala.lapkernel(eps = 0.05)`.
    Path {
        path: String,
        args: Vec<Arg>,
        span: Span,
    },
    /// A library kernel: `langevin::mala(eps = 0.05)` (Phase 4).
    Library {
        package: String,
        name: String,
        args: Vec<Arg>,
        span: Span,
    },
    Seq(Vec<KExpr>, Span),
    Mix(Vec<(f64, KExpr)>, Span),
    Repeat(f64, Box<KExpr>, Span),
    Adapt(Box<KExpr>, Vec<Arg>, Span),
    On(Vec<String>, Box<KExpr>, Span),
}

impl KExpr {
    pub fn span(&self) -> Span {
        match self {
            KExpr::BuiltIn { span, .. }
            | KExpr::Path { span, .. }
            | KExpr::Library { span, .. } => *span,
            KExpr::Seq(_, s)
            | KExpr::Mix(_, s)
            | KExpr::Repeat(_, _, s)
            | KExpr::Adapt(_, _, s)
            | KExpr::On(_, _, s) => *s,
        }
    }
}

fn span(p: &Pair<Rule>) -> Span {
    let (line, col) = p.as_span().start_pos().line_col();
    Span { line, col }
}

fn rule_name(r: &Rule) -> String {
    match r {
        Rule::EOI => "end of file".into(),
        Rule::kw_sampler => "`sampler`".into(),
        Rule::ident => "a name".into(),
        Rule::kernel_line => "a kernel line (`block(...) : kernel` or `rest : kernel`)".into(),
        Rule::line_target => "`block(...)` or `rest`".into(),
        Rule::kexpr => "a kernel".into(),
        Rule::setting => "`name = value;`".into(),
        Rule::number => "a number".into(),
        Rule::args | Rule::arg => "`name = value`".into(),
        Rule::kw_as => "`as`".into(),
        Rule::library_block => "`library {`".into(),
        Rule::kernel_block => "`kernel {`".into(),
        Rule::warmup_block => "`warmup {`".into(),
        Rule::draws_block => "`draws {`".into(),
        Rule::import => "`import <package>`".into(),
        other => format!("{other:?}"),
    }
}

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

pub fn parse(file: &str, src: &str) -> Result<SamplerAst, LapError> {
    let mut pairs = SamplerParser::parse(Rule::file, src).map_err(|e| {
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
    let sampler = pairs.next().unwrap().into_inner().next().unwrap();
    let sp = span(&sampler);
    let mut it = sampler.into_inner();
    it.next(); // `sampler`
    let mut ast = SamplerAst {
        name: it.next().unwrap().as_str().to_string(),
        span: sp,
        imports: vec![],
        lines: vec![],
        kernel_block: None,
        warmup: vec![],
        draws: vec![],
        sections: vec![],
    };
    for s in it {
        let ssp = span(&s);
        match s.as_rule() {
            Rule::library_block => {
                ast.sections.push(("library", ssp));
                for imp in s.into_inner().filter(|p| p.as_rule() == Rule::import) {
                    let isp = span(&imp);
                    let mut name = None;
                    let mut version = None;
                    for p in imp.into_inner() {
                        match p.as_rule() {
                            Rule::ident => name = Some(p.as_str().to_string()),
                            Rule::version => version = Some(p.as_str().to_string()),
                            _ => {}
                        }
                    }
                    ast.imports.push(Import {
                        name: name.unwrap(),
                        version,
                        span: isp,
                    });
                }
            }
            Rule::kernel_block => {
                ast.sections.push(("kernel", ssp));
                // A duplicate block is reported by the checker; only the
                // first one's lines count.
                if ast.kernel_block.is_none() {
                    ast.kernel_block = Some(ssp);
                    ast.lines.extend(
                        s.into_inner()
                            .filter(|p| p.as_rule() == Rule::kernel_line)
                            .map(build_line),
                    );
                }
            }
            Rule::warmup_block | Rule::draws_block => {
                let which = if s.as_rule() == Rule::warmup_block {
                    "warmup"
                } else {
                    "draws"
                };
                ast.sections.push((which, ssp));
                let settings: Vec<Setting> = s
                    .into_inner()
                    .filter(|p| p.as_rule() == Rule::setting)
                    .map(|p| {
                        let ssp = span(&p);
                        let mut i = p.into_inner();
                        Setting {
                            name: i.next().unwrap().as_str().to_string(),
                            value: i.next().unwrap().as_str().parse().unwrap_or(f64::NAN),
                            span: ssp,
                        }
                    })
                    .collect();
                if which == "warmup" {
                    ast.warmup.extend(settings)
                } else {
                    ast.draws.extend(settings)
                }
            }
            r => unreachable!("section {r:?}"),
        }
    }
    Ok(ast)
}

fn block_names(p: Pair<Rule>) -> Vec<String> {
    p.into_inner()
        .filter(|c| c.as_rule() == Rule::ident)
        .map(|c| c.as_str().to_string())
        .collect()
}

fn build_line(p: Pair<Rule>) -> Line {
    let sp = span(&p);
    let mut it = p.into_inner();
    let target_pair = it.next().unwrap();
    let mut t = target_pair.into_inner();
    let first = t.next().unwrap();
    let target = if first.as_rule() == Rule::kw_rest {
        LineTarget::Rest
    } else {
        let params = block_names(first);
        let label = t
            .find(|c| c.as_rule() == Rule::ident)
            .map(|c| c.as_str().to_string());
        LineTarget::Block { params, label }
    };
    Line {
        target,
        kernel: build_kexpr(it.next().unwrap()),
        span: sp,
    }
}

fn build_args(p: Option<Pair<Rule>>) -> Vec<Arg> {
    let Some(p) = p else { return vec![] };
    p.into_inner()
        .map(|a| {
            let asp = span(&a);
            let mut i = a.into_inner();
            let name = i.next().unwrap().as_str().to_string();
            let v = i.next().unwrap().into_inner().next().unwrap();
            let value = match v.as_rule() {
                Rule::boolean => ArgValue::Bool(v.as_str() == "true"),
                _ => ArgValue::Number(v.as_str().parse().unwrap_or(f64::NAN)),
            };
            Arg {
                name,
                value,
                span: asp,
            }
        })
        .collect()
}

fn build_kexpr(p: Pair<Rule>) -> KExpr {
    let inner = p.into_inner().next().unwrap();
    let sp = span(&inner);
    match inner.as_rule() {
        Rule::seq_call => KExpr::Seq(
            inner
                .into_inner()
                .filter(|c| c.as_rule() == Rule::kexpr)
                .map(build_kexpr)
                .collect(),
            sp,
        ),
        Rule::mix_call => KExpr::Mix(
            inner
                .into_inner()
                .filter(|c| c.as_rule() == Rule::mix_item)
                .map(|m| {
                    let mut i = m.into_inner();
                    let w = i.next().unwrap().as_str().parse().unwrap_or(f64::NAN);
                    (w, build_kexpr(i.next().unwrap()))
                })
                .collect(),
            sp,
        ),
        Rule::repeat_call => {
            let mut i = inner.into_inner().skip(1);
            let n = i.next().unwrap().as_str().parse().unwrap_or(f64::NAN);
            KExpr::Repeat(n, Box::new(build_kexpr(i.next().unwrap())), sp)
        }
        Rule::adapt_call => {
            let mut k = None;
            let mut args = vec![];
            for c in inner.into_inner() {
                match c.as_rule() {
                    Rule::kexpr => k = Some(build_kexpr(c)),
                    Rule::arg => {
                        args.extend(build_args_one(c));
                    }
                    _ => {}
                }
            }
            KExpr::Adapt(Box::new(k.unwrap()), args, sp)
        }
        Rule::on_call => {
            let mut i = inner.into_inner().skip(1);
            let names = block_names(i.next().unwrap());
            KExpr::On(names, Box::new(build_kexpr(i.next().unwrap())), sp)
        }
        Rule::namespaced => {
            let mut i = inner.into_inner();
            KExpr::Library {
                package: i.next().unwrap().as_str().to_string(),
                name: i.next().unwrap().as_str().to_string(),
                args: build_args(i.next()),
                span: sp,
            }
        }
        Rule::path_ref => {
            let mut i = inner.into_inner();
            let path = i.next().unwrap().as_str().to_string();
            KExpr::Path {
                path,
                args: build_args(i.next()),
                span: sp,
            }
        }
        Rule::named_call => {
            let mut i = inner.into_inner();
            let name = i.next().unwrap().as_str().to_string();
            KExpr::BuiltIn {
                name,
                args: build_args(i.next()),
                span: sp,
            }
        }
        r => unreachable!("kexpr {r:?}"),
    }
}

fn build_args_one(a: Pair<Rule>) -> Vec<Arg> {
    let asp = span(&a);
    let mut i = a.into_inner();
    let name = i.next().unwrap().as_str().to_string();
    let v = i.next().unwrap().into_inner().next().unwrap();
    let value = match v.as_rule() {
        Rule::boolean => ArgValue::Bool(v.as_str() == "true"),
        _ => ArgValue::Number(v.as_str().parse().unwrap_or(f64::NAN)),
    };
    vec![Arg {
        name,
        value,
        span: asp,
    }]
}
