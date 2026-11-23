//! The .lapkernel syntax tree. Every node that can be the subject of an
//! error carries the [`Span`] (line and column) where it starts.

use std::fmt;

/// 1-based line and column in the source file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// Doc comments from the file header (`//@laplace`, `//@brief`, `//@param`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Docs {
    pub laplace: bool,
    pub brief: Option<String>,
    pub params: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelFile {
    pub docs: Docs,
    pub name: String,
    pub span: Span,
    pub tuning: Vec<Decl>,
    pub state: Vec<Decl>,
    pub init: Option<Block>,
    pub functions: Vec<FunDef>,
    /// `None` only if the file has no `step` block (a check error).
    pub step: Option<Block>,
    /// Sections in source order, for duplicate/ordering errors.
    pub sections: Vec<(Section, Span)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Tuning,
    State,
    Init,
    Functions,
    Step,
}

impl Section {
    pub fn name(self) -> &'static str {
        match self {
            Section::Tuning => "tuning",
            Section::State => "state",
            Section::Init => "init",
            Section::Functions => "functions",
            Section::Step => "step",
        }
    }
}

/// A declared type. Sizes are `None` in function signatures (unsized, as
/// in Stan) and `Some` in declarations.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeExpr {
    Void,
    Int,
    Real,
    Vector(Option<Box<Expr>>),
    Matrix(Option<(Box<Expr>, Box<Expr>)>),
    Tuple(Vec<TypeExpr>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bounds {
    pub lower: Option<Expr>,
    pub upper: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptTag {
    /// `step_size` or `metric` (checked in 2.2).
    pub name: String,
    pub args: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decl {
    pub ty: TypeExpr,
    pub bounds: Bounds,
    pub name: String,
    pub init: Option<Expr>,
    pub adapt: Option<AdaptTag>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunDef {
    pub ret: TypeExpr,
    pub name: String,
    pub params: Vec<(TypeExpr, String)>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LValue {
    pub name: String,
    pub index: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DestructTarget {
    /// `(real a, ...)`: declares a new variable.
    Declare(TypeExpr, String, Span),
    /// `(a, ...)`: assigns an existing one.
    Existing(String, Span),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Decl(Decl),
    Assign {
        target: LValue,
        op: AssignOp,
        value: Expr,
        span: Span,
    },
    Destructure {
        targets: Vec<DestructTarget>,
        value: Expr,
        span: Span,
    },
    If {
        cond: Expr,
        then: Box<Stmt>,
        els: Option<Box<Stmt>>,
        span: Span,
    },
    For {
        var: String,
        lo: Expr,
        hi: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    Return(Option<Expr>, Span),
    Block(Block),
    Expr(Expr),
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Decl(d) => d.span,
            Stmt::Assign { span, .. }
            | Stmt::Destructure { span, .. }
            | Stmt::If { span, .. }
            | Stmt::For { span, .. }
            | Stmt::While { span, .. }
            | Stmt::Return(_, span) => *span,
            Stmt::Block(b) => b.span,
            Stmt::Expr(e) => e.span,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    ElMul,
    ElDiv,
    Mod,
    Pow,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::ElMul => ".*",
            BinOp::ElDiv => "./",
            BinOp::Mod => "%",
            BinOp::Pow => "^",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(i64),
    Real(f64),
    Var(String),
    /// `f(a, b)`, or `f(a | b, c)` with `conditional = true` (Stan's
    /// density-call syntax; the bar is a separator like a comma).
    Call {
        name: String,
        args: Vec<Expr>,
        conditional: bool,
    },
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Index(Box<Expr>, Vec<Expr>),
    /// `t.1` (1-based, as in Stan).
    TupleIndex(Box<Expr>, usize),
    /// `(a, b, ...)`, two or more elements.
    Tuple(Vec<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}
