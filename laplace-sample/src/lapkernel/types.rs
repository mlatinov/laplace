//! Static types, and the runtime values they describe.

use std::fmt;

use nalgebra::{DMatrix, DVector};

/// A size known when checking (`dim` is bound to the block size, literals,
/// and integer arithmetic on them), or only at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Known(usize),
    Unknown,
}

impl Size {
    /// Sizes agree unless both are known and differ.
    pub fn agrees(self, other: Size) -> bool {
        match (self, other) {
            (Size::Known(a), Size::Known(b)) => a == b,
            _ => true,
        }
    }
    /// The more informative of two agreeing sizes.
    pub fn meet(self, other: Size) -> Size {
        match self {
            Size::Known(_) => self,
            Size::Unknown => other,
        }
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Size::Known(n) => write!(f, "{n}"),
            Size::Unknown => f.write_str("?"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Void,
    Int,
    Real,
    Vector(Size),
    Matrix(Size, Size),
    Tuple(Vec<Ty>),
}

impl Ty {
    pub fn is_scalar(&self) -> bool {
        matches!(self, Ty::Int | Ty::Real)
    }

    /// Can a value of type `value` be stored in a variable of type `self`?
    /// Same shape, agreeing sizes; `int` promotes to `real`.
    pub fn accepts(&self, value: &Ty) -> bool {
        match (self, value) {
            (Ty::Real, Ty::Int) => true,
            (Ty::Vector(a), Ty::Vector(b)) => a.agrees(*b),
            (Ty::Matrix(a, b), Ty::Matrix(c, d)) => a.agrees(*c) && b.agrees(*d),
            (Ty::Tuple(a), Ty::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.accepts(y))
            }
            (a, b) => a == b,
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Void => f.write_str("void"),
            Ty::Int => f.write_str("int"),
            Ty::Real => f.write_str("real"),
            Ty::Vector(Size::Unknown) => f.write_str("vector"),
            Ty::Vector(n) => write!(f, "vector[{n}]"),
            Ty::Matrix(Size::Unknown, Size::Unknown) => f.write_str("matrix"),
            Ty::Matrix(r, c) => write!(f, "matrix[{r}, {c}]"),
            Ty::Tuple(ts) => {
                f.write_str("tuple(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{t}")?;
                }
                f.write_str(")")
            }
        }
    }
}

/// A run-time value. Vectors and matrices are nalgebra types (2.3).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Void,
    Int(i64),
    Real(f64),
    Vector(DVector<f64>),
    Matrix(DMatrix<f64>),
    Tuple(Vec<Value>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Void => "void",
            Value::Int(_) => "int",
            Value::Real(_) => "real",
            Value::Vector(_) => "vector",
            Value::Matrix(_) => "matrix",
            Value::Tuple(_) => "tuple",
        }
    }

    pub fn ty(&self) -> Ty {
        match self {
            Value::Void => Ty::Void,
            Value::Int(_) => Ty::Int,
            Value::Real(_) => Ty::Real,
            Value::Vector(v) => Ty::Vector(Size::Known(v.len())),
            Value::Matrix(m) => Ty::Matrix(Size::Known(m.nrows()), Size::Known(m.ncols())),
            Value::Tuple(vs) => Ty::Tuple(vs.iter().map(Value::ty).collect()),
        }
    }

    /// A scalar as f64 (`int` promotes).
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Real(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_vector(&self) -> Option<&DVector<f64>> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }
}
