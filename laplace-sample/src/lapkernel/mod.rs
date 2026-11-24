//! `.lapkernel`: kernels written in a small Stan-like language (Level 3).
//!
//! The grammar, syntax tree, static checks, built-in registry and kernel
//! documentation are in the `laplace-sampler-syntax` crate, so the editor
//! can use them without this crate's runtime; they are re-exported here
//! under their usual paths. This crate adds what runs a kernel:
//! - [`natives`]: the built-ins' implementations (2.3);
//! - [`interp`]: the interpreter (2.3);
//! - [`kernel`]: the `Kernel` implementation (2.4).

pub use laplace_sampler_syntax::lapkernel::{
    ast, builtins, check, doc, file, parse, types, LapError,
};

pub mod interp;
pub mod kernel;
pub mod natives;

#[cfg(test)]
mod tests;
