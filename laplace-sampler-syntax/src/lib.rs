//! `laplace-sampler-syntax`: the `.lapkernel` and `.lapsampler` languages
//! without a sampler runtime: grammars, syntax trees, the model-free
//! checks, the built-in registry and the built-in kernels' settings, kernel
//! documentation, and kernel-package lookup.
//!
//! It exists so an editor (laplace-lsp) can check these files exactly as
//! `laplace sampler check` does, without BridgeStan, nuts-rs or libclang
//! at build time (it needs a C compiler, as linking any Rust program does:
//! pest assembles one file through `cc`). laplace-sample builds on it and
//! re-exports its modules under their usual `laplace_sample::...` paths.
//!
//! The supported surface for editors is [`editor`]; see
//! `docs/sampler/lsp-integration.md`. Everything else may change.

pub mod builtin_kernels;
pub mod editor;
pub mod lapkernel;
pub mod lapsampler;
pub mod libraries;
pub mod symbols;
