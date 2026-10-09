//! Library surface for `laplace`'s core modules, so other crates (notably
//! `laplace-lsp`) can reuse the parser/resolve/docs/codegen/validate logic
//! directly instead of re-implementing any of it.
//!
//! The `laplace` binary (`src/main.rs`) is a thin shell over this crate --
//! argument parsing, filesystem paths, and terminal output only. Anything a
//! second frontend would need lives here.

pub mod codegen;
pub mod docs;
pub mod expand;
pub mod init;
pub mod manifest;
pub mod monomorphize;
pub mod package;
pub mod parser;
pub mod pipeline;
pub mod resolve;
pub mod validate;
pub mod version;
