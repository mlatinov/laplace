//! The compiler's passes, as one ordered list.
//!
//! laplace used to compile by nesting calls: the CLI parsed a file, loaded
//! packages, and handed both to codegen, which did resolution, renaming
//! and splicing on the way through. That works for two passes and stops
//! working at seven, because nobody can tell from the code what runs
//! before what.
//!
//! So the order lives in [`PASSES`], in one place, and [`compile`] walks
//! it. Passes that nothing implements yet are in the list as explicit
//! no-ops: a later patch fills one in without having to re-derive where
//! it belongs.
//!
//! # Order, and why
//!
//! 1. [`Pass::Parse`] -- read the entry file's structure.
//! 2. [`Pass::Resolve`] -- imports, namespaces and visibility.
//! 3. [`Pass::ExpandTemplates`] -- `@use`. Driven from pass 7, which
//!    owns the resolved package order it needs, but sequenced before
//!    the passes below within it.
//! 4. [`Pass::ExpandMacros`] -- `@expand`. Driven from pass 7, after
//!    pass 3 and before pass 6.
//! 5. [`Pass::ReResolve`] -- names introduced by 3 and 4. Expansions can
//!    call functions, and those calls resolve in the *defining* package's
//!    scope, so they cannot be resolved before the expansion exists.
//! 6. [`Pass::Monomorphize`] -- specialize higher-order functions. After
//!    expansion, so a template body may call one. Driven from pass 7,
//!    which owns the resolved package order it needs, but sequenced
//!    before mangling within it.
//! 7. [`Pass::MangleAndCodegen`] -- `pkg::item` to `pkg__item`, splice,
//!    emit.

use thiserror::Error;

use crate::codegen::{self, CodegenError, CodegenOptions, GeneratedStan, InstalledPackage};
use crate::parser::identifiers::{check_identifiers, ReservedIdentifier};
use crate::parser::library_block::{parse_library_block, LibraryBlock, LibraryBlockError};
use crate::parser::origin::line_col;

/// One pass of the compiler, in the order it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Parse,
    Resolve,
    ExpandTemplates,
    ExpandMacros,
    ReResolve,
    Monomorphize,
    MangleAndCodegen,
}

impl Pass {
    pub fn describe(self) -> &'static str {
        match self {
            Pass::Parse => "parse",
            Pass::Resolve => "resolve imports, namespaces and visibility",
            Pass::ExpandTemplates => "expand templates (@use)",
            Pass::ExpandMacros => "expand statement macros (@expand)",
            Pass::ReResolve => "re-resolve names introduced by expansion",
            Pass::Monomorphize => "monomorphize higher-order functions",
            Pass::MangleAndCodegen => "mangle names and generate Stan",
        }
    }
}

/// Every pass, in execution order. The single source of truth for what
/// runs when.
pub const PASSES: [Pass; 7] = [
    Pass::Parse,
    Pass::Resolve,
    Pass::ExpandTemplates,
    Pass::ExpandMacros,
    Pass::ReResolve,
    Pass::Monomorphize,
    Pass::MangleAndCodegen,
];

/// What to compile.
pub struct CompileRequest<'a> {
    /// The entry file's text (a `.laplace` project/model file).
    pub source: &'a str,
    /// How to name that file in diagnostics, e.g. `model.laplace`.
    pub source_name: String,
    /// Every resolved package: the direct imports and every transitive
    /// dependency. Loaded by [`crate::package::load`], which is where a
    /// package's own visibility is worked out.
    pub installed: &'a [InstalledPackage],
    pub options: CodegenOptions,
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("{error}\n  --> {name}:{line}:{column}\n  help: {help}")]
    ReservedIdentifier {
        name: String,
        line: usize,
        column: usize,
        help: String,
        #[source]
        error: ReservedIdentifier,
    },

    #[error(transparent)]
    LibraryBlock(#[from] LibraryBlockError),

    #[error(transparent)]
    Codegen(#[from] CodegenError),
}

/// Compile one entry file, running [`PASSES`] in order.
pub fn compile(request: CompileRequest<'_>) -> Result<GeneratedStan, PipelineError> {
    let CompileRequest {
        source,
        source_name,
        installed,
        options,
    } = request;
    let options = CodegenOptions {
        source_name: options.source_name.or(Some(source_name.clone())),
        ..options
    };

    let mut library_block: Option<LibraryBlock> = None;
    let mut generated: Option<GeneratedStan> = None;

    for pass in PASSES {
        match pass {
            Pass::Parse => {
                library_block = parse_library_block(source)?;
            }

            Pass::Resolve => {
                // `__` belongs to generated names, so a hand-written one
                // is rejected before any name is generated from it.
                if let Err(error) = check_identifiers(source) {
                    let (line, column) = line_col(source, error.offset);
                    return Err(PipelineError::ReservedIdentifier {
                        name: source_name.clone(),
                        line,
                        column,
                        help: error.help(),
                        error,
                    });
                }
                // Each package's own item visibility is already resolved,
                // by `package::load`. Call-site resolution -- which
                // `pkg::item` is in scope, and whether it is public --
                // still happens inside `Pass::MangleAndCodegen`, because
                // codegen is a public entry point that has to validate
                // its own inputs anyway.
                //
                // TODO(session 2): lift that into a symbol table built
                // here, which monomorphization needs in order to look up
                // a bound function's signature.
            }

            // Implemented, and driven from `Pass::MangleAndCodegen`
            // for the same reason monomorphization is: it needs the
            // resolved package list, which codegen is what computes.
            // It still runs before pass 6 within that call -- see
            // `crate::expand::blocks`.
            Pass::ExpandTemplates => {}

            // Implemented, and driven from `Pass::MangleAndCodegen`
            // like the two passes around it, for the same reason: it
            // needs the resolved package list. It runs after template
            // expansion and before monomorphization within that call --
            // see `crate::expand::macros`.
            Pass::ExpandMacros => {}

            // Nothing left to do. Calls inside a template or macro
            // body already resolve in the defining package's scope,
            // because expansion mangles the body against that package
            // before splicing it in
            // (`codegen::rename::mangle_fragment`). The pass stays in
            // the list because a future expansion kind that generates
            // *calls* rather than copying them would need it.
            Pass::ReResolve => {}

            // Implemented, but driven from inside
            // `Pass::MangleAndCodegen`: the pass needs the resolved,
            // dependency-ordered package list, and codegen is what
            // computes that order. It still runs *before* mangling,
            // which is the ordering that matters -- see
            // `crate::monomorphize` and `codegen::generate_with_options`.
            Pass::Monomorphize => {}

            Pass::MangleAndCodegen => {
                generated = Some(codegen::generate_with_options(
                    source,
                    library_block.as_ref(),
                    installed,
                    &options,
                )?);
            }
        }
    }

    Ok(generated.expect("MangleAndCodegen is in PASSES and always produces output"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(source: &'a str, installed: &'a [InstalledPackage]) -> CompileRequest<'a> {
        CompileRequest {
            source,
            source_name: "model.laplace".to_string(),
            installed,
            options: CodegenOptions::inline(),
        }
    }

    #[test]
    fn the_pass_list_is_in_the_documented_order() {
        assert_eq!(PASSES[0], Pass::Parse);
        assert_eq!(PASSES[1], Pass::Resolve);
        assert_eq!(PASSES[6], Pass::MangleAndCodegen);
        // Expansion runs before monomorphization, so a template body can
        // call a higher-order function.
        let position = |p: Pass| PASSES.iter().position(|q| *q == p).unwrap();
        assert!(position(Pass::ExpandTemplates) < position(Pass::Monomorphize));
        assert!(position(Pass::ExpandMacros) < position(Pass::ReResolve));
        assert!(position(Pass::ReResolve) < position(Pass::Monomorphize));
    }

    #[test]
    fn every_pass_describes_itself() {
        for pass in PASSES {
            assert!(!pass.describe().is_empty(), "{pass:?}");
        }
    }

    #[test]
    fn a_file_with_no_imports_compiles_through_every_pass_unchanged() {
        let source = "data {\n  int N;\n}\nmodel {\n}\n";
        let generated = compile(request(source, &[])).unwrap();
        assert_eq!(generated.source, source);
    }

    #[test]
    fn a_reserved_identifier_is_rejected_with_the_entry_file_name() {
        let source = "parameters {\n  real my__theta;\n}\n";
        let err = compile(request(source, &[])).unwrap_err();
        let rendered = err.to_string();
        assert!(
            matches!(err, PipelineError::ReservedIdentifier { .. }),
            "{err:?}"
        );
        assert!(rendered.contains("model.laplace:2:8"), "{rendered}");
        assert!(rendered.contains("my__theta"), "{rendered}");
        assert!(rendered.contains("help:"), "{rendered}");
    }

    #[test]
    fn a_codegen_error_carries_the_entry_file_name_into_its_location() {
        let stats = InstalledPackage::leaf(
            "stats",
            "1.0.0",
            "real mean_(vector x) {\n  return 1;\n}\nreal helper(real x) {\n  return x;\n}\n",
            vec!["mean_".to_string()],
        );
        let source = "library {\n  import stats\n}\n\nmodel {\n  real h = stats::helper(1);\n}\n";
        let installed = vec![stats];
        let err = compile(request(source, &installed)).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("is private to package `stats`"),
            "{rendered}"
        );
        assert!(rendered.contains("model.laplace:6:12"), "{rendered}");
    }

    #[test]
    fn an_explicit_source_name_in_the_options_wins() {
        let source = "parameters {\n  real a__b;\n}\n";
        let err = compile(CompileRequest {
            source,
            source_name: "ignored.laplace".to_string(),
            installed: &[],
            options: CodegenOptions::inline().named("chosen.laplace"),
        })
        .unwrap_err();
        // The identifier check reports the request's name; the option
        // only renames the file in codegen's own diagnostics.
        assert!(err.to_string().contains("ignored.laplace"), "{err}");
    }

    #[test]
    fn a_higher_order_function_is_specialized_through_the_pipeline() {
        let source = concat!(
            "functions {\n",
            "  real add_one(real x) {\n    return x + 1;\n  }\n",
            "  real apply_twice(real x, func(real) -> real f) {\n",
            "    real a = f(x);\n    return f(a);\n  }\n",
            "}\n",
            "model {\n",
            "  real r = apply_twice(5, add_one);\n",
            "}\n",
        );
        let generated = compile(request(source, &[])).unwrap();
        assert!(
            generated
                .source
                .contains("real apply_twice__add_one(real x)"),
            "{}",
            generated.source
        );
        assert!(generated
            .source
            .contains("real r = apply_twice__add_one(5);"));
        assert!(!generated.source.contains("func("), "{}", generated.source);
    }

    #[test]
    fn a_monomorphization_error_surfaces_with_the_entry_file_name() {
        let source = concat!(
            "functions {\n",
            "  real twice(real x, func(real) -> real f) {\n    return f(f(x));\n  }\n",
            "}\n",
            "model {\n",
            "  real r = twice(1, nope);\n",
            "}\n",
        );
        let err = compile(request(source, &[])).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("no such function is in scope"),
            "{rendered}"
        );
        assert!(rendered.contains("model.laplace:7"), "{rendered}");
    }

    #[test]
    fn every_pass_but_re_resolve_now_does_something() {
        // A reminder, not a rule: `ReResolve` is the one pass still
        // waiting for a reason to exist.
        assert_eq!(PASSES.len(), 7);
        assert_eq!(PASSES[4], Pass::ReResolve);
    }

    #[test]
    fn compiling_twice_gives_byte_identical_output() {
        let stats = InstalledPackage::leaf(
            "stats",
            "1.0.0",
            "real mean_(vector x) {\n  return sum(x);\n}\n",
            vec!["mean_".to_string()],
        );
        let source = "library {\n  import stats\n}\n\nmodel {\n  real m = stats::mean_(y);\n}\n";
        let installed = vec![stats];
        let first = compile(request(source, &installed)).unwrap();
        let second = compile(request(source, &installed)).unwrap();
        assert_eq!(first.source, second.source);
    }
}
