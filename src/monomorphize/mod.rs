//! Compile-time monomorphization of higher-order functions.
//!
//! Stan cannot pass a function to a function. laplace adds it the way
//! C++ templates and Rust generics do: it generates one specialized copy
//! of the higher-order function per distinct function actually passed
//! in, and rewrites the call sites to use the copy. Nothing generic
//! survives into the `.stan` output, and a higher-order function that is
//! never called is not emitted at all -- it has no valid Stan form.
//!
//! ```stan
//! real apply_twice(real x, func(real) -> real f) {   // laplace source
//!   real a = f(x);
//!   return f(a);
//! }
//! real r = apply_twice(5, add_one);
//! ```
//!
//! ```stan
//! real apply_twice__add_one(real x) {                // Stan output
//!   real a = add_one(x);
//!   return add_one(a);
//! }
//! real r = apply_twice__add_one(5);
//! ```
//!
//! # Where the copies go
//!
//! All of them at the *end* of the compiled `functions { }` block, with
//! a forward declaration at the top for any that another function calls.
//!
//! Ordering is why. A copy calls the function bound into it, Stan wants
//! a function declared before it is used, and a library's higher-order
//! function bound to one of the *user's* functions would otherwise be
//! emitted in the library's section -- before the function it calls, and
//! in `--split-functions` mode in a different file entirely. Emitting
//! every copy last makes "a copy follows everything it calls" true by
//! construction, whatever is bound into it, and a forward declaration
//! covers the one remaining case: a caller that precedes it.
//!
//! # Name space
//!
//! The pass runs *before* mangling, on source-name text, because that is
//! the text whose line numbers the origin map can explain. The copies it
//! generates are written straight into output-name space -- they are
//! emitted outside any package's text, so nothing will mangle them
//! later, and this module therefore applies a package's own mangling to
//! a copy body itself.

pub mod calls;
pub mod wait;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use thiserror::Error;

use crate::codegen::rename::{find_qualified_calls, mangle, rename_identifier_calls};
use crate::parser::brace_match::CodeMask;
use crate::parser::functional::{find_wait_uses, FunctionalError, FunctionalParam};
use crate::parser::origin::{line_col, PackageOrigin};
use crate::parser::signatures::{extract_signatures, FunctionSig};
use crate::parser::types::{parse_type, StanType, TypeCategory};

use calls::{find_calls, uses_outside_call_position, CallSite, Reference};
use wait::{ReturnShape, WaitError};

/// Stan built-ins common enough that binding one deserves a dedicated
/// message rather than "no such function".
///
/// Diagnostics only. Actually binding a built-in needs a real signature
/// table, which is deliberately left for later; until then this list
/// exists so the most likely mistake explains itself.
const COMMON_BUILTINS: &[&str] = &[
    "Phi", "Phi_approx", "abs", "acos", "asin", "atan", "cbrt", "ceil", "cos", "cosh",
    "cumulative_sum", "digamma", "erf", "erfc", "exp", "exp2", "expm1", "fabs", "floor", "inv",
    "inv_cloglog", "inv_logit", "inv_sqrt", "inv_square", "lgamma", "log", "log10", "log1m",
    "log1m_exp", "log1p", "log1p_exp", "log2", "log_softmax", "logit", "max", "mean", "min",
    "prod", "round", "sd", "sin", "sinh", "softmax", "sqrt", "square", "step", "sum", "tan",
    "tanh", "tgamma", "trigamma", "trunc", "variance",
];

/// Where a unit's code comes from.
#[derive(Debug, Clone)]
pub enum UnitOrigin {
    Package {
        name: String,
        version: String,
        files: PackageOrigin,
    },
    Project {
        file: String,
    },
}

impl UnitOrigin {
    /// The package this unit is, if it is one.
    fn package(&self) -> Option<&str> {
        match self {
            UnitOrigin::Package { name, .. } => Some(name),
            UnitOrigin::Project { .. } => None,
        }
    }

    /// How a function of this unit is named in compiled output.
    fn output_name(&self, func: &str) -> String {
        match self {
            UnitOrigin::Package { name, .. } => mangle(name, func),
            UnitOrigin::Project { .. } => func.to_string(),
        }
    }
}

/// One body of code that may define and call functions.
#[derive(Debug, Clone)]
pub struct Unit {
    pub origin: UnitOrigin,
    /// The unit's text, in source-name space.
    pub text: String,
    /// Where function definitions live in `text`: the whole text for a
    /// package, the `functions { }` block's contents for the project. A
    /// call inside this range is inside a function body, which is what
    /// decides whether a specialized function needs forward declaring.
    pub definitions: Range<usize>,
    /// Item names this unit exposes to others as `pkg::name`.
    pub public_names: Vec<String>,
    /// Package names this unit may refer to with `pkg::`.
    pub visible_packages: Vec<String>,
    /// Byte ranges an earlier pass owns, such as a `@use` line. A
    /// higher-order call here cannot be specialized, because the text
    /// around it is about to be replaced.
    pub reserved: Vec<Range<usize>>,
}

impl Unit {
    /// The project's entry file. Function definitions live in its
    /// `functions { }` block; a file without one defines none.
    pub fn project(
        file: impl Into<String>,
        text: impl Into<String>,
        visible_packages: Vec<String>,
    ) -> Self {
        let text = text.into();
        let definitions = crate::parser::blocks::find_top_level_blocks(&text)
            .into_iter()
            .find(|block| block.kind == crate::parser::blocks::BlockKind::Functions)
            .map(|block| block.body_range)
            .unwrap_or(0..0);
        Unit {
            origin: UnitOrigin::Project { file: file.into() },
            text,
            definitions,
            public_names: Vec::new(),
            visible_packages,
            reserved: Vec::new(),
        }
    }

    /// A resolved package. Its whole source is definitions.
    pub fn package(
        name: impl Into<String>,
        version: impl Into<String>,
        text: impl Into<String>,
        files: PackageOrigin,
        public_names: Vec<String>,
        visible_packages: Vec<String>,
    ) -> Self {
        let text = text.into();
        let definitions = 0..text.len();
        Unit {
            origin: UnitOrigin::Package {
                name: name.into(),
                version: version.into(),
                files,
            },
            text,
            definitions,
            public_names,
            visible_packages,
            reserved: Vec::new(),
        }
    }

    fn definition_text(&self) -> &str {
        &self.text[self.definitions.clone()]
    }

    fn is_public(&self, func: &str) -> bool {
        self.public_names.iter().any(|n| n == func)
    }

    fn label(&self) -> String {
        match &self.origin {
            UnitOrigin::Package { name, .. } => name.clone(),
            UnitOrigin::Project { file } => file.clone(),
        }
    }
}

/// A replacement to apply to a unit's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEdit {
    pub range: Range<usize>,
    pub replacement: String,
}

/// Everything monomorphization decided.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    /// Edits per unit, by unit index: sorted and non-overlapping.
    pub edits: Vec<Vec<TextEdit>>,
    /// Forward declarations, for the top of the `functions { }` block.
    pub declarations: Vec<String>,
    /// Specialized definitions, for the end of the `functions { }` block.
    pub definitions: Vec<String>,
}

impl Plan {
    /// Whether anything has to be emitted into the functions block.
    pub fn emits_anything(&self) -> bool {
        !self.declarations.is_empty() || !self.definitions.is_empty()
    }

    /// Whether an edit for `unit` covers `range`, so a caller that wants
    /// to edit the same text knows to stand aside.
    pub fn covers(&self, unit: usize, range: &Range<usize>) -> bool {
        self.edits.get(unit).is_some_and(|edits| {
            edits
                .iter()
                .any(|edit| edit.range.start <= range.start && range.end <= edit.range.end)
        })
    }

    /// Edits for one unit, for a caller that applies them itself.
    pub fn edits_for(&self, unit: usize) -> &[TextEdit] {
        self.edits.get(unit).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Apply this plan's edits for `unit` to `text`.
    pub fn apply(&self, unit: usize, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for edit in self.edits_for(unit) {
            if edit.range.start < cursor {
                continue;
            }
            out.push_str(&text[cursor..edit.range.start]);
            out.push_str(&edit.replacement);
            cursor = edit.range.end;
        }
        out.push_str(&text[cursor..]);
        out
    }

    /// The forward declarations as one text block, ready to splice at
    /// the top of the functions block.
    pub fn declaration_block(&self) -> String {
        if self.declarations.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "// laplace: specialized higher-order functions, defined at the end of this block\n",
        );
        for declaration in &self.declarations {
            out.push_str(declaration);
            out.push('\n');
        }
        out
    }

    /// The specialized definitions as one text block, for the end of the
    /// functions block.
    pub fn definition_block(&self) -> String {
        self.definitions.join("\n")
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MonomorphizeError {
    #[error("{error}\n  --> {location}\n  help: {help}")]
    FunctionalShape {
        location: String,
        help: &'static str,
        #[source]
        error: FunctionalError,
    },

    #[error(
        "`{function}` uses its functional parameter `{param}` somewhere other than a call\n  \
         --> {location}\n  help: a functional parameter may only be called, as `{param}(...)` \
         -- it cannot be stored, returned, or passed on to another function"
    )]
    FunctionalParamNotCalled {
        function: String,
        param: String,
        location: String,
    },

    #[error(
        "`{function}` calls `{param}` with {found} argument(s), but `{param}` is declared as \
         `{shape}`\n  --> {location}\n  help: call it with {expected} argument(s), or change \
         the declared shape"
    )]
    FunctionalArity {
        function: String,
        param: String,
        shape: String,
        expected: usize,
        found: usize,
        location: String,
    },

    #[error(
        "`{function}` takes a function argument and calls itself\n  --> {location}\n  help: \
         recursive higher-order functions are not supported in this version -- a specialized \
         copy would have to call a name that does not exist yet"
    )]
    RecursiveHigherOrder { function: String, location: String },

    #[error(
        "`{function}` calls the higher-order function `{callee}`\n  --> {location}\n  help: \
         calling one higher-order function from inside another is not supported in this \
         version -- move the call into an ordinary function"
    )]
    NestedHigherOrderCall {
        function: String,
        callee: String,
        location: String,
    },

    #[error("{error}\n  --> {location}\n  help: {help}")]
    Wait {
        location: String,
        help: String,
        #[source]
        error: Box<WaitError>,
    },

    #[error(
        "`@wait` is malformed in `{function}`\n  --> {location}\n  help: write `@wait(f)` or \
         `@wait(f).size`, naming one of the function's `func(...) -> ...` parameters"
    )]
    WaitSyntax { function: String, location: String },

    #[error(
        "`{hof}` is called inside a `@use` argument\n  --> {location}\n  help: a template \
         argument is substituted before higher-order functions are specialized -- assign \
         `{hof}(...)` to a variable in an earlier block and pass that variable instead"
    )]
    CallInsideUseArgument { hof: String, location: String },

    #[error("`{hof}` is called with {found} argument(s) but takes {expected}\n  --> {location}")]
    CallArity {
        hof: String,
        expected: usize,
        found: usize,
        location: String,
    },

    #[error(
        "`{hof}`'s parameter `{param}` needs a function name, but `{argument}` is not one\n  \
         --> {location}\n  help: pass a bare function name, as `{hof}(..., my_func)` or \
         `{hof}(..., pkg::my_func)` -- there are no function literals"
    )]
    ArgumentNotAName {
        hof: String,
        param: String,
        argument: String,
        location: String,
    },

    #[error(
        "`{name}` is bound to `{hof}`'s parameter `{param}`, but no such function is in \
         scope\n  --> {location}\n  help: define it, import the package it comes from, or -- \
         if it is a Stan built-in -- wrap it in a function of your own and bind that"
    )]
    BoundFunctionUnknown {
        hof: String,
        param: String,
        name: String,
        location: String,
    },

    #[error(
        "`{name}` is a Stan built-in, and built-ins cannot be bound in this version\n  --> \
         {location}\n  help: wrap it in a function of your own and bind that, e.g. \
         `real {name}_(real x) {{ return {name}(x); }}`"
    )]
    BoundFunctionIsBuiltin {
        hof: String,
        param: String,
        name: String,
        location: String,
    },

    #[error(
        "`{name}` is itself a higher-order function, so it cannot be bound to `{hof}`'s \
         parameter `{param}`\n  --> {location}\n  help: a higher-order function has no Stan \
         form of its own, so there is nothing to pass -- bind an ordinary function"
    )]
    BoundFunctionIsHigherOrder {
        hof: String,
        param: String,
        name: String,
        location: String,
    },

    #[error(
        "`{package}::{func}` is private to package `{package}`\n  --> {location}\n  help: only \
         items marked `pub` can be used outside their package"
    )]
    BoundFunctionPrivate {
        package: String,
        func: String,
        location: String,
    },

    #[error(
        "`{package}::{func}` cannot be used here: `{unit}` does not import `{package}`\n  --> \
         {location}\n  help: a package's imports are private to it -- declare `{package}` as a \
         dependency of `{unit}` to use it"
    )]
    BoundPackageNotVisible {
        unit: String,
        package: String,
        func: String,
        location: String,
    },

    #[error(transparent)]
    BoundSignatureMismatch(Box<SignatureMismatch>),
}

/// A bound function whose signature is not the shape the functional
/// parameter declares.
///
/// Its own struct, boxed into the enum, because six strings of
/// diagnostics would otherwise make every `Result` in the compiler as
/// wide as the worst error it can carry.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error(
    "`{name}` does not match the shape `{hof}`'s parameter `{param}` declares, `{shape}`\n  \
     --> {location}\n  help: {available}"
)]
pub struct SignatureMismatch {
    pub hof: String,
    pub param: String,
    pub name: String,
    pub shape: String,
    pub available: String,
    pub location: String,
}

/// Run the pass over every unit.
///
/// An empty [`Plan`] means nothing in this build uses a functional
/// parameter or a sized return type, and the compiled output is exactly
/// what it would have been without this feature.
pub fn run(units: &[Unit]) -> Result<Plan, MonomorphizeError> {
    let table = SymbolTable::build(units)?;
    let mut plan = Plan {
        edits: vec![Vec::new(); units.len()],
        ..Default::default()
    };

    // Strip laplace's size annotations off every return type. This pass
    // owns sized signature types; the copies it generates build their
    // signatures from the bare type directly.
    for index in 0..units.len() {
        for sig in &table.signatures[index] {
            if let Some(span) = &sig.return_size_span {
                if table.inside_removed_hof(index, span.start) {
                    continue;
                }
                plan.edits[index].push(TextEdit {
                    range: span.clone(),
                    replacement: String::new(),
                });
            }
        }
    }

    // Every higher-order function definition goes: it has no Stan form.
    for hof in &table.hofs {
        plan.edits[hof.unit].push(TextEdit {
            range: hof.removal.clone(),
            replacement: String::new(),
        });
    }

    let mut instances: BTreeMap<InstanceKey, Instance> = BTreeMap::new();
    for (index, unit) in units.iter().enumerate() {
        let mask = CodeMask::new(&unit.text);
        for (hof_index, hof) in table.hofs.iter().enumerate() {
            let Some(reference) = table.reference_from(units, index, hof) else {
                continue;
            };
            for call in find_calls(&unit.text, &mask, &reference) {
                // Calls inside a higher-order function's own body were
                // already rejected while checking definitions.
                if table.inside_removed_hof(index, call.name_range.start) {
                    continue;
                }
                if unit
                    .reserved
                    .iter()
                    .any(|range| range.contains(&call.name_range.start))
                {
                    return Err(MonomorphizeError::CallInsideUseArgument {
                        hof: hof.source_name.clone(),
                        location: location(unit, call.name_range.start),
                    });
                }
                let bound = table.bind_call(units, index, hof, &call)?;
                let key = InstanceKey {
                    hof: hof.output_name.clone(),
                    bound: bound.iter().map(|s| s.output_name.clone()).collect(),
                };
                let name = key.specialized_name();
                let inside_body = unit.definitions.contains(&call.name_range.start);

                let entry = instances.entry(key).or_insert_with(|| Instance {
                    name: name.clone(),
                    hof: hof_index,
                    bound,
                    needs_declaration: false,
                    location: location(unit, call.name_range.start),
                });
                entry.needs_declaration |= inside_body;

                plan.edits[index].push(TextEdit {
                    range: call.full_range.clone(),
                    replacement: rewritten_call(&unit.text, &call, &name, hof),
                });
            }
        }
    }

    for instance in instances.values() {
        let hof = &table.hofs[instance.hof];
        let unit = &units[hof.unit];
        plan.definitions
            .push(table.specialize(unit, hof, instance)?);
        if instance.needs_declaration {
            plan.declarations
                .push(format!("{};", signature_of(hof, &instance.name)));
        }
    }

    for edits in &mut plan.edits {
        edits.sort_by_key(|edit| (edit.range.start, edit.range.end));
    }
    Ok(plan)
}

/// A call to a higher-order function, with its functional arguments
/// dropped and its name replaced by the specialized one.
fn rewritten_call(text: &str, call: &CallSite, name: &str, hof: &Hof) -> String {
    let kept: Vec<&str> = call
        .arg_ranges
        .iter()
        .enumerate()
        .filter(|(index, _)| !hof.functional_indices.contains(index))
        .map(|(_, range)| text[range.clone()].trim())
        .collect();
    format!("{name}({})", kept.join(", "))
}

/// The specialized copy's signature: the higher-order function's return
/// type and value parameters, under the specialized name.
fn signature_of(hof: &Hof, name: &str) -> String {
    let params: Vec<String> = hof
        .value_params
        .iter()
        .map(|(ty, param)| format!("{ty} {param}"))
        .collect();
    format!("{} {name}({})", hof.return_type, params.join(", "))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct InstanceKey {
    hof: String,
    bound: Vec<String>,
}

impl InstanceKey {
    /// `<hof>__<bound>`, with one suffix per functional parameter in
    /// parameter order.
    fn specialized_name(&self) -> String {
        let mut name = self.hof.clone();
        for bound in &self.bound {
            name.push_str("__");
            name.push_str(bound);
        }
        name
    }
}

#[derive(Debug, Clone)]
struct Instance {
    name: String,
    hof: usize,
    bound: Vec<Symbol>,
    needs_declaration: bool,
    /// A call site that asked for this instance. Binding-dependent
    /// problems -- a bound function with no return size, say -- are the
    /// caller's to fix, so they are reported here rather than at the
    /// higher-order function's own `@wait`.
    location: String,
}

/// A function that can be bound to a functional parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Symbol {
    output_name: String,
    /// Name as written in its own unit, for messages and lookup.
    source_name: String,
    unit: usize,
    params: Vec<StanType>,
    param_names: Vec<String>,
    return_type: StanType,
    return_sizes: Vec<String>,
    is_higher_order: bool,
}

impl Symbol {
    /// The signature written as a `func` shape, for a mismatch message.
    fn shape(&self) -> String {
        let args: Vec<String> = self.params.iter().map(|t| t.bare.clone()).collect();
        format!("func({}) -> {}", args.join(", "), self.return_type.bare)
    }

    /// Whether this function has exactly the shape a functional
    /// parameter declares. Sizes are laplace's own annotation and never
    /// part of a Stan signature, so they are not compared.
    fn matches(&self, param: &FunctionalParam) -> bool {
        self.params.len() == param.arg_types.len()
            && self
                .params
                .iter()
                .zip(&param.arg_types)
                .all(|(a, b)| a.matches(b))
            && self.return_type.matches(&param.return_type)
    }

    fn return_shape(&self) -> ReturnShape {
        ReturnShape {
            bound: self.output_name.clone(),
            category: self.return_type.category,
            bare: self.return_type.bare.clone(),
            sizes: self.return_sizes.clone(),
            param_names: self.param_names.clone(),
        }
    }
}

/// A higher-order function: one taking at least one functional
/// parameter.
#[derive(Debug, Clone)]
struct Hof {
    source_name: String,
    output_name: String,
    unit: usize,
    /// The whole definition plus the newline that ended it, so deleting
    /// it leaves no blank gap behind.
    removal: Range<usize>,
    body: Range<usize>,
    return_type: String,
    /// Value parameters as `(type, name)`, in order.
    value_params: Vec<(String, String)>,
    functional_params: Vec<FunctionalParam>,
    functional_indices: BTreeSet<usize>,
    total_params: usize,
    header_offset: usize,
    /// How far the original definition was indented. A copy is emitted
    /// at the top level of the functions block, so the body it inherits
    /// from a definition inside a `functions { }` wrapper is shifted
    /// back by this much.
    indent: usize,
}

struct SymbolTable {
    signatures: Vec<Vec<FunctionSig>>,
    /// Every name each unit defines, sorted, for mangling a copy body.
    unit_names: Vec<Vec<String>>,
    symbols: Vec<Symbol>,
    hofs: Vec<Hof>,
}

impl SymbolTable {
    fn build(units: &[Unit]) -> Result<Self, MonomorphizeError> {
        let mut signatures = Vec::with_capacity(units.len());
        let mut unit_names = Vec::with_capacity(units.len());
        let mut symbols = Vec::new();
        let mut hofs = Vec::new();

        for (index, unit) in units.iter().enumerate() {
            // Offsets come back relative to the scanned slice; shift
            // them so every offset in this table indexes `unit.text`.
            let base = unit.definitions.start;
            let sigs: Vec<FunctionSig> = extract_signatures(unit.definition_text())
                .into_iter()
                .map(|sig| shift(sig, base))
                .collect();

            let mut names: Vec<String> = sigs.iter().map(|sig| sig.name.clone()).collect();
            names.sort();
            names.dedup();
            unit_names.push(names);

            for sig in &sigs {
                if sig.is_higher_order() {
                    hofs.push(Self::hof_of(unit, index, sig)?);
                } else {
                    symbols.push(Symbol {
                        output_name: unit.origin.output_name(&sig.name),
                        source_name: sig.name.clone(),
                        unit: index,
                        params: sig.params.iter().map(|(_, ty)| parse_type(ty)).collect(),
                        param_names: sig.params.iter().map(|(name, _)| name.clone()).collect(),
                        return_type: parse_type(&sig.return_type),
                        return_sizes: sig.return_sizes.clone(),
                        is_higher_order: false,
                    });
                }
            }
            signatures.push(sigs);
        }

        // A higher-order function is a symbol too: binding one has to be
        // reported as "that is higher-order", not "no such function".
        for hof in &hofs {
            symbols.push(Symbol {
                output_name: hof.output_name.clone(),
                source_name: hof.source_name.clone(),
                unit: hof.unit,
                params: Vec::new(),
                param_names: Vec::new(),
                return_type: parse_type(&hof.return_type),
                return_sizes: Vec::new(),
                is_higher_order: true,
            });
        }

        let table = SymbolTable {
            signatures,
            unit_names,
            symbols,
            hofs,
        };
        table.check_definitions(units)?;
        Ok(table)
    }

    fn hof_of(unit: &Unit, index: usize, sig: &FunctionSig) -> Result<Hof, MonomorphizeError> {
        if let Some(error) = sig.functional_errors.first() {
            return Err(MonomorphizeError::FunctionalShape {
                location: location(unit, sig.header_offset),
                help: error.help(),
                error: error.clone(),
            });
        }
        let body = sig
            .body_span
            .clone()
            .expect("a scanned signature always has a body");
        let definition = sig
            .definition_span()
            .expect("a scanned signature always has a body");
        // Swallow the newline the definition ended with, and any blank
        // lines after it, so removing the definition leaves no gap. The
        // blank line *before* it stays, and goes on separating whatever
        // ends up on either side.
        let mut end = definition.end;
        if unit.text.as_bytes().get(end) == Some(&b'\n') {
            end += 1;
        }
        while end < unit.text.len() {
            let line_end = unit.text[end..]
                .find('\n')
                .map_or(unit.text.len(), |i| end + i + 1);
            if line_end > end && unit.text[end..line_end].trim().is_empty() {
                end = line_end;
            } else {
                break;
            }
        }

        let functional_indices: BTreeSet<usize> =
            sig.functional_params.iter().map(|f| f.index).collect();
        let value_params = sig
            .params
            .iter()
            .enumerate()
            .filter(|(i, _)| !functional_indices.contains(i))
            .map(|(_, (name, ty))| (ty.clone(), name.clone()))
            .collect();

        Ok(Hof {
            source_name: sig.name.clone(),
            output_name: unit.origin.output_name(&sig.name),
            unit: index,
            removal: definition.start..end,
            body,
            return_type: sig.return_type.clone(),
            value_params,
            functional_params: sig.functional_params.clone(),
            functional_indices,
            total_params: sig.params.len(),
            header_offset: sig.header_offset,
            indent: {
                let line_start = unit.text[..sig.header_offset]
                    .rfind('\n')
                    .map_or(0, |i| i + 1);
                sig.header_offset - line_start
            },
        })
    }

    /// Everything that can be checked once, where the higher-order
    /// function is defined, rather than per instantiation.
    fn check_definitions(&self, units: &[Unit]) -> Result<(), MonomorphizeError> {
        for hof in &self.hofs {
            let unit = &units[hof.unit];
            let body = &unit.text[hof.body.clone()];
            // `@wait(f)` names the parameter without calling it, so the
            // placeholders are blanked before the "only ever called"
            // check -- byte for byte, so offsets still line up.
            let scanned = wait::blank_uses(body);
            let body = scanned.as_str();
            let mask = CodeMask::new(body);
            let at = |offset: usize| location(unit, hof.body.start + offset);

            for param in &hof.functional_params {
                // Every use being a call is what rules out storing,
                // returning, or forwarding the parameter -- no Stan
                // expression parser required.
                if let Some(offset) = uses_outside_call_position(body, &mask, &param.name) {
                    return Err(MonomorphizeError::FunctionalParamNotCalled {
                        function: hof.source_name.clone(),
                        param: param.name.clone(),
                        location: at(offset),
                    });
                }
                for call in find_calls(body, &mask, &param.name) {
                    if call.arity() != param.arg_types.len() {
                        return Err(MonomorphizeError::FunctionalArity {
                            function: hof.source_name.clone(),
                            param: param.name.clone(),
                            shape: param.shape(),
                            expected: param.arg_types.len(),
                            found: call.arity(),
                            location: at(call.name_range.start),
                        });
                    }
                }
            }

            if let Some(call) = find_calls(body, &mask, &hof.source_name).first() {
                return Err(MonomorphizeError::RecursiveHigherOrder {
                    function: hof.source_name.clone(),
                    location: at(call.name_range.start),
                });
            }

            for other in &self.hofs {
                if std::ptr::eq(other, hof) {
                    continue;
                }
                let Some(reference) = self.reference_from(units, hof.unit, other) else {
                    continue;
                };
                if let Some(call) = find_calls(body, &mask, &reference).first() {
                    return Err(MonomorphizeError::NestedHigherOrderCall {
                        function: hof.source_name.clone(),
                        callee: reference,
                        location: at(call.name_range.start),
                    });
                }
            }

            let real_body = &unit.text[hof.body.clone()];
            if find_wait_uses(real_body).is_err() {
                return Err(MonomorphizeError::WaitSyntax {
                    function: hof.source_name.clone(),
                    location: at(0),
                });
            }
            let declared: Vec<(String, TypeCategory)> = hof
                .functional_params
                .iter()
                .map(|p| (p.name.clone(), p.return_type.category))
                .collect();
            wait::check_declared(&hof.source_name, real_body, &declared).map_err(|error| {
                MonomorphizeError::Wait {
                    location: at(error.offset()),
                    help: error.help(),
                    error: Box::new(error),
                }
            })?;
        }
        Ok(())
    }

    /// Whether `offset` in unit `index` falls inside a higher-order
    /// function definition, which is about to be deleted.
    fn inside_removed_hof(&self, index: usize, offset: usize) -> bool {
        self.hofs
            .iter()
            .any(|hof| hof.unit == index && hof.removal.contains(&offset))
    }

    /// The text unit `index` would use to call `hof`, if it can at all.
    fn reference_from(&self, units: &[Unit], index: usize, hof: &Hof) -> Option<String> {
        if hof.unit == index {
            return Some(hof.source_name.clone());
        }
        let package = units[hof.unit].origin.package()?;
        if !units[index].visible_packages.iter().any(|p| p == package) {
            return None;
        }
        if !units[hof.unit].is_public(&hof.source_name) {
            return None;
        }
        Some(format!("{package}::{}", hof.source_name))
    }

    /// Build one specialized copy.
    fn specialize(
        &self,
        unit: &Unit,
        hof: &Hof,
        instance: &Instance,
    ) -> Result<String, MonomorphizeError> {
        let body = &unit.text[hof.body.clone()];

        // `@wait` first: resolving it reads the `f(...)` call that
        // initializes the declaration, which the next step rewrites.
        let shapes: Vec<(String, ReturnShape)> = hof
            .functional_params
            .iter()
            .zip(&instance.bound)
            .map(|(param, bound)| (param.name.clone(), bound.return_shape()))
            .collect();
        let mut body =
            wait::substitute(&hof.source_name, body, &shapes).map_err(|error| {
                MonomorphizeError::Wait {
                    location: instance.location.clone(),
                    help: error.help(),
                    error: Box::new(error),
                }
            })?;

        for (param, bound) in hof.functional_params.iter().zip(&instance.bound) {
            body = rename_identifier_calls(&body, &param.name, &bound.output_name);
        }

        // The copy is emitted outside its unit's text, so the unit's own
        // mangling has to be applied here instead of by codegen.
        let body = self.mangle_body(&body, unit, hof.unit);
        let body = dedent(&body, hof.indent);

        let mut out = String::new();
        out.push_str(&provenance(unit, hof, instance));
        out.push_str(&signature_of(hof, &instance.name));
        out.push(' ');
        out.push_str(&body);
        out.push('\n');
        Ok(out)
    }

    /// Apply a unit's own name mangling to a piece of its code.
    fn mangle_body(&self, body: &str, unit: &Unit, index: usize) -> String {
        // `dep::func(` -> `dep__func(`.
        let mut out = String::with_capacity(body.len());
        let mut cursor = 0usize;
        for call in find_qualified_calls(body) {
            out.push_str(&body[cursor..call.range.start]);
            out.push_str(&mangle(&call.package, &call.func));
            cursor = call.range.end;
        }
        out.push_str(&body[cursor..]);

        // The unit's own names, if it mangles them at all.
        if let UnitOrigin::Package { name, .. } = &unit.origin {
            for func in &self.unit_names[index] {
                out = rename_identifier_calls(&out, func, &mangle(name, func));
            }
        }
        out
    }

    /// Resolve every functional argument of one call.
    fn bind_call(
        &self,
        units: &[Unit],
        index: usize,
        hof: &Hof,
        call: &CallSite,
    ) -> Result<Vec<Symbol>, MonomorphizeError> {
        let unit = &units[index];
        let at = location(unit, call.name_range.start);

        if call.arity() != hof.total_params {
            return Err(MonomorphizeError::CallArity {
                hof: hof.source_name.clone(),
                expected: hof.total_params,
                found: call.arity(),
                location: at,
            });
        }

        let mut bound = Vec::with_capacity(hof.functional_params.len());
        for param in &hof.functional_params {
            let argument = call
                .arg(&unit.text, param.index)
                .expect("arity was checked above");
            let reference = Reference::parse(argument).ok_or_else(|| {
                MonomorphizeError::ArgumentNotAName {
                    hof: hof.source_name.clone(),
                    param: param.name.clone(),
                    argument: argument.to_string(),
                    location: at.clone(),
                }
            })?;
            bound.push(self.resolve_binding(units, index, hof, param, &reference, &at)?);
        }
        Ok(bound)
    }

    fn resolve_binding(
        &self,
        units: &[Unit],
        index: usize,
        hof: &Hof,
        param: &FunctionalParam,
        reference: &Reference,
        at: &str,
    ) -> Result<Symbol, MonomorphizeError> {
        let unit = &units[index];
        let candidates: Vec<&Symbol> = match reference {
            Reference::Plain(name) => self
                .symbols
                .iter()
                .filter(|s| s.unit == index && s.source_name == *name)
                .collect(),
            Reference::Qualified { package, func } => {
                let target = units
                    .iter()
                    .position(|u| u.origin.package() == Some(package.as_str()));
                let visible = unit.visible_packages.iter().any(|p| p == package);
                let Some(target) = target.filter(|_| visible) else {
                    return Err(MonomorphizeError::BoundPackageNotVisible {
                        unit: unit.label(),
                        package: package.clone(),
                        func: func.clone(),
                        location: at.to_string(),
                    });
                };
                // A bound name is not a call, so codegen's own
                // visibility check never sees it -- do it here.
                if !units[target].is_public(func) {
                    return Err(MonomorphizeError::BoundFunctionPrivate {
                        package: package.clone(),
                        func: func.clone(),
                        location: at.to_string(),
                    });
                }
                self.symbols
                    .iter()
                    .filter(|s| s.unit == target && s.source_name == *func)
                    .collect()
            }
        };

        let written = reference.as_written();
        if candidates.is_empty() {
            if COMMON_BUILTINS.contains(&written.as_str()) {
                return Err(MonomorphizeError::BoundFunctionIsBuiltin {
                    hof: hof.source_name.clone(),
                    param: param.name.clone(),
                    name: written,
                    location: at.to_string(),
                });
            }
            return Err(MonomorphizeError::BoundFunctionUnknown {
                hof: hof.source_name.clone(),
                param: param.name.clone(),
                name: written,
                location: at.to_string(),
            });
        }
        if candidates.iter().all(|s| s.is_higher_order) {
            return Err(MonomorphizeError::BoundFunctionIsHigherOrder {
                hof: hof.source_name.clone(),
                param: param.name.clone(),
                name: written,
                location: at.to_string(),
            });
        }

        match candidates.iter().find(|s| s.matches(param)) {
            Some(symbol) => Ok((*symbol).clone()),
            None => {
                let available: Vec<String> = candidates
                    .iter()
                    .filter(|s| !s.is_higher_order)
                    .map(|s| s.shape())
                    .collect();
                let available = if available.len() == 1 {
                    format!("`{written}` is `{}`", available[0])
                } else {
                    format!(
                        "`{written}` has these overloads: {}",
                        available
                            .iter()
                            .map(|s| format!("`{s}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                Err(MonomorphizeError::BoundSignatureMismatch(Box::new(
                    SignatureMismatch {
                        hof: hof.source_name.clone(),
                        param: param.name.clone(),
                        name: written,
                        shape: param.shape(),
                        available,
                        location: at.to_string(),
                    },
                )))
            }
        }
    }
}

/// Shift every line after the first back by up to `width` columns.
///
/// A definition inside a `functions { }` wrapper is indented; its copy
/// is emitted at the top level, so carrying the original indentation
/// across would leave the body and its closing brace hanging.
fn dedent(body: &str, width: usize) -> String {
    if width == 0 {
        return body.to_string();
    }
    let mut out = String::with_capacity(body.len());
    for (index, line) in body.split_inclusive('\n').enumerate() {
        if index == 0 {
            out.push_str(line);
            continue;
        }
        let strip = line
            .bytes()
            .take_while(|b| matches!(b, b' ' | b'\t'))
            .count()
            .min(width);
        out.push_str(&line[strip..]);
    }
    out
}

/// The provenance comment above a specialized copy.
fn provenance(unit: &Unit, hof: &Hof, instance: &Instance) -> String {
    let bindings: Vec<String> = hof
        .functional_params
        .iter()
        .zip(&instance.bound)
        .map(|(param, bound)| format!("{} = {}", param.name, bound.output_name))
        .collect();
    format!(
        "// monomorphized: {} with {} -- {}\n",
        hof.output_name,
        bindings.join(", "),
        provenance_location(unit, hof.header_offset),
    )
}

/// Shift a scanned signature's offsets so they index the whole unit text
/// rather than the slice it was scanned from.
fn shift(mut sig: FunctionSig, base: usize) -> FunctionSig {
    sig.header_offset += base;
    sig.item_offset += base;
    sig.body_span = sig.body_span.map(|r| r.start + base..r.end + base);
    sig.return_size_span = sig.return_size_span.map(|r| r.start + base..r.end + base);
    sig
}

/// `file:line:column` for an offset in a unit, for an error message.
fn location(unit: &Unit, offset: usize) -> String {
    let line_start = unit.text[..offset.min(unit.text.len())]
        .rfind('\n')
        .map_or(0, |i| i + 1);
    let column = unit.text[line_start..offset.min(unit.text.len())]
        .chars()
        .count()
        + 1;
    match &unit.origin {
        UnitOrigin::Package { name, files, .. } => match files.locate(&unit.text, offset) {
            Some(at) => format!("{name}/{}:{}:{}", at.file, at.line, column),
            None => name.clone(),
        },
        UnitOrigin::Project { file } => {
            let (line, column) = line_col(&unit.text, offset);
            format!("{file}:{line}:{column}")
        }
    }
}

/// `package version -- file:line` or `file:line`, matching the shape of
/// the provenance comments the rest of the compiler writes.
fn provenance_location(unit: &Unit, offset: usize) -> String {
    match &unit.origin {
        UnitOrigin::Package {
            name,
            version,
            files,
        } => match files.locate(&unit.text, offset) {
            Some(at) => format!("{name} v{version} ({name}/{}:{})", at.file, at.line),
            None => format!("{name} v{version}"),
        },
        UnitOrigin::Project { file } => {
            let (line, _) = line_col(&unit.text, offset);
            format!("{file}:{line}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(text: &str, imports: &[&str]) -> Unit {
        Unit::project(
            "model.laplace",
            text,
            imports.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn package(name: &str, text: &str, public: &[&str], deps: &[&str]) -> Unit {
        Unit::package(
            name,
            "1.0.0",
            text,
            PackageOrigin::single_file(format!("{name}.laplacelib"), text.len()),
            public.iter().map(|s| s.to_string()).collect(),
            deps.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// Apply a plan to every unit and join the result the way codegen
    /// would, so a test can read the whole outcome at once.
    fn compile(units: &[Unit]) -> String {
        let plan = run(units).expect("plan");
        let mut out = plan.declaration_block();
        for (index, unit) in units.iter().enumerate() {
            out.push_str(&plan.apply(index, &unit.text));
        }
        out.push_str(&plan.definition_block());
        out
    }

    const APPLY_TWICE: &str = concat!(
        "functions {\n",
        "  real add_one(real x) {\n",
        "    return x + 1;\n",
        "  }\n",
        "  real apply_twice(real x, func(real) -> real f) {\n",
        "    real a = f(x);\n",
        "    return f(a);\n",
        "  }\n",
        "}\n",
        "model {\n",
        "  real r = apply_twice(5, add_one);\n",
        "}\n",
    );

    // ---- the core case -----------------------------------------------

    #[test]
    fn a_user_hof_is_specialized_and_its_call_site_rewritten() {
        let units = vec![project(APPLY_TWICE, &[])];
        let plan = run(&units).unwrap();

        assert_eq!(plan.definitions.len(), 1);
        let definition = &plan.definitions[0];
        assert!(definition.contains("real apply_twice__add_one(real x)"), "{definition}");
        assert!(definition.contains("real a = add_one(x);"), "{definition}");
        assert!(definition.contains("return add_one(a);"), "{definition}");
        assert!(
            definition.starts_with("// monomorphized: apply_twice with f = add_one -- model.laplace:5"),
            "{definition}"
        );

        let rewritten = plan.apply(0, &units[0].text);
        assert!(rewritten.contains("real r = apply_twice__add_one(5);"), "{rewritten}");
        // The generic definition is gone; nothing `func` survives.
        assert!(!rewritten.contains("func("), "{rewritten}");
        assert!(!rewritten.contains("apply_twice(real x"), "{rewritten}");
    }

    #[test]
    fn a_call_from_the_model_block_needs_no_forward_declaration() {
        let plan = run(&[project(APPLY_TWICE, &[])]).unwrap();
        assert!(plan.declarations.is_empty());
    }

    #[test]
    fn a_call_from_inside_a_function_body_is_forward_declared() {
        let source = concat!(
            "functions {\n",
            "  real add_one(real x) {\n",
            "    return x + 1;\n",
            "  }\n",
            "  real apply_twice(real x, func(real) -> real f) {\n",
            "    return f(f(x));\n",
            "  }\n",
            "  real driver(real x) {\n",
            "    return apply_twice(x, add_one);\n",
            "  }\n",
            "}\n",
            "model {\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert_eq!(
            plan.declarations,
            vec!["real apply_twice__add_one(real x);".to_string()]
        );
        assert!(plan.declaration_block().contains("defined at the end of this block"));
    }

    #[test]
    fn the_same_binding_twice_produces_one_copy() {
        let source = concat!(
            "functions {\n",
            "  real add_one(real x) { return x + 1; }\n",
            "  real twice(real x, func(real) -> real f) { return f(f(x)); }\n",
            "}\n",
            "model {\n",
            "  real a = twice(1, add_one);\n",
            "  real b = twice(2, add_one);\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert_eq!(plan.definitions.len(), 1, "{:?}", plan.definitions);
    }

    #[test]
    fn two_different_bindings_produce_two_copies() {
        let source = concat!(
            "functions {\n",
            "  real add_one(real x) { return x + 1; }\n",
            "  real double_it(real x) { return 2 * x; }\n",
            "  real twice(real x, func(real) -> real f) { return f(f(x)); }\n",
            "}\n",
            "model {\n",
            "  real a = twice(1, add_one);\n",
            "  real b = twice(2, double_it);\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert_eq!(plan.definitions.len(), 2);
        let all = plan.definition_block();
        assert!(all.contains("real twice__add_one(real x)"), "{all}");
        assert!(all.contains("real twice__double_it(real x)"), "{all}");
    }

    #[test]
    fn a_hof_that_is_never_called_is_not_emitted_at_all() {
        let source = concat!(
            "functions {\n",
            "  real twice(real x, func(real) -> real f) { return f(f(x)); }\n",
            "}\n",
            "model {\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert!(plan.definitions.is_empty());
        let rewritten = plan.apply(0, source);
        assert!(!rewritten.contains("twice"), "{rewritten}");
    }

    #[test]
    fn several_functional_parameters_append_their_bindings_in_order() {
        let source = concat!(
            "functions {\n",
            "  real a_fn(real x) { return x; }\n",
            "  real b_fn(real x) { return x; }\n",
            "  real both(real x, func(real) -> real f, func(real) -> real g) {\n",
            "    return f(x) + g(x);\n",
            "  }\n",
            "}\n",
            "model {\n",
            "  real r = both(1, a_fn, b_fn);\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert!(
            plan.definitions[0].contains("real both__a_fn__b_fn(real x)"),
            "{}",
            plan.definitions[0]
        );
        assert!(plan.apply(0, source).contains("both__a_fn__b_fn(1)"));
    }

    #[test]
    fn a_value_argument_list_keeps_its_other_arguments() {
        let source = concat!(
            "functions {\n",
            "  real f_fn(real x) { return x; }\n",
            "  real h(real a, func(real) -> real f, int b) { return f(a) + b; }\n",
            "}\n",
            "model {\n",
            "  real r = h(1, f_fn, 2);\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert!(plan.definitions[0].contains("real h__f_fn(real a, int b)"), "{:?}", plan.definitions);
        assert!(plan.apply(0, source).contains("h__f_fn(1, 2)"));
    }

    // ---- across packages ---------------------------------------------

    const STATS: &str = concat!(
        "pub real mean_(vector x) {\n",
        "  return sum(x) / num_elements(x);\n",
        "}\n",
        "real helper(real x) {\n",
        "  return x;\n",
        "}\n",
        "pub real apply(real x, func(real) -> real f) {\n",
        "  return f(x);\n",
        "}\n",
    );

    /// The `pub` markers are stripped before monomorphization runs, so
    /// a unit's text never contains them.
    fn stats_unit() -> Unit {
        package(
            "stats",
            &STATS.replace("pub ", ""),
            &["mean_", "apply"],
            &[],
        )
    }

    #[test]
    fn a_library_hof_bound_to_a_user_function_is_mangled_on_both_sides() {
        let source = concat!(
            "library {\n  import stats\n}\n",
            "functions {\n",
            "  real add_one(real x) { return x + 1; }\n",
            "}\n",
            "model {\n",
            "  real r = stats::apply(1, add_one);\n",
            "}\n",
        );
        let units = vec![stats_unit(), project(source, &["stats"])];
        let plan = run(&units).unwrap();

        let definition = &plan.definitions[0];
        assert!(definition.contains("real stats__apply__add_one(real x)"), "{definition}");
        assert!(definition.contains("return add_one(x);"), "{definition}");
        assert!(plan.apply(1, source).contains("stats__apply__add_one(1)"));
        // The library's generic definition is gone from its own text.
        assert!(!plan.apply(0, &units[0].text).contains("func("));
    }

    #[test]
    fn a_library_hof_bound_to_its_own_private_function_mangles_that_too() {
        let stats = package(
            "stats",
            concat!(
                "real helper(real x) {\n  return x * 2;\n}\n",
                "real apply(real x, func(real) -> real f) {\n  return f(x);\n}\n",
                "real driver(real x) {\n  return apply(x, helper);\n}\n",
            ),
            &["driver"],
            &[],
        );
        let source = "library {\n  import stats\n}\nmodel {\n  real r = stats::driver(1);\n}\n";
        let units = vec![stats, project(source, &["stats"])];
        let plan = run(&units).unwrap();

        let definition = &plan.definitions[0];
        assert!(
            definition.contains("real stats__apply__stats__helper(real x)"),
            "{definition}"
        );
        // The private helper is called by its mangled name, because the
        // copy lives outside the package's text.
        assert!(definition.contains("return stats__helper(x);"), "{definition}");
        // Called from inside `driver`, so it needs declaring first.
        assert_eq!(plan.declarations.len(), 1);
        assert!(plan.apply(0, &units[0].text).contains("apply__stats__helper(x)"));
    }

    #[test]
    fn a_user_hof_bound_to_a_public_library_function() {
        let source = concat!(
            "library {\n  import stats\n}\n",
            "functions {\n",
            "  real apply_mean(vector v, func(vector) -> real f) { return f(v); }\n",
            "}\n",
            "model {\n",
            "  real m = apply_mean(y, stats::mean_);\n",
            "}\n",
        );
        let units = vec![stats_unit(), project(source, &["stats"])];
        let plan = run(&units).unwrap();
        let definition = &plan.definitions[0];
        assert!(
            definition.contains("real apply_mean__stats__mean_(vector v)"),
            "{definition}"
        );
        assert!(definition.contains("return stats__mean_(v);"), "{definition}");
    }

    #[test]
    fn binding_a_private_library_function_from_the_project_is_an_error() {
        let source = concat!(
            "library {\n  import stats\n}\n",
            "functions {\n",
            "  real apply_one(real x, func(real) -> real f) { return f(x); }\n",
            "}\n",
            "model {\n",
            "  real r = apply_one(1, stats::helper);\n",
            "}\n",
        );
        let err = run(&[stats_unit(), project(source, &["stats"])]).unwrap_err();
        assert!(matches!(err, MonomorphizeError::BoundFunctionPrivate { .. }), "{err:?}");
        assert!(err.to_string().contains("private to package `stats`"), "{err}");
    }

    #[test]
    fn binding_from_a_package_the_unit_does_not_import_is_an_error() {
        let source = concat!(
            "functions {\n",
            "  real apply_one(real x, func(real) -> real f) { return f(x); }\n",
            "}\n",
            "model {\n",
            "  real r = apply_one(1, stats::mean_);\n",
            "}\n",
        );
        let err = run(&[stats_unit(), project(source, &[])]).unwrap_err();
        assert!(
            matches!(err, MonomorphizeError::BoundPackageNotVisible { .. }),
            "{err:?}"
        );
    }

    // ---- definition-time checks --------------------------------------

    fn hof_error(body: &str) -> MonomorphizeError {
        let source = format!("functions {{\n{body}}}\nmodel {{\n}}\n");
        run(&[project(&source, &[])]).unwrap_err()
    }

    #[test]
    fn storing_a_functional_parameter_is_rejected() {
        let err = hof_error("  real h(func(real) -> real f) {\n    real g = f;\n    return g;\n  }\n");
        assert!(
            matches!(err, MonomorphizeError::FunctionalParamNotCalled { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("may only be called"), "{err}");
    }

    #[test]
    fn forwarding_a_functional_parameter_is_rejected() {
        let err = hof_error(
            "  real inner(real x, func(real) -> real g) { return g(x); }\n  real outer(real x, func(real) -> real f) { return inner(x, f); }\n",
        );
        assert!(
            matches!(err, MonomorphizeError::FunctionalParamNotCalled { .. })
                || matches!(err, MonomorphizeError::NestedHigherOrderCall { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn calling_a_functional_parameter_with_the_wrong_arity_is_rejected() {
        let err = hof_error("  real h(real x, func(real) -> real f) {\n    return f(x, x);\n  }\n");
        assert!(matches!(err, MonomorphizeError::FunctionalArity { .. }), "{err:?}");
        assert!(err.to_string().contains("2 argument(s)"), "{err}");
    }

    #[test]
    fn a_recursive_higher_order_function_is_rejected() {
        // Recursing without forwarding `f`, so the "only ever called"
        // check does not fire first.
        let err = hof_error(
            "  real inc(real x) { return x + 1; }\n  real h(real x, func(real) -> real f) {\n    if (x > 0) return h(x - 1, inc);\n    return f(x);\n  }\n",
        );
        assert!(
            matches!(err, MonomorphizeError::RecursiveHigherOrder { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_nested_functional_shape_is_rejected() {
        let err = hof_error("  real h(func(func(real) -> real) -> real f) {\n    return 1;\n  }\n");
        assert!(matches!(err, MonomorphizeError::FunctionalShape { .. }), "{err:?}");
        assert!(err.to_string().contains("has a `func` inside it"), "{err}");
    }

    #[test]
    fn an_array_return_shape_is_rejected() {
        let err = hof_error("  real h(func(real) -> array[] real f) {\n    return 1;\n  }\n");
        assert!(err.to_string().contains("array return types"), "{err}");
    }

    #[test]
    fn a_malformed_functional_shape_is_rejected_with_its_help() {
        let err = hof_error("  real h(real x, func(real) real f) {\n    return x;\n  }\n");
        assert!(matches!(err, MonomorphizeError::FunctionalShape { .. }), "{err:?}");
        assert!(err.to_string().contains("help:"), "{err}");
    }

    #[test]
    fn calling_one_higher_order_function_from_another_is_rejected() {
        let err = hof_error(
            "  real inner(real x, func(real) -> real g) { return g(x); }\n  real add(real x) { return x + 1; }\n  real outer(real x, func(real) -> real f) { return inner(x, add) + f(x); }\n",
        );
        assert!(
            matches!(err, MonomorphizeError::NestedHigherOrderCall { .. }),
            "{err:?}"
        );
    }

    // ---- call-site checks --------------------------------------------

    fn call_error(model: &str) -> MonomorphizeError {
        let source = format!(
            concat!(
                "functions {{\n",
                "  real add_one(real x) {{ return x + 1; }}\n",
                "  vector vec_one(vector x) {{ return x; }}\n",
                "  real twice(real x, func(real) -> real f) {{ return f(f(x)); }}\n",
                "}}\n",
                "model {{\n{}}}\n",
            ),
            model
        );
        run(&[project(&source, &[])]).unwrap_err()
    }

    #[test]
    fn a_non_name_in_a_functional_position_is_rejected() {
        let err = call_error("  real r = twice(1, x + 1);\n");
        assert!(matches!(err, MonomorphizeError::ArgumentNotAName { .. }), "{err:?}");
        assert!(err.to_string().contains("there are no function literals"), "{err}");
    }

    #[test]
    fn an_unknown_bound_name_is_rejected() {
        let err = call_error("  real r = twice(1, nope);\n");
        assert!(
            matches!(err, MonomorphizeError::BoundFunctionUnknown { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn binding_a_stan_builtin_suggests_a_wrapper() {
        let err = call_error("  real r = twice(1, exp);\n");
        assert!(
            matches!(err, MonomorphizeError::BoundFunctionIsBuiltin { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("real exp_(real x)"), "{err}");
    }

    #[test]
    fn a_signature_mismatch_names_the_expected_shape_and_what_was_found() {
        let err = call_error("  real r = twice(1, vec_one);\n");
        assert!(
            matches!(err, MonomorphizeError::BoundSignatureMismatch(_)),
            "{err:?}"
        );
        let rendered = err.to_string();
        assert!(rendered.contains("func(real) -> real"), "{rendered}");
        assert!(rendered.contains("`vec_one` is `func(vector) -> vector`"), "{rendered}");
    }

    #[test]
    fn the_wrong_number_of_arguments_at_a_call_site_is_rejected() {
        let err = call_error("  real r = twice(1);\n");
        assert!(matches!(err, MonomorphizeError::CallArity { .. }), "{err:?}");
    }

    #[test]
    fn binding_a_higher_order_function_is_rejected() {
        let source = concat!(
            "functions {\n",
            "  real inner(real x, func(real) -> real g) { return g(x); }\n",
            "  real twice(real x, func(real) -> real f) { return f(f(x)); }\n",
            "}\n",
            "model {\n",
            "  real r = twice(1, inner);\n",
            "}\n",
        );
        let err = run(&[project(source, &[])]).unwrap_err();
        assert!(
            matches!(err, MonomorphizeError::BoundFunctionIsHigherOrder { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn overload_selection_picks_the_matching_signature() {
        let source = concat!(
            "functions {\n",
            "  real scale(real x) { return 2 * x; }\n",
            "  vector scale(vector x) { return 2 * x; }\n",
            "  real twice(real x, func(real) -> real f) { return f(f(x)); }\n",
            "}\n",
            "model {\n",
            "  real r = twice(1, scale);\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        assert_eq!(plan.definitions.len(), 1);
        assert!(plan.definitions[0].contains("real twice__scale(real x)"));
    }

    #[test]
    fn no_matching_overload_lists_every_one() {
        let source = concat!(
            "functions {\n",
            "  vector scale(vector x) { return x; }\n",
            "  matrix scale(matrix x) { return x; }\n",
            "  real twice(real x, func(real) -> real f) { return f(f(x)); }\n",
            "}\n",
            "model {\n",
            "  real r = twice(1, scale);\n",
            "}\n",
        );
        let err = run(&[project(source, &[])]).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("overloads"), "{rendered}");
        assert!(rendered.contains("func(vector) -> vector"), "{rendered}");
        assert!(rendered.contains("func(matrix) -> matrix"), "{rendered}");
    }

    // ---- `@wait` and sized return types ------------------------------

    #[test]
    fn wait_expands_to_the_bound_functions_sized_return_type() {
        let source = concat!
        (
            "functions {\n",
            "  vector[2] to_pair(real x) { return [x, x * 2]'; }\n",
            "  matrix expand_rows(vector x, func(real) -> vector f) {\n",
            "    matrix[num_elements(x), @wait(f).size] out;\n",
            "    for (i in 1:num_elements(x)) {\n",
            "      @wait(f) row = f(x[i]);\n",
            "      out[i] = row';\n",
            "    }\n",
            "    return out;\n",
            "  }\n",
            "}\n",
            "model {\n",
            "  matrix[3, 2] m = expand_rows(y, to_pair);\n",
            "}\n",
        );
        let units = vec![project(source, &[])];
        let plan = run(&units).unwrap();
        let definition = &plan.definitions[0];

        assert!(definition.contains("matrix[num_elements(x), 2] out;"), "{definition}");
        assert!(definition.contains("vector[2] row = to_pair(x[i]);"), "{definition}");
        assert!(!definition.contains("@wait"), "{definition}");

        // The size annotation is stripped from `to_pair` itself.
        let rewritten = plan.apply(0, source);
        assert!(rewritten.contains("vector to_pair(real x)"), "{rewritten}");
        assert!(!rewritten.contains("vector[2] to_pair"), "{rewritten}");
    }

    #[test]
    fn a_missing_return_size_is_an_error_at_the_call_site() {
        let source = concat!(
            "functions {\n",
            "  vector to_pair(real x) { return [x, x]'; }\n",
            "  matrix rows_of(vector x, func(real) -> vector f) {\n",
            "    @wait(f) row = f(x[1]);\n",
            "    return [row']';\n",
            "  }\n",
            "}\n",
            "model {\n",
            "  matrix[1, 2] m = rows_of(y, to_pair);\n",
            "}\n",
        );
        let err = run(&[project(source, &[])]).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("needs the return size of `to_pair`"), "{rendered}");
        assert!(rendered.contains("annotate the return type"), "{rendered}");
    }

    #[test]
    fn a_parameter_dependent_return_size_works_in_the_direct_call_pattern() {
        let source = concat!(
            "functions {\n",
            "  vector[K] basis(real t, int K) { return rep_vector(t, K); }\n",
            "  vector first_basis(real t, int k, func(real, int) -> vector f) {\n",
            "    @wait(f) r = f(t, k);\n",
            "    return r;\n",
            "  }\n",
            "}\n",
            "model {\n",
            "  vector[4] b = first_basis(1.0, 4, basis);\n",
            "}\n",
        );
        let plan = run(&[project(source, &[])]).unwrap();
        let definition = &plan.definitions[0];
        assert!(definition.contains("vector[(k)] r = basis(t, k);"), "{definition}");
    }

    #[test]
    fn a_parameter_dependent_size_outside_the_direct_call_pattern_errors() {
        let source = concat!(
            "functions {\n",
            "  vector[K] basis(real t, int K) { return rep_vector(t, K); }\n",
            "  vector first_basis(real t, int k, func(real, int) -> vector f) {\n",
            "    vector[@wait(f).size] r;\n",
            "    r = f(t, k);\n",
            "    return r;\n",
            "  }\n",
            "}\n",
            "model {\n",
            "  vector[4] b = first_basis(1.0, 4, basis);\n",
            "}\n",
        );
        let err = run(&[project(source, &[])]).unwrap_err();
        assert!(
            err.to_string().contains("depends on its own parameters"),
            "{err}"
        );
    }

    #[test]
    fn a_wait_naming_an_unknown_parameter_is_a_definition_time_error() {
        let err = hof_error(
            "  real h(real x, func(real) -> real f) {\n    @wait(g) a = f(x);\n    return a;\n  }\n",
        );
        assert!(matches!(err, MonomorphizeError::Wait { .. }), "{err:?}");
        assert!(err.to_string().contains("does not name a functional parameter"), "{err}");
    }

    #[test]
    fn a_wait_accessor_that_does_not_fit_the_declared_category_is_rejected() {
        let err = hof_error(
            "  matrix h(vector x, func(real) -> vector f) {\n    matrix[@wait(f).rows, 1] m;\n    return m;\n  }\n",
        );
        assert!(matches!(err, MonomorphizeError::Wait { .. }), "{err:?}");
        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn a_copied_body_is_shifted_back_to_the_top_level() {
        let plan = run(&[project(APPLY_TWICE, &[])]).unwrap();
        assert_eq!(
            plan.definitions[0].lines().skip(1).collect::<Vec<_>>(),
            vec![
                "real apply_twice__add_one(real x) {",
                "  real a = add_one(x);",
                "  return add_one(a);",
                "}",
            ],
        );
    }

    #[test]
    fn an_already_top_level_body_is_left_alone() {
        let stats = package(
            "stats",
            "real inc(real x) {\n  return x + 1;\n}\nreal apply(real x, func(real) -> real f) {\n  return f(x);\n}\nreal go(real x) {\n  return apply(x, inc);\n}\n",
            &["go"],
            &[],
        );
        let plan = run(&[stats]).unwrap();
        assert!(
            plan.definitions[0].contains("{\n  return stats__inc(x);\n}"),
            "{}",
            plan.definitions[0]
        );
    }

    // ---- nothing to do ------------------------------------------------

    #[test]
    fn a_build_with_no_functional_parameters_plans_nothing() {
        let source = "data {\n  int N;\n}\nmodel {\n}\n";
        let plan = run(&[project(source, &[])]).unwrap();
        assert!(!plan.emits_anything());
        assert_eq!(plan.apply(0, source), source);
    }

    #[test]
    fn a_package_with_no_functional_parameters_is_untouched() {
        let units = vec![package("stats", "real mean_(vector x) {\n  return 1;\n}\n", &["mean_"], &[])];
        let plan = run(&units).unwrap();
        assert!(!plan.emits_anything());
        assert_eq!(plan.apply(0, &units[0].text), units[0].text);
    }

    #[test]
    fn the_plan_is_deterministic() {
        let units = vec![project(APPLY_TWICE, &[])];
        assert_eq!(run(&units).unwrap(), run(&units).unwrap());
        assert_eq!(compile(&units), compile(&units));
    }
}

