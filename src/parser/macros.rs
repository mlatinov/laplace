//! `@macro` definitions and `@expand` invocations.
//!
//! A statement macro is a template that expands at **one spot inside a
//! block** rather than across several, plus one capability templates do
//! not have: repetition over a list.
//!
//! ```stan
//! pub @macro priors(each $p: ident, $dist: expr) : stmt in model {
//!   $p ~ $dist;
//! }
//! ```
//!
//! ```stan
//! model {
//!   @expand stats::priors([alpha, beta, gamma], normal(0, 1));
//!   y ~ normal(alpha + beta * x, gamma);
//! }
//! ```
//!
//! Verbose on purpose: the author says what the body expands to
//! (`: stmt`) and where it may be used (`in model`), so the compiler
//! never has to guess and the user gets told at the call site rather
//! than by stanc.
//!
//! # What the header buys
//!
//! `in <blocks>` is checked both ways. A `@expand` in a block the macro
//! does not list is an error at the call site; a macro listing a block
//! its own body could not legally appear in is an error at the
//! definition, which is where the author can still fix it. Stan will
//! not accept `y ~ normal(0, 1);` in `generated quantities`, so a macro
//! claiming both `model` and `generated quantities` for a `~` body is
//! wrong before anyone uses it.

use std::ops::Range;

use thiserror::Error;

use crate::parser::blocks::BlockKind;
use crate::parser::body::{self, BodyError, BodyOwner};
use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::placeholder::{parse_declarations, PlaceholderDecl, PlaceholderError};
use crate::parser::statements::statements;
use crate::parser::types::{matching_paren, split_top_level_args};
use crate::parser::visibility::Visibility;

/// The keyword that introduces a macro definition.
pub const MACRO_KEYWORD: &str = "@macro";
/// The keyword that invokes one.
pub const EXPAND_KEYWORD: &str = "@expand";

/// What a macro expands to.
///
/// Only statements today. `decl` and `expr` kinds are the obvious next
/// ones, and every match on this is written so adding them does not
/// reshape anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroKind {
    Stmt,
}

impl MacroKind {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "stmt" => Some(MacroKind::Stmt),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MacroKind::Stmt => "stmt",
        }
    }

    pub fn all() -> &'static [&'static str] {
        &["stmt"]
    }
}

/// A parsed, validated macro definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroDef {
    pub name: String,
    pub visibility: Visibility,
    pub kind: MacroKind,
    pub params: Vec<PlaceholderDecl>,
    /// The blocks this macro may be expanded in, in header order.
    pub targets: Vec<BlockKind>,
    /// The body, verbatim, placeholders included.
    pub body: String,
    /// Byte offset of the body's first byte in the file it came from.
    pub body_offset: usize,
    /// Byte range of the whole definition, its doc comment and the
    /// blank lines after it included: what has to be cut from the
    /// library body, since none of it is Stan.
    pub range: Range<usize>,
    /// Byte offset of the `@macro` keyword, for diagnostics.
    pub keyword_offset: usize,
    /// Placeholders declared but never used, for a warning.
    pub unused: Vec<String>,
}

impl MacroDef {
    /// The parameter marked `each`, if there is one.
    pub fn each_param(&self) -> Option<&PlaceholderDecl> {
        self.params.iter().find(|p| p.each)
    }

    /// The header as a reader would write it, for an error message.
    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                format!(
                    "{}${}: {}",
                    if p.each { "each " } else { "" },
                    p.name,
                    p.kind
                )
            })
            .collect();
        let targets: Vec<&str> = self.targets.iter().map(|b| b.keyword()).collect();
        format!(
            "`{}({}) : {} in {}`",
            self.name,
            params.join(", "),
            self.kind.as_str(),
            targets.join(", ")
        )
    }
}

/// A macro plus where it was written, for provenance once the
/// definition has been cut out of its file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocatedMacro {
    pub def: MacroDef,
    pub file: String,
    pub line: usize,
}

/// One `@expand pkg::name(args);` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandStatement {
    pub package: String,
    pub name: String,
    /// Argument texts, trimmed, in order. A list argument keeps its
    /// brackets; [`ExpandStatement::list_elements`] takes them apart.
    pub args: Vec<String>,
    /// Byte range of the whole statement, including its `;` and the
    /// rest of its line.
    pub range: Range<usize>,
    pub keyword_offset: usize,
    /// Byte range of `pkg::name`, so another pass can tell that this
    /// qualified name is not a function call.
    pub reference_range: Range<usize>,
    /// The indentation the statement sat at, which its expansion
    /// inherits.
    pub indent: usize,
}

impl ExpandStatement {
    pub fn label(&self) -> String {
        format!("@expand {}::{}", self.package, self.name)
    }

    /// Split a `[a, b, c]` argument into its elements.
    ///
    /// `None` when the argument is not a list at all, which is how a
    /// missing list is told apart from an empty one.
    pub fn list_elements(argument: &str) -> Option<Vec<String>> {
        let trimmed = argument.trim();
        let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
        Some(
            split_top_level_args(inner)
                .into_iter()
                .map(|element| element.trim().to_string())
                .filter(|element| !element.is_empty())
                .collect(),
        )
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MacroError {
    #[error(
        "`@macro` must be written `@macro name($p: kind) : {kinds} in <blocks> {{ ... }}`",
        kinds = MacroKind::all().join(" | ")
    )]
    MalformedDefinition { offset: usize },

    #[error("`@expand` must be written `@expand pkg::macro(args);`")]
    MalformedExpand { offset: usize },

    #[error("`@expand` needs a package-qualified macro name, as `pkg::{name}`")]
    UnqualifiedExpand { name: String, offset: usize },

    #[error("macro `{name}`: `{kind}` is not a macro kind")]
    UnknownKind {
        name: String,
        kind: String,
        offset: usize,
    },

    #[error("macro `{name}`: `{block}` is not a Stan program block")]
    UnknownTarget {
        name: String,
        block: String,
        offset: usize,
    },

    #[error("macro `{name}` lists no target block")]
    NoTargets { name: String, offset: usize },

    #[error("macro `{name}` lists `{block}` twice")]
    DuplicateTarget {
        name: String,
        block: String,
        offset: usize,
    },

    #[error("macro `{name}` marks {count} parameters `each`")]
    SeveralEach {
        name: String,
        count: usize,
        offset: usize,
    },

    #[error(
        "macro `{name}` cannot be expanded in `{block}`: {reason}, and `{statement}` is in its \
         body"
    )]
    BodyIllegalInTarget {
        name: String,
        block: String,
        reason: String,
        statement: String,
        offset: usize,
    },

    #[error("macro `{name}`: {source}")]
    Placeholder {
        name: String,
        offset: usize,
        #[source]
        source: PlaceholderError,
    },

    #[error(transparent)]
    Body(#[from] BodyError),
}

impl MacroError {
    pub fn offset(&self) -> usize {
        match self {
            MacroError::MalformedDefinition { offset }
            | MacroError::MalformedExpand { offset }
            | MacroError::UnqualifiedExpand { offset, .. }
            | MacroError::UnknownKind { offset, .. }
            | MacroError::UnknownTarget { offset, .. }
            | MacroError::NoTargets { offset, .. }
            | MacroError::DuplicateTarget { offset, .. }
            | MacroError::SeveralEach { offset, .. }
            | MacroError::BodyIllegalInTarget { offset, .. }
            | MacroError::Placeholder { offset, .. } => *offset,
            MacroError::Body(inner) => inner.offset(),
        }
    }

    pub fn help(&self) -> String {
        match self {
            MacroError::MalformedDefinition { .. } => {
                "the header is a name, a parameter list, `: stmt`, and `in` one or more Stan \
                 block names"
                    .to_string()
            }
            MacroError::MalformedExpand { .. } => {
                "write `@expand pkg::macro(arg, [a, b]);` inside a Stan block".to_string()
            }
            MacroError::UnqualifiedExpand { name, .. } => format!(
                "macros live in packages, so name the package: `@expand pkg::{name}(...);`"
            ),
            MacroError::UnknownKind { .. } => {
                format!("the kinds are {}", MacroKind::all().join(", "))
            }
            MacroError::UnknownTarget { .. } => {
                "name Stan's own blocks: data, transformed data, parameters, transformed \
                 parameters, model, generated quantities"
                    .to_string()
            }
            MacroError::NoTargets { .. } => {
                "say where it may be used, as `in model`".to_string()
            }
            MacroError::DuplicateTarget { block, .. } => {
                format!("list `{block}` once")
            }
            MacroError::SeveralEach { .. } => {
                "at most one parameter may be a list in this version -- expand the macro twice, \
                 or take a single list"
                    .to_string()
            }
            MacroError::BodyIllegalInTarget { block, .. } => {
                format!("drop `{block}` from the `in` list, or move that statement out of the body")
            }
            MacroError::Placeholder { source, .. } => source.help(),
            MacroError::Body(inner) => inner.help(),
        }
    }
}

/// Find every `@macro` definition in a `.laplacelib` source.
///
/// `public` is given the byte offset of the `@macro` keyword and
/// answers whether a `pub` marker sits in front of it.
pub fn find_macros(
    source: &str,
    public: &dyn Fn(usize) -> bool,
) -> Result<Vec<MacroDef>, MacroError> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut macros = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = source[search_from..].find(MACRO_KEYWORD) {
        let keyword = search_from + rel;
        let after_keyword = keyword + MACRO_KEYWORD.len();
        search_from = after_keyword;
        if !mask.is_real(keyword) {
            continue;
        }
        if after_keyword < bytes.len() && is_ident_char(bytes[after_keyword]) {
            continue;
        }

        let malformed = || MacroError::MalformedDefinition { offset: keyword };

        let name_start = skip_space(source, after_keyword);
        let mut name_end = name_start;
        while name_end < bytes.len() && is_ident_char(bytes[name_end]) {
            name_end += 1;
        }
        if name_end == name_start {
            return Err(malformed());
        }
        let name = source[name_start..name_end].to_string();

        let open_paren = skip_space(source, name_end);
        if bytes.get(open_paren) != Some(&b'(') {
            return Err(malformed());
        }
        let close_paren = matching_paren(source, open_paren).ok_or_else(malformed)?;
        let params =
            parse_declarations(&source[open_paren + 1..close_paren]).map_err(|source| {
                MacroError::Placeholder {
                    name: name.clone(),
                    offset: open_paren,
                    source,
                }
            })?;

        // `: <kind>`
        let colon = skip_space(source, close_paren + 1);
        if bytes.get(colon) != Some(&b':') {
            return Err(malformed());
        }
        let kind_start = skip_space(source, colon + 1);
        let mut kind_end = kind_start;
        while kind_end < bytes.len() && is_ident_char(bytes[kind_end]) {
            kind_end += 1;
        }
        let kind_text = &source[kind_start..kind_end];
        if kind_text.is_empty() {
            return Err(malformed());
        }
        let kind = MacroKind::parse(kind_text).ok_or(MacroError::UnknownKind {
            name: name.clone(),
            kind: kind_text.to_string(),
            offset: kind_start,
        })?;

        // `in <blocks>`, up to the body's brace.
        let in_start = skip_space(source, kind_end);
        if !source[in_start..].starts_with("in")
            || source[in_start + 2..]
                .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(malformed());
        }
        let targets_start = in_start + 2;
        let open_brace = source[targets_start..]
            .find('{')
            .map(|rel| targets_start + rel)
            .ok_or_else(malformed)?;
        let targets = parse_targets(&name, &source[targets_start..open_brace], targets_start)?;

        let close_brace = mask
            .match_closing_brace(source, open_brace)
            .ok_or_else(malformed)?;

        let mut end = close_brace + 1;
        if bytes.get(end) == Some(&b'\n') {
            end += 1;
        }
        while end < bytes.len() {
            let line_end = source[end..]
                .find('\n')
                .map_or(source.len(), |i| end + i + 1);
            if line_end > end && source[end..line_end].trim().is_empty() {
                end = line_end;
            } else {
                break;
            }
        }

        let definition = MacroDef {
            name,
            visibility: if public(keyword) {
                Visibility::Public
            } else {
                Visibility::Private
            },
            kind,
            params,
            targets,
            body: source[open_brace + 1..close_brace].to_string(),
            body_offset: open_brace + 1,
            range: attached_comment_start(source, keyword)..end,
            keyword_offset: keyword,
            unused: Vec::new(),
        };
        macros.push(validate(definition)?);
        search_from = close_brace + 1;
    }

    Ok(macros)
}

fn parse_targets(
    name: &str,
    text: &str,
    base: usize,
) -> Result<Vec<BlockKind>, MacroError> {
    let mut targets = Vec::new();
    for chunk in split_top_level_args(text) {
        let block = chunk.trim();
        if block.is_empty() {
            continue;
        }
        let kind = BlockKind::from_keyword(block).ok_or(MacroError::UnknownTarget {
            name: name.to_string(),
            block: block.to_string(),
            offset: base,
        })?;
        if !crate::parser::template::is_template_block(kind) {
            return Err(MacroError::UnknownTarget {
                name: name.to_string(),
                block: block.to_string(),
                offset: base,
            });
        }
        if targets.contains(&kind) {
            return Err(MacroError::DuplicateTarget {
                name: name.to_string(),
                block: block.to_string(),
                offset: base,
            });
        }
        targets.push(kind);
    }
    if targets.is_empty() {
        return Err(MacroError::NoTargets {
            name: name.to_string(),
            offset: base,
        });
    }
    Ok(targets)
}

/// Everything that can be checked where the macro is written.
fn validate(mut definition: MacroDef) -> Result<MacroDef, MacroError> {
    let each: Vec<&PlaceholderDecl> = definition.params.iter().filter(|p| p.each).collect();
    if each.len() > 1 {
        return Err(MacroError::SeveralEach {
            name: definition.name.clone(),
            count: each.len(),
            offset: definition.keyword_offset,
        });
    }

    let used = body::check(
        BodyOwner::Macro,
        &definition.name,
        &definition.params,
        &definition.body,
        definition.body_offset,
    )?;

    // Every statement has to be legal in *every* block the header
    // claims. A target the body cannot be used in is the author's
    // mistake, and this is where they can still fix it.
    for statement in statements(&definition.body) {
        for block in &definition.targets {
            if !statement.is_legal_in(*block) {
                return Err(MacroError::BodyIllegalInTarget {
                    name: definition.name.clone(),
                    block: block.keyword().to_string(),
                    reason: statement.rejection(*block),
                    statement: first_line(&statement.text),
                    offset: definition.body_offset + statement.offset,
                });
            }
        }
    }

    definition.unused = definition
        .params
        .iter()
        .filter(|p| !used.contains(&p.name))
        .map(|p| p.name.clone())
        .collect();
    Ok(definition)
}

/// Find every `@expand` statement in a `.laplace` source.
pub fn find_expand_statements(source: &str) -> Result<Vec<ExpandStatement>, MacroError> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut statements = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = source[search_from..].find(EXPAND_KEYWORD) {
        let keyword = search_from + rel;
        let after_keyword = keyword + EXPAND_KEYWORD.len();
        search_from = after_keyword;
        if !mask.is_real(keyword) {
            continue;
        }
        if after_keyword < bytes.len() && is_ident_char(bytes[after_keyword]) {
            continue;
        }

        let malformed = || MacroError::MalformedExpand { offset: keyword };

        let ref_start = skip_space(source, after_keyword);
        let mut at = ref_start;
        while at < bytes.len() && (is_ident_char(bytes[at]) || bytes[at] == b':') {
            at += 1;
        }
        let reference = &source[ref_start..at];
        if reference.is_empty() {
            return Err(malformed());
        }
        let Some((package, name)) = reference.split_once("::") else {
            return Err(MacroError::UnqualifiedExpand {
                name: reference.to_string(),
                offset: keyword,
            });
        };
        if package.is_empty() || name.is_empty() || name.contains(':') {
            return Err(malformed());
        }

        let open_paren = skip_space(source, at);
        if bytes.get(open_paren) != Some(&b'(') {
            return Err(malformed());
        }
        let close_paren = matching_paren(source, open_paren).ok_or_else(malformed)?;
        let args: Vec<String> = split_top_level_args(&source[open_paren + 1..close_paren])
            .into_iter()
            .map(|arg| arg.trim().to_string())
            .filter(|arg| !arg.is_empty())
            .collect();

        // The expansion replaces the statement in place, so it takes the
        // whole line -- indentation included, which it inherits.
        let line_start = source[..keyword].rfind('\n').map_or(0, |i| i + 1);
        let indent = keyword - line_start;
        let mut end = skip_space(source, close_paren + 1);
        if bytes.get(end) == Some(&b';') {
            end += 1;
        }
        while end < bytes.len() && matches!(bytes[end], b' ' | b'\t' | b'\r') {
            end += 1;
        }
        if bytes.get(end) == Some(&b'\n') {
            end += 1;
        }

        statements.push(ExpandStatement {
            package: package.to_string(),
            name: name.to_string(),
            args,
            range: line_start..end,
            keyword_offset: keyword,
            reference_range: ref_start..at,
            indent,
        });
        search_from = end;
    }

    Ok(statements)
}

/// Where the item starting at `at` really begins: the first line of the
/// `//` comment block directly above it, if there is one.
fn attached_comment_start(source: &str, at: usize) -> usize {
    let mut start = source[..at].rfind('\n').map_or(0, |i| i + 1);
    loop {
        let Some(previous_end) = source[..start].strip_suffix('\n') else {
            return start;
        };
        let previous_start = previous_end.rfind('\n').map_or(0, |i| i + 1);
        if !source[previous_start..start].trim_start().starts_with("//") {
            return start;
        }
        start = previous_start;
    }
}

fn skip_space(text: &str, from: usize) -> usize {
    let bytes = text.as_bytes();
    let mut at = from;
    while at < bytes.len() && bytes[at].is_ascii_whitespace() {
        at += 1;
    }
    at
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIORS: &str = r#"@macro priors(each $p: ident, $dist: expr) : stmt in model {
  $p ~ $dist;
}
"#;

    fn parse(source: &str) -> Result<Vec<MacroDef>, MacroError> {
        find_macros(source, &|_| false)
    }

    fn parse_one(source: &str) -> MacroDef {
        let mut found = parse(source).expect("should parse");
        assert_eq!(found.len(), 1);
        found.remove(0)
    }

    fn error(source: &str) -> MacroError {
        parse(source).expect_err("should not parse")
    }

    #[test]
    fn the_worked_example_parses() {
        let definition = parse_one(PRIORS);
        assert_eq!(definition.name, "priors");
        assert_eq!(definition.kind, MacroKind::Stmt);
        assert_eq!(definition.targets, vec![BlockKind::Model]);
        assert_eq!(definition.body.trim(), "$p ~ $dist;");
        assert_eq!(definition.each_param().map(|p| p.name.as_str()), Some("p"));
        assert!(definition.unused.is_empty());
        assert_eq!(
            definition.signature(),
            "`priors(each $p: ident, $dist: expr) : stmt in model`"
        );
    }

    #[test]
    fn several_target_blocks_are_parsed_in_order() {
        let definition = parse_one(
            "@macro m($x: ident) : stmt in transformed parameters, generated quantities {\n  real ${x}_z = 0;\n}\n",
        );
        assert_eq!(
            definition.targets,
            vec![BlockKind::TransformedParameters, BlockKind::GeneratedQuantities]
        );
    }

    #[test]
    fn a_macro_without_each_parses() {
        let definition = parse_one("@macro m($x: ident) : stmt in model {\n  $x ~ std_normal();\n}\n");
        assert!(definition.each_param().is_none());
    }

    #[test]
    fn visibility_comes_from_the_caller() {
        assert_eq!(
            find_macros(PRIORS, &|_| true).unwrap()[0].visibility,
            Visibility::Public
        );
        assert_eq!(parse_one(PRIORS).visibility, Visibility::Private);
    }

    #[test]
    fn the_range_takes_the_doc_comment_and_trailing_blank_lines() {
        let source = format!("// @laplace\n// @brief Priors.\n{PRIORS}\nreal f() {{\n  return 1;\n}}\n");
        let definition = parse_one(&source);
        assert_eq!(definition.range.start, 0);
        assert!(source[definition.range.clone()].starts_with("// @laplace"));
        assert!(source[definition.range.clone()].ends_with("}\n\n"));
    }

    #[test]
    fn a_macro_in_a_comment_is_not_a_macro() {
        assert!(parse("// @macro m($x: ident) : stmt in model { }\n").unwrap().is_empty());
    }

    #[test]
    fn an_unused_placeholder_is_recorded_for_a_warning() {
        let definition =
            parse_one("@macro m($x: ident, $spare: expr) : stmt in model {\n  $x ~ std_normal();\n}\n");
        assert_eq!(definition.unused, vec!["spare"]);
    }

    // ---- definition-time errors --------------------------------------

    #[test]
    fn two_each_parameters_are_rejected() {
        let err = error(
            "@macro m(each $a: ident, each $b: ident) : stmt in model {\n  $a ~ std_normal();\n}\n",
        );
        assert!(matches!(err, MacroError::SeveralEach { .. }), "{err:?}");
        assert!(err.help().contains("at most one"), "{}", err.help());
    }

    #[test]
    fn a_sampling_body_targeting_generated_quantities_is_rejected() {
        let err = error("@macro m($p: ident) : stmt in generated quantities {\n  $p ~ std_normal();\n}\n");
        assert!(matches!(err, MacroError::BodyIllegalInTarget { .. }), "{err:?}");
        let rendered = err.to_string();
        assert!(rendered.contains("`~` statement is not legal in `generated quantities`"), "{rendered}");
        assert!(err.help().contains("drop `generated quantities`"), "{}", err.help());
    }

    #[test]
    fn a_sampling_body_targeting_model_and_generated_quantities_is_rejected() {
        let err = error(
            "@macro m($p: ident) : stmt in model, generated quantities {\n  $p ~ std_normal();\n}\n",
        );
        assert!(matches!(err, MacroError::BodyIllegalInTarget { .. }), "{err:?}");
    }

    #[test]
    fn an_rng_body_targeting_the_model_block_is_rejected() {
        let err = error(
            "@macro m($x: ident) : stmt in model {\n  real ${x}_sim = normal_rng(0, 1);\n}\n",
        );
        assert!(matches!(err, MacroError::BodyIllegalInTarget { .. }), "{err:?}");
        assert!(err.to_string().contains("`_rng` function"), "{err}");
    }

    #[test]
    fn a_declaration_body_may_target_parameters() {
        assert!(parse("@macro m($x: ident) : stmt in parameters {\n  real $x;\n}\n").is_ok());
    }

    #[test]
    fn a_statement_body_may_not_target_parameters() {
        let err = error("@macro m($x: ident) : stmt in parameters {\n  $x = 1;\n}\n");
        assert!(matches!(err, MacroError::BodyIllegalInTarget { .. }), "{err:?}");
    }

    #[test]
    fn an_unknown_kind_lists_the_ones_that_exist() {
        let err = error("@macro m($x: ident) : decl in model {\n  real $x;\n}\n");
        assert!(matches!(err, MacroError::UnknownKind { .. }), "{err:?}");
        assert!(err.help().contains("stmt"), "{}", err.help());
    }

    #[test]
    fn an_unknown_target_block_is_rejected() {
        let err = error("@macro m($x: ident) : stmt in priors {\n  $x ~ std_normal();\n}\n");
        assert!(matches!(err, MacroError::UnknownTarget { .. }), "{err:?}");
        assert!(err.help().contains("generated quantities"), "{}", err.help());
    }

    #[test]
    fn the_functions_block_is_not_a_target() {
        let err = error("@macro m($x: ident) : stmt in functions {\n  real $x;\n}\n");
        assert!(matches!(err, MacroError::UnknownTarget { .. }), "{err:?}");
    }

    #[test]
    fn a_duplicated_target_is_rejected() {
        let err = error("@macro m($x: ident) : stmt in model, model {\n  $x ~ std_normal();\n}\n");
        assert!(matches!(err, MacroError::DuplicateTarget { .. }), "{err:?}");
    }

    #[test]
    fn a_header_with_no_targets_is_rejected() {
        let err = error("@macro m($x: ident) : stmt in {\n  $x ~ std_normal();\n}\n");
        assert!(matches!(err, MacroError::NoTargets { .. }), "{err:?}");
        assert!(err.help().contains("in model"), "{}", err.help());
    }

    #[test]
    fn a_malformed_header_is_rejected() {
        for source in [
            "@macro {\n}\n",
            "@macro m {\n}\n",
            "@macro m($x: ident) {\n}\n",
            "@macro m($x: ident) : stmt {\n}\n",
            "@macro m($x: ident) : stmt in model\n",
        ] {
            assert!(
                matches!(error(source), MacroError::MalformedDefinition { .. }),
                "{source:?} should be malformed"
            );
        }
    }

    #[test]
    fn body_hygiene_is_enforced_the_same_way_as_a_templates() {
        let err = error("@macro m($p: ident) : stmt in model {\n  real tmp = 1;\n  $p ~ normal(tmp, 1);\n}\n");
        assert!(matches!(err, MacroError::Body(_)), "{err:?}");
        assert!(err.to_string().contains("macro `m`"), "{err}");
    }

    #[test]
    fn reaching_for_a_model_variable_is_rejected() {
        let err = error("@macro m($p: ident) : stmt in model {\n  $p ~ normal(mu, 1);\n}\n");
        assert!(matches!(err, MacroError::Body(_)), "{err:?}");
        assert!(err.to_string().contains("`mu`"), "{err}");
    }

    #[test]
    fn a_nested_expand_is_rejected() {
        let err = error("@macro m($p: ident) : stmt in model {\n  @expand other::n($p);\n}\n");
        assert!(matches!(err, MacroError::Body(_)), "{err:?}");
    }

    // ---- `@expand` ---------------------------------------------------

    #[test]
    fn the_worked_invocation_parses() {
        let source = "model {\n  @expand stats::priors([alpha, beta, gamma], normal(0, 1));\n  y ~ normal(0, 1);\n}\n";
        let found = find_expand_statements(source).unwrap();
        assert_eq!(found.len(), 1);
        let expand = &found[0];
        assert_eq!(expand.package, "stats");
        assert_eq!(expand.name, "priors");
        assert_eq!(expand.args, vec!["[alpha, beta, gamma]", "normal(0, 1)"]);
        assert_eq!(expand.label(), "@expand stats::priors");
        assert_eq!(expand.indent, 2);
        assert_eq!(&source[expand.reference_range.clone()], "stats::priors");
    }

    #[test]
    fn the_statement_range_covers_its_whole_line() {
        let source = "model {\n  @expand stats::priors([a], normal(0, 1));\n  y ~ normal(0, 1);\n}\n";
        let found = find_expand_statements(source).unwrap();
        assert_eq!(
            &source[found[0].range.clone()],
            "  @expand stats::priors([a], normal(0, 1));\n"
        );
    }

    #[test]
    fn a_list_argument_splits_into_its_elements() {
        assert_eq!(
            ExpandStatement::list_elements("[alpha, beta, gamma]"),
            Some(vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string()
            ])
        );
    }

    #[test]
    fn an_element_with_its_own_commas_is_not_split() {
        assert_eq!(
            ExpandStatement::list_elements("[f(1, 2), g(3)]"),
            Some(vec!["f(1, 2)".to_string(), "g(3)".to_string()])
        );
    }

    #[test]
    fn an_empty_list_parses_as_a_list_with_no_elements() {
        // Told apart from "not a list at all", which is a different
        // mistake with a different message.
        assert_eq!(ExpandStatement::list_elements("[]"), Some(vec![]));
        assert_eq!(ExpandStatement::list_elements("alpha"), None);
    }

    #[test]
    fn several_expand_statements_are_found_in_order() {
        let found = find_expand_statements(
            "model {\n  @expand s::a([x], 1);\n  @expand s::b([y], 2);\n}\n",
        )
        .unwrap();
        assert_eq!(
            found.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn an_expand_in_a_comment_is_ignored() {
        assert!(find_expand_statements("// @expand s::m([x], 1);\n").unwrap().is_empty());
    }

    #[test]
    fn an_unqualified_expand_says_to_name_the_package() {
        let err = find_expand_statements("model {\n  @expand priors([a], 1);\n}\n").unwrap_err();
        assert!(matches!(err, MacroError::UnqualifiedExpand { .. }), "{err:?}");
        assert!(err.help().contains("pkg::priors"), "{}", err.help());
    }

    #[test]
    fn a_malformed_expand_is_rejected() {
        assert!(find_expand_statements("@expand s::m;\n").is_err());
        assert!(find_expand_statements("@expand ;\n").is_err());
        assert!(find_expand_statements("@expand s::m([a];\n").is_err());
    }

    #[test]
    fn parsing_is_deterministic() {
        assert_eq!(parse(PRIORS).unwrap(), parse(PRIORS).unwrap());
        let source = "model {\n  @expand s::m([a, b], 1);\n}\n";
        assert_eq!(
            find_expand_statements(source).unwrap(),
            find_expand_statements(source).unwrap()
        );
    }
}
