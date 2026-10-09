//! `@template` definitions and `@use` invocations.
//!
//! A template is a pre-cut set of Stan block pieces with typed blanks.
//! The library author writes the boilerplate once:
//!
//! ```stan
//! pub @template ncp($name: ident, $N: expr) {
//!   parameters {
//!     vector[$N] ${name}_raw;
//!     real<lower=0> ${name}_sigma;
//!   }
//!   transformed parameters {
//!     vector[$N] $name = ${name}_sigma * ${name}_raw;
//!   }
//!   model {
//!     ${name}_raw ~ std_normal();
//!     ${name}_sigma ~ exponential(1);
//!   }
//! }
//! ```
//!
//! and the user drops it in with one line, as many times as they like:
//!
//! ```stan
//! @use stats::ncp(theta, K);
//! @use stats::ncp(beta, P);
//! ```
//!
//! # Hygiene is what makes reuse safe
//!
//! Two uses of one template must not collide, so **every variable a
//! template declares has to be named from an `ident` placeholder**. A
//! fixed `real tmp;` is rejected at definition time, because the second
//! use would redeclare it.
//!
//! The same rule makes the reference check cheap. Once fixed-name
//! declarations are gone, the only names a body may refer to are its
//! placeholders -- which are not identifiers at all until substitution --
//! and its own `for` loop variables. Anything else left over is a
//! reference to one of the *user's* variables, which a template may not
//! silently capture.
//!
//! # How much Stan this knows
//!
//! Only what [`crate::parser::declarations`] knows: the shape of a
//! declaration, and whether an identifier is followed by `(`. A template
//! body is laplace's own construct and is fully its business; the Stan
//! inside the pieces is still never type-checked or interpreted.

use std::ops::Range;

use thiserror::Error;

use crate::parser::blocks::{find_top_level_blocks, BlockKind};
use crate::parser::body::{self, BodyError, BodyOwner};
use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::declarations::declarations;
use crate::parser::placeholder::{parse_declarations, PlaceholderDecl, PlaceholderError};
use crate::parser::types::{matching_paren, split_top_level_args};
use crate::parser::visibility::Visibility;

/// The keyword that introduces a template definition.
pub const TEMPLATE_KEYWORD: &str = "@template";
/// The keyword that invokes one.
pub const USE_KEYWORD: &str = "@use";

/// The Stan blocks a template may contribute to.
///
/// No `functions` block: a template contributes model structure, and a
/// library that wants to contribute a function just defines one.
pub fn is_template_block(kind: BlockKind) -> bool {
    matches!(
        kind,
        BlockKind::Data
            | BlockKind::TransformedData
            | BlockKind::Parameters
            | BlockKind::TransformedParameters
            | BlockKind::Model
            | BlockKind::GeneratedQuantities
    )
}

/// One block piece of a template body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplatePiece {
    pub block: BlockKind,
    /// The piece's contents, verbatim, placeholders included.
    pub body: String,
    /// Byte offset of the piece's contents in the file it came from.
    pub offset: usize,
}

/// A parsed, validated template definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateDef {
    pub name: String,
    pub visibility: Visibility,
    pub params: Vec<PlaceholderDecl>,
    pub pieces: Vec<TemplatePiece>,
    /// Byte range of the whole `@template ... { ... }` definition, plus
    /// the newline after it: what has to be cut from the library body,
    /// since none of it is Stan.
    pub range: Range<usize>,
    /// Byte offset of the `@template` keyword, for diagnostics.
    pub keyword_offset: usize,
    /// Placeholders declared but never used, for a warning.
    pub unused: Vec<String>,
}

impl TemplateDef {
    /// The names this template declares, as written -- placeholder
    /// tokens, not yet substituted.
    pub fn declared_tokens(&self) -> Vec<String> {
        self.pieces
            .iter()
            .flat_map(|piece| declarations(&piece.body))
            .map(|d| d.name)
            .collect()
    }

    pub fn piece(&self, block: BlockKind) -> Option<&TemplatePiece> {
        self.pieces.iter().find(|piece| piece.block == block)
    }
}

/// A template plus where it was written, which is what a provenance
/// comment needs once the definition has been cut out of its file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocatedTemplate {
    pub def: TemplateDef,
    /// File name relative to the package directory.
    pub file: String,
    /// 1-indexed line of the `@template` keyword in that file.
    pub line: usize,
}

/// One `@use pkg::name(args);` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UseStatement {
    pub package: String,
    pub template: String,
    /// Argument texts, trimmed, in order.
    pub args: Vec<String>,
    /// Byte range of the whole statement including its `;` and the
    /// newline that ends its line, so removing it leaves no blank gap.
    pub range: Range<usize>,
    /// Byte offset of the `@use` keyword.
    pub keyword_offset: usize,
    /// Byte range of `pkg::name`, so another pass can tell that this
    /// qualified name is not a function call.
    pub reference_range: Range<usize>,
}

impl UseStatement {
    /// How the invocation reads, for a provenance comment.
    pub fn label(&self) -> String {
        format!(
            "@use {}::{}({})",
            self.package,
            self.template,
            self.args.join(", ")
        )
    }

    /// The short form for the closing provenance comment.
    pub fn short_label(&self) -> String {
        format!("@use {}::{}", self.package, self.template)
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TemplateError {
    #[error("`@template` must be followed by a name, a parameter list, and a body")]
    MalformedDefinition { offset: usize },

    #[error("`@use` must be written `@use pkg::template(args);`")]
    MalformedUse { offset: usize },

    #[error("`@use` needs a package-qualified template name, as `pkg::{template}`")]
    UnqualifiedUse { template: String, offset: usize },

    #[error("template `{name}`: `{block}` is not a Stan program block a template may fill")]
    BadBlock {
        name: String,
        block: String,
        offset: usize,
    },

    #[error("template `{name}` has two `{block}` pieces")]
    DuplicateBlock {
        name: String,
        block: String,
        offset: usize,
    },

    #[error(
        "template `{name}`: a template body holds only Stan block pieces, and `{text}` is not one"
    )]
    StrayText {
        name: String,
        text: String,
        offset: usize,
    },

    #[error("template `{name}`: {source}")]
    Placeholder {
        name: String,
        offset: usize,
        #[source]
        source: PlaceholderError,
    },

    /// Everything a template body shares with a macro body: hygiene,
    /// capture, nesting, undeclared placeholders. See
    /// [`crate::parser::body`].
    #[error(transparent)]
    Body(#[from] BodyError),
}

impl TemplateError {
    pub fn offset(&self) -> usize {
        match self {
            TemplateError::MalformedDefinition { offset }
            | TemplateError::MalformedUse { offset }
            | TemplateError::UnqualifiedUse { offset, .. }
            | TemplateError::BadBlock { offset, .. }
            | TemplateError::DuplicateBlock { offset, .. }
            | TemplateError::StrayText { offset, .. }
            | TemplateError::Placeholder { offset, .. } => *offset,
            TemplateError::Body(inner) => inner.offset(),
        }
    }

    pub fn help(&self) -> String {
        match self {
            TemplateError::MalformedDefinition { .. } => {
                "write `@template name($p: ident) { ... }`".to_string()
            }
            TemplateError::MalformedUse { .. } => {
                "write `@use pkg::template(arg, arg);` at the top level of the file".to_string()
            }
            TemplateError::UnqualifiedUse { template, .. } => format!(
                "templates live in packages, so name the package: `@use pkg::{template}(...);`"
            ),
            TemplateError::BadBlock { .. } => format!(
                "a template may fill {}; `functions` is not one of them",
                TEMPLATE_BLOCK_NAMES.join(", ")
            ),
            TemplateError::DuplicateBlock { block, .. } => {
                format!("merge the two `{block}` pieces into one")
            }
            TemplateError::StrayText { .. } => {
                "put every statement inside a Stan block piece, as `model { ... }`".to_string()
            }
            TemplateError::Placeholder { source, .. } => source.help(),
            TemplateError::Body(inner) => inner.help(),
        }
    }
}

/// The block names a template may use, for an error message.
const TEMPLATE_BLOCK_NAMES: &[&str] = &[
    "data",
    "transformed data",
    "parameters",
    "transformed parameters",
    "model",
    "generated quantities",
];

/// Find every `@template` definition in a `.laplacelib` source.
///
/// `public` decides each one's visibility: it is given the byte offset
/// of the `@template` keyword and answers whether a `pub` marker sits
/// in front of it.
pub fn find_templates(
    source: &str,
    public: &dyn Fn(usize) -> bool,
) -> Result<Vec<TemplateDef>, TemplateError> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut templates = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = source[search_from..].find(TEMPLATE_KEYWORD) {
        let keyword = search_from + rel;
        let after_keyword = keyword + TEMPLATE_KEYWORD.len();
        search_from = after_keyword;
        if !mask.is_real(keyword) {
            continue;
        }
        if after_keyword < bytes.len() && is_ident_char(bytes[after_keyword]) {
            continue;
        }

        let malformed = || TemplateError::MalformedDefinition { offset: keyword };

        // name
        let name_start = skip_space(source, after_keyword);
        let mut name_end = name_start;
        while name_end < bytes.len() && is_ident_char(bytes[name_end]) {
            name_end += 1;
        }
        if name_end == name_start {
            return Err(malformed());
        }
        let name = source[name_start..name_end].to_string();

        // parameter list
        let open_paren = skip_space(source, name_end);
        if bytes.get(open_paren) != Some(&b'(') {
            return Err(malformed());
        }
        let close_paren = matching_paren(source, open_paren).ok_or_else(malformed)?;
        let params =
            parse_declarations(&source[open_paren + 1..close_paren]).map_err(|source| {
                TemplateError::Placeholder {
                    name: name.clone(),
                    offset: open_paren,
                    source,
                }
            })?;

        // body
        let open_brace = skip_space(source, close_paren + 1);
        if bytes.get(open_brace) != Some(&b'{') {
            return Err(malformed());
        }
        let close_brace = mask
            .match_closing_brace(source, open_brace)
            .ok_or_else(malformed)?;

        // The definition goes whole, and so does everything that only
        // described it: the doc comment above it, and the blank lines
        // it left behind. None of it is Stan.
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
        let start = attached_comment_start(source, keyword);

        let pieces = split_pieces(&name, source, open_brace + 1..close_brace)?;
        let template = TemplateDef {
            name,
            visibility: if public(keyword) {
                Visibility::Public
            } else {
                Visibility::Private
            },
            params,
            pieces,
            range: start..end,
            keyword_offset: keyword,
            unused: Vec::new(),
        };
        templates.push(validate(template)?);
        search_from = close_brace + 1;
    }

    Ok(templates)
}

/// Split a template body into its block pieces.
fn split_pieces(
    name: &str,
    source: &str,
    body: Range<usize>,
) -> Result<Vec<TemplatePiece>, TemplateError> {
    let text = &source[body.clone()];
    let blocks = find_top_level_blocks(text);

    let mut pieces: Vec<TemplatePiece> = Vec::new();
    let mut covered = 0usize;

    for block in &blocks {
        // Nothing but whitespace may sit between pieces: a template body
        // is a list of blocks, not a Stan program.
        let between = text[covered..block.byte_range.start].trim();
        if !between.is_empty() {
            return Err(TemplateError::StrayText {
                name: name.to_string(),
                text: first_line(between),
                offset: body.start + covered,
            });
        }
        covered = block.byte_range.end;

        if !is_template_block(block.kind) {
            return Err(TemplateError::BadBlock {
                name: name.to_string(),
                block: block.kind.keyword().to_string(),
                offset: body.start + block.byte_range.start,
            });
        }
        if pieces.iter().any(|piece| piece.block == block.kind) {
            return Err(TemplateError::DuplicateBlock {
                name: name.to_string(),
                block: block.kind.keyword().to_string(),
                offset: body.start + block.byte_range.start,
            });
        }
        pieces.push(TemplatePiece {
            block: block.kind,
            body: text[block.body_range.clone()].to_string(),
            offset: body.start + block.body_range.start,
        });
    }

    let trailing = text[covered..].trim();
    if !trailing.is_empty() {
        return Err(TemplateError::StrayText {
            name: name.to_string(),
            text: first_line(trailing),
            offset: body.start + covered,
        });
    }
    Ok(pieces)
}

/// Everything that can be checked where the template is written.
fn validate(mut template: TemplateDef) -> Result<TemplateDef, TemplateError> {
    let mut used: Vec<String> = Vec::new();

    for piece in &template.pieces {
        // Hygiene, capture, nesting and undeclared placeholders are the
        // same questions a macro body has to answer, and are asked in
        // one place for both.
        for name in body::check(
            BodyOwner::Template,
            &template.name,
            &template.params,
            &piece.body,
            piece.offset,
        )? {
            if !used.contains(&name) {
                used.push(name);
            }
        }
    }

    template.unused = template
        .params
        .iter()
        .filter(|p| !used.contains(&p.name))
        .map(|p| p.name.clone())
        .collect();
    Ok(template)
}

/// Find every `@use` statement in a `.laplace` source.
pub fn find_use_statements(source: &str) -> Result<Vec<UseStatement>, TemplateError> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut statements = Vec::new();
    let mut search_from = 0usize;

    while let Some(rel) = source[search_from..].find(USE_KEYWORD) {
        let keyword = search_from + rel;
        let after_keyword = keyword + USE_KEYWORD.len();
        search_from = after_keyword;
        if !mask.is_real(keyword) {
            continue;
        }
        if after_keyword < bytes.len() && is_ident_char(bytes[after_keyword]) {
            continue;
        }

        let malformed = || TemplateError::MalformedUse { offset: keyword };

        let ref_start = skip_space(source, after_keyword);
        let mut at = ref_start;
        while at < bytes.len() && (is_ident_char(bytes[at]) || bytes[at] == b':') {
            at += 1;
        }
        let reference = &source[ref_start..at];
        if reference.is_empty() {
            return Err(malformed());
        }
        let Some((package, template)) = reference.split_once("::") else {
            return Err(TemplateError::UnqualifiedUse {
                template: reference.to_string(),
                offset: keyword,
            });
        };
        if package.is_empty() || template.is_empty() || template.contains(':') {
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

        // The statement ends at its `;`, and takes the rest of its line
        // with it so removing it leaves no stray blank line.
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

        statements.push(UseStatement {
            package: package.to_string(),
            template: template.to_string(),
            args,
            range: keyword..end,
            keyword_offset: keyword,
            reference_range: ref_start..at,
        });
        search_from = end;
    }

    Ok(statements)
}

/// Where the item starting at `at` really begins: the first line of
/// the `//` comment block directly above it, if there is one.
///
/// A comment only counts when nothing but whitespace separates it from
/// the item, matching how a function's doc comment attaches.
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

    const NCP: &str = r#"@template ncp($name: ident, $N: expr) {
  parameters {
    vector[$N] ${name}_raw;
    real<lower=0> ${name}_sigma;
  }
  transformed parameters {
    vector[$N] $name = ${name}_sigma * ${name}_raw;
  }
  model {
    ${name}_raw ~ std_normal();
    ${name}_sigma ~ exponential(1);
  }
}
"#;

    fn parse(source: &str) -> Result<Vec<TemplateDef>, TemplateError> {
        find_templates(source, &|_| false)
    }

    fn parse_one(source: &str) -> TemplateDef {
        let mut found = parse(source).expect("should parse");
        assert_eq!(found.len(), 1);
        found.remove(0)
    }

    #[test]
    fn the_worked_example_parses_into_three_pieces() {
        let template = parse_one(NCP);
        assert_eq!(template.name, "ncp");
        assert_eq!(
            template
                .params
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["name", "N"]
        );
        assert_eq!(
            template.pieces.iter().map(|p| p.block).collect::<Vec<_>>(),
            vec![
                BlockKind::Parameters,
                BlockKind::TransformedParameters,
                BlockKind::Model
            ]
        );
        assert!(template
            .piece(BlockKind::Parameters)
            .unwrap()
            .body
            .contains("${name}_raw"));
        assert!(template.unused.is_empty());
    }

    #[test]
    fn the_definitions_range_covers_the_whole_thing() {
        let template = parse_one(NCP);
        assert_eq!(template.range.start, 0);
        assert_eq!(&NCP[template.range.clone()], NCP);
    }

    #[test]
    fn the_range_takes_the_doc_comment_above_it_too() {
        // The comment describes a template, so it has no business in
        // the Stan output once the template is gone.
        let source = format!("// @laplace\n// @brief Non-centred.\n{NCP}");
        let template = parse_one(&source);
        assert_eq!(template.range.start, 0);
        assert_eq!(&source[template.range.clone()], source);
    }

    #[test]
    fn a_comment_separated_by_a_blank_line_is_not_taken() {
        let source = format!("// unrelated note\n\n{NCP}");
        let template = parse_one(&source);
        assert_eq!(&source[template.range.clone()], NCP);
    }

    #[test]
    fn the_range_swallows_the_blank_lines_the_definition_left_behind() {
        let source = format!("{NCP}\n\nreal f() {{\n  return 1;\n}}\n");
        let template = parse_one(&source);
        assert_eq!(&source[template.range.clone()], &format!("{NCP}\n\n"));
    }

    #[test]
    fn visibility_comes_from_the_caller() {
        let public = find_templates(NCP, &|_| true).unwrap();
        assert_eq!(public[0].visibility, Visibility::Public);
        assert_eq!(parse_one(NCP).visibility, Visibility::Private);
    }

    #[test]
    fn several_templates_in_one_file_are_all_found() {
        let source = format!(
            "{NCP}\n@template other($y: ident) {{\n  model {{\n    $y ~ std_normal();\n  }}\n}}\n"
        );
        let found = parse(&source).unwrap();
        assert_eq!(
            found.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["ncp", "other"]
        );
    }

    #[test]
    fn a_template_in_a_comment_is_not_a_template() {
        assert!(parse("// @template ncp($n: ident) { }\n")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_source_with_no_templates_yields_none() {
        assert!(parse("real f(real x) {\n  return x;\n}\n")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn an_unused_placeholder_is_recorded_for_a_warning() {
        let template = parse_one(
            "@template t($used: ident, $spare: expr) {\n  model {\n    $used ~ std_normal();\n  }\n}\n",
        );
        assert_eq!(template.unused, vec!["spare"]);
    }

    // ---- definition-time errors --------------------------------------

    fn error(source: &str) -> TemplateError {
        parse(source).expect_err("should not parse")
    }

    #[test]
    fn a_functions_block_in_a_template_is_rejected() {
        let err =
            error("@template t($n: ident) {\n  functions {\n    real f() { return 1; }\n  }\n}\n");
        assert!(matches!(err, TemplateError::BadBlock { .. }), "{err:?}");
        assert!(
            err.help().contains("functions` is not one"),
            "{}",
            err.help()
        );
    }

    #[test]
    fn an_unknown_block_name_is_reported_as_stray_text() {
        let err = error("@template t($n: ident) {\n  priors {\n    $n ~ std_normal();\n  }\n}\n");
        assert!(matches!(err, TemplateError::StrayText { .. }), "{err:?}");
    }

    #[test]
    fn a_duplicated_block_is_rejected() {
        let err = error(
            "@template t($n: ident) {\n  model {\n    $n ~ std_normal();\n  }\n  model {\n    $n ~ std_normal();\n  }\n}\n",
        );
        assert!(
            matches!(err, TemplateError::DuplicateBlock { .. }),
            "{err:?}"
        );
        assert!(err.help().contains("merge"), "{}", err.help());
    }

    #[test]
    fn a_statement_outside_any_block_piece_is_rejected() {
        let err = error("@template t($n: ident) {\n  $n ~ std_normal();\n}\n");
        assert!(matches!(err, TemplateError::StrayText { .. }), "{err:?}");
    }

    #[test]
    fn a_fixed_name_declaration_is_rejected_because_reuse_would_collide() {
        let err = error(
            "@template t($n: ident) {\n  transformed parameters {\n    real tmp = 1;\n    real $n = tmp;\n  }\n}\n",
        );
        assert!(matches!(err, TemplateError::Body(_)), "{err:?}");
        assert!(err.to_string().contains("`tmp`"), "{err}");
        assert!(err.help().contains("collide"), "{}", err.help());
    }

    #[test]
    fn declaring_a_variable_from_an_expr_placeholder_is_rejected() {
        let err = error("@template t($n: expr) {\n  parameters {\n    real ${n}_raw;\n  }\n}\n");
        assert!(matches!(err, TemplateError::Body(_)), "{err:?}");
        assert!(err.help().contains("ident"), "{}", err.help());
    }

    #[test]
    fn an_undeclared_placeholder_is_rejected() {
        let err = error("@template t($n: ident) {\n  model {\n    $n ~ normal($mu, 1);\n  }\n}\n");
        assert!(matches!(err, TemplateError::Body(_)), "{err:?}");
        assert!(err.to_string().contains("$mu"), "{err}");
    }

    #[test]
    fn reaching_for_a_user_variable_is_rejected() {
        let err = error("@template t($n: ident) {\n  model {\n    $n ~ normal(mu, 1);\n  }\n}\n");
        assert!(matches!(err, TemplateError::Body(_)), "{err:?}");
        assert!(err.to_string().contains("`mu`"), "{err}");
        assert!(
            err.help().contains("pass `mu` in as a placeholder"),
            "{}",
            err.help()
        );
    }

    #[test]
    fn a_loop_variable_is_a_legal_reference() {
        let template = parse_one(
            "@template t($n: ident, $N: expr) {\n  model {\n    for (i in 1:$N) {\n      ${n}_raw[i] ~ std_normal();\n    }\n  }\n}\n",
        );
        assert_eq!(template.name, "t");
    }

    #[test]
    fn calling_a_function_is_not_a_capture() {
        // `std_normal` and `exponential` are calls, not references.
        assert_eq!(parse_one(NCP).name, "ncp");
    }

    #[test]
    fn a_nested_use_is_rejected() {
        let err =
            error("@template t($n: ident) {\n  model {\n    @use other::thing($n);\n  }\n}\n");
        assert!(matches!(err, TemplateError::Body(_)), "{err:?}");
    }

    #[test]
    fn a_malformed_header_is_rejected() {
        assert!(matches!(
            error("@template {\n}\n"),
            TemplateError::MalformedDefinition { .. }
        ));
        assert!(matches!(
            error("@template t {\n}\n"),
            TemplateError::MalformedDefinition { .. }
        ));
        assert!(matches!(
            error("@template t($n: ident)\n"),
            TemplateError::MalformedDefinition { .. }
        ));
    }

    #[test]
    fn an_unknown_placeholder_kind_is_reported_with_the_template_name() {
        let err = error("@template t($n: thing) {\n  model {\n  }\n}\n");
        assert!(matches!(err, TemplateError::Placeholder { .. }), "{err:?}");
        assert!(err.to_string().contains("template `t`"), "{err}");
    }

    // ---- `@use` ------------------------------------------------------

    #[test]
    fn a_use_statement_parses_into_a_package_name_and_arguments() {
        let source = "library {\n  import stats\n}\n\n@use stats::ncp(theta, K);\n\ndata {\n}\n";
        let found = find_use_statements(source).unwrap();
        assert_eq!(found.len(), 1);
        let use_ = &found[0];
        assert_eq!(use_.package, "stats");
        assert_eq!(use_.template, "ncp");
        assert_eq!(use_.args, vec!["theta", "K"]);
        assert_eq!(use_.label(), "@use stats::ncp(theta, K)");
        assert_eq!(use_.short_label(), "@use stats::ncp");
        assert_eq!(&source[use_.reference_range.clone()], "stats::ncp");
    }

    #[test]
    fn a_use_statement_takes_its_whole_line_with_it() {
        let source = "@use stats::ncp(theta, K);\ndata {\n}\n";
        let found = find_use_statements(source).unwrap();
        assert_eq!(
            &source[found[0].range.clone()],
            "@use stats::ncp(theta, K);\n"
        );
    }

    #[test]
    fn an_argument_may_be_a_whole_expression() {
        let found =
            find_use_statements("@use stats::observation(y, mu + theta, sigma);\n").unwrap();
        assert_eq!(found[0].args, vec!["y", "mu + theta", "sigma"]);
    }

    #[test]
    fn an_argument_with_its_own_commas_is_not_split() {
        let found = find_use_statements("@use s::t(y, normal(0, 1));\n").unwrap();
        assert_eq!(found[0].args, vec!["y", "normal(0, 1)"]);
    }

    #[test]
    fn a_use_with_no_arguments_parses() {
        let found = find_use_statements("@use s::t();\n").unwrap();
        assert!(found[0].args.is_empty());
    }

    #[test]
    fn several_use_statements_are_found_in_source_order() {
        let found = find_use_statements("@use s::a(x);\n@use s::b(y);\n").unwrap();
        assert_eq!(
            found
                .iter()
                .map(|u| u.template.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn a_use_in_a_comment_is_ignored() {
        assert!(find_use_statements("// @use s::t(x);\n")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn an_unqualified_use_says_to_name_the_package() {
        let err = find_use_statements("@use ncp(theta, K);\n").unwrap_err();
        assert!(
            matches!(err, TemplateError::UnqualifiedUse { .. }),
            "{err:?}"
        );
        assert!(err.help().contains("pkg::ncp"), "{}", err.help());
    }

    #[test]
    fn a_malformed_use_is_rejected() {
        assert!(find_use_statements("@use stats::ncp;\n").is_err());
        assert!(find_use_statements("@use ;\n").is_err());
        assert!(find_use_statements("@use stats::ncp(theta;\n").is_err());
    }

    #[test]
    fn parsing_is_deterministic() {
        assert_eq!(parse(NCP).unwrap(), parse(NCP).unwrap());
        let source = "@use s::t(a, b);\n";
        assert_eq!(
            find_use_statements(source).unwrap(),
            find_use_statements(source).unwrap()
        );
    }
}
