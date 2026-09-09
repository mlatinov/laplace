//! A shallow scanner for Stan's *top-level block* keywords (`data { }`,
//! `model { }`, ...).
//!
//! This is deliberately not a Stan parser: it only locates where each
//! top-level block starts and ends so other passes can strip, unwrap, or
//! reject one. Block *contents* stay opaque text, per the project's
//! non-negotiable design constraints.

use std::ops::Range;

use crate::parser::brace_match::{is_ident_char, CodeMask};

/// The top-level blocks laplace knows by name. `Library` is laplace's own
/// addition; everything else is Stan's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Library,
    Functions,
    Data,
    TransformedData,
    Parameters,
    TransformedParameters,
    Model,
    GeneratedQuantities,
}

impl BlockKind {
    /// How the block is spelled in source, for error messages.
    pub fn keyword(self) -> &'static str {
        match self {
            BlockKind::Library => "library",
            BlockKind::Functions => "functions",
            BlockKind::Data => "data",
            BlockKind::TransformedData => "transformed data",
            BlockKind::Parameters => "parameters",
            BlockKind::TransformedParameters => "transformed parameters",
            BlockKind::Model => "model",
            BlockKind::GeneratedQuantities => "generated quantities",
        }
    }
}

/// One top-level block found in a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopLevelBlock {
    pub kind: BlockKind,
    /// Byte range of the whole block, from the first keyword byte through
    /// the closing `}` inclusive.
    pub byte_range: Range<usize>,
    /// Byte range of the block's contents, between the braces.
    pub body_range: Range<usize>,
}

/// Find every top-level (brace-depth-zero) block in `source`.
///
/// Comment and string contents are skipped, and a keyword that isn't
/// followed by `{` -- or whose braces never close -- is ignored rather than
/// reported, since this scanner's job is to locate blocks, not to validate
/// Stan.
pub fn find_top_level_blocks(source: &str) -> Vec<TopLevelBlock> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut blocks = Vec::new();
    let mut depth = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        if !mask.is_real(i) {
            i += 1;
            continue;
        }
        match bytes[i] {
            b'{' => {
                depth += 1;
                i += 1;
                continue;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
                continue;
            }
            _ => {}
        }

        if depth > 0 || !is_ident_char(bytes[i]) || (i > 0 && is_ident_char(bytes[i - 1])) {
            i += 1;
            continue;
        }

        let Some((kind, keyword_end)) = match_block_keyword(source, &mask, i) else {
            // Skip the whole identifier, so `datastore` can't re-match at
            // its `data` prefix on the next byte.
            i = end_of_identifier(bytes, i);
            continue;
        };

        let after = &source[keyword_end..];
        let trimmed = after.trim_start();
        if !trimmed.starts_with('{') {
            i = end_of_identifier(bytes, i);
            continue;
        }
        let open_brace = keyword_end + (after.len() - trimmed.len());
        let Some(close_brace) = mask.match_closing_brace(source, open_brace) else {
            i = end_of_identifier(bytes, i);
            continue;
        };

        blocks.push(TopLevelBlock {
            kind,
            byte_range: i..close_brace + 1,
            body_range: open_brace + 1..close_brace,
        });
        i = close_brace + 1;
    }

    blocks
}

fn end_of_identifier(bytes: &[u8], from: usize) -> usize {
    let mut end = from;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    end.max(from + 1)
}

/// If a block keyword starts at `at`, return it and the byte offset just
/// past it. Handles the two-word keywords (`transformed data`,
/// `transformed parameters`, `generated quantities`).
fn match_block_keyword(source: &str, mask: &CodeMask, at: usize) -> Option<(BlockKind, usize)> {
    let bytes = source.as_bytes();
    let word_end = end_of_identifier(bytes, at);
    let word = &source[at..word_end];

    let single = match word {
        "library" => Some(BlockKind::Library),
        "functions" => Some(BlockKind::Functions),
        "data" => Some(BlockKind::Data),
        "parameters" => Some(BlockKind::Parameters),
        "model" => Some(BlockKind::Model),
        _ => None,
    };
    if let Some(kind) = single {
        return Some((kind, word_end));
    }

    if word != "transformed" && word != "generated" {
        return None;
    }

    // Skip the whitespace between the two words, then read the second one.
    let mut next = word_end;
    while next < bytes.len() && bytes[next].is_ascii_whitespace() {
        next += 1;
    }
    if next >= bytes.len() || !mask.is_real(next) || !is_ident_char(bytes[next]) {
        return None;
    }
    let next_end = end_of_identifier(bytes, next);
    match (word, &source[next..next_end]) {
        ("transformed", "data") => Some((BlockKind::TransformedData, next_end)),
        ("transformed", "parameters") => Some((BlockKind::TransformedParameters, next_end)),
        ("generated", "quantities") => Some((BlockKind::GeneratedQuantities, next_end)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<BlockKind> {
        find_top_level_blocks(source)
            .into_iter()
            .map(|b| b.kind)
            .collect()
    }

    #[test]
    fn finds_every_standard_stan_block() {
        let source = concat!(
            "functions {\n  real f(real x) { return x; }\n}\n",
            "data {\n  int n;\n}\n",
            "transformed data {\n  int m = 1;\n}\n",
            "parameters {\n  real mu;\n}\n",
            "transformed parameters {\n  real nu = mu;\n}\n",
            "model {\n  mu ~ normal(0, 1);\n}\n",
            "generated quantities {\n  real y = mu;\n}\n",
        );
        assert_eq!(
            kinds(source),
            vec![
                BlockKind::Functions,
                BlockKind::Data,
                BlockKind::TransformedData,
                BlockKind::Parameters,
                BlockKind::TransformedParameters,
                BlockKind::Model,
                BlockKind::GeneratedQuantities,
            ]
        );
    }

    #[test]
    fn finds_the_laplace_library_block() {
        let source = "library {\n  import gps\n}\ndata {\n}\n";
        assert_eq!(kinds(source), vec![BlockKind::Library, BlockKind::Data]);
    }

    #[test]
    fn nested_braces_do_not_produce_nested_blocks() {
        // `data` appears as a *variable* name inside a function body; it is
        // not at depth zero, so it must not be reported as a block.
        let source = "functions {\n  real f() {\n    real data_x = 1;\n    return data_x;\n  }\n}\n";
        assert_eq!(kinds(source), vec![BlockKind::Functions]);
    }

    #[test]
    fn identifiers_that_merely_start_with_a_keyword_are_ignored() {
        let source = "datastore { int n; }\nmodelling { }\n";
        assert_eq!(kinds(source), Vec::new());
    }

    #[test]
    fn keywords_in_comments_and_strings_are_ignored() {
        let source = "// model { }\nreal f() { return 1; }\n";
        assert_eq!(kinds(source), Vec::new());
    }

    #[test]
    fn body_range_covers_exactly_the_text_between_the_braces() {
        let source = "functions {\n  real f() { return 1; }\n}\n";
        let block = &find_top_level_blocks(source)[0];
        assert_eq!(&source[block.body_range.clone()], "\n  real f() { return 1; }\n");
        assert_eq!(&source[block.byte_range.clone()], source.trim_end());
    }

    #[test]
    fn a_keyword_with_no_brace_is_not_a_block() {
        let source = "int data = 1;\n";
        assert_eq!(kinds(source), Vec::new());
    }

    #[test]
    fn bare_function_definitions_are_not_blocks() {
        let source = "real f(real x) {\n  return x;\n}\nreal g(real x) {\n  return x;\n}\n";
        assert_eq!(kinds(source), Vec::new());
    }
}
