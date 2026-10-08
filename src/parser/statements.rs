//! Which Stan statements are legal in which block.
//!
//! A statement macro declares where it may be used (`in model`, or
//! `in transformed parameters, generated quantities`). Stan has its own
//! opinion about that: a `~` statement only works in `model`, an `_rng`
//! call only in `transformed data` and `generated quantities`, and
//! `data` and `parameters` take declarations and nothing else. A macro
//! whose body cannot be used in a block it claims is a mistake worth
//! catching where the macro is written, not where someone uses it.
//!
//! # What this knows, and what it does not
//!
//! Four shapes, told apart lexically: a declaration (via
//! [`crate::parser::declarations`]), a `~` statement, a `target +=`
//! statement, and everything else. Plus one convention: a function
//! whose name ends in `_rng` draws a random number. That is a rule in
//! Stan's own naming, not a list of functions to keep up to date.
//!
//! Nothing here knows what a statement *means*, and a shape it does not
//! recognise is classified as an ordinary statement rather than
//! rejected.

use crate::parser::blocks::BlockKind;
use crate::parser::brace_match::CodeMask;
use crate::parser::declarations::{identifier_uses, parse_declaration};

/// The kinds of statement laplace tells apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    /// `vector[N] x;` -- legal anywhere a variable can be introduced.
    Declaration,
    /// `y ~ normal(0, 1);` -- the model block's alone.
    Sampling,
    /// `target += ...;` -- likewise.
    TargetIncrement,
    /// An assignment, a call, control flow: anything else.
    Statement,
}

impl StatementKind {
    /// How this reads in an error message.
    pub fn describe(self) -> &'static str {
        match self {
            StatementKind::Declaration => "a declaration",
            StatementKind::Sampling => "a `~` statement",
            StatementKind::TargetIncrement => "a `target +=` statement",
            StatementKind::Statement => "a statement",
        }
    }

    /// Whether a block accepts this kind of statement.
    pub fn is_legal_in(self, block: BlockKind) -> bool {
        match self {
            // A declaration belongs anywhere a variable can be
            // introduced, which is every program block.
            StatementKind::Declaration => matches!(
                block,
                BlockKind::Data
                    | BlockKind::TransformedData
                    | BlockKind::Parameters
                    | BlockKind::TransformedParameters
                    | BlockKind::Model
                    | BlockKind::GeneratedQuantities
            ),
            // `data` and `parameters` declare; they do not compute.
            StatementKind::Statement => matches!(
                block,
                BlockKind::TransformedData
                    | BlockKind::TransformedParameters
                    | BlockKind::Model
                    | BlockKind::GeneratedQuantities
            ),
            StatementKind::Sampling | StatementKind::TargetIncrement => {
                matches!(block, BlockKind::Model)
            }
        }
    }
}

/// One statement found in a body, with where it is and what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statement {
    pub kind: StatementKind,
    /// Byte offset of the statement's first real byte.
    pub offset: usize,
    /// The statement's text, trimmed, for an error message.
    pub text: String,
    /// Whether it calls a `_rng` function, which narrows where it may
    /// appear further than its kind does.
    pub draws_random: bool,
}

impl Statement {
    /// Whether this statement may appear in `block`.
    pub fn is_legal_in(&self, block: BlockKind) -> bool {
        if !self.kind.is_legal_in(block) {
            return false;
        }
        // Stan only lets you draw random numbers where the result is
        // not part of the log density.
        !self.draws_random
            || matches!(
                block,
                BlockKind::TransformedData | BlockKind::GeneratedQuantities
            )
    }

    /// Why `block` will not take this statement.
    pub fn rejection(&self, block: BlockKind) -> String {
        if !self.kind.is_legal_in(block) {
            return format!(
                "{} is not legal in `{}`",
                self.kind.describe(),
                block.keyword()
            );
        }
        format!(
            "a `_rng` function cannot be called in `{}`",
            block.keyword()
        )
    }
}

/// Split a body into its top-level statements and classify each one.
///
/// Statements nested inside a `for` or `if` body are not reported
/// separately: the enclosing statement stands for them, and a block
/// that accepts the outer one accepts what it contains.
pub fn statements(body: &str) -> Vec<Statement> {
    let mask = CodeMask::new(body);
    let bytes = body.as_bytes();
    let mut found = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        if !mask.is_real(i) {
            i += 1;
            continue;
        }
        // `${name}` is one token, braces and all.
        if bytes[i] == b'$' && bytes.get(i + 1) == Some(&b'{') {
            i = body[i + 1..]
                .find('}')
                .map_or(bytes.len(), |rel| i + 1 + rel + 1);
            continue;
        }
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    // The closing brace ends the compound statement
                    // that opened it.
                    push(body, start..i + 1, &mut found);
                    start = i + 1;
                }
            }
            b';' if depth == 0 => {
                push(body, start..i, &mut found);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    push(body, start..bytes.len(), &mut found);
    found
}

fn push(body: &str, range: std::ops::Range<usize>, out: &mut Vec<Statement>) {
    let text = &body[range.clone()];
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    let lead = text.len() - text.trim_start().len();
    out.push(classify(trimmed, range.start + lead));
}

/// Classify one statement's text.
pub fn classify(text: &str, offset: usize) -> Statement {
    let trimmed = text.trim();
    let mask = CodeMask::new(trimmed);
    let draws_random = identifier_uses(trimmed)
        .iter()
        .any(|use_| use_.is_call && use_.name.ends_with("_rng"));

    let kind = if has_real_byte(trimmed, &mask, b'~') {
        StatementKind::Sampling
    } else if is_target_increment(trimmed) {
        StatementKind::TargetIncrement
    } else if parse_declaration(trimmed.trim_end_matches(';').trim_end(), 0, 0).is_some() {
        StatementKind::Declaration
    } else {
        StatementKind::Statement
    };

    Statement {
        kind,
        offset,
        text: trimmed.to_string(),
        draws_random,
    }
}

/// Whether `text` starts with `target` followed by `+=`.
fn is_target_increment(text: &str) -> bool {
    let rest = text.trim_start();
    let Some(rest) = rest.strip_prefix("target") else {
        return false;
    };
    let rest = rest.trim_start();
    rest.starts_with("+=")
}

/// Whether `byte` appears in `text` outside comments and strings.
fn has_real_byte(text: &str, mask: &CodeMask, byte: u8) -> bool {
    text.bytes()
        .enumerate()
        .any(|(i, b)| b == byte && mask.is_real(i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(body: &str) -> Vec<StatementKind> {
        statements(body).into_iter().map(|s| s.kind).collect()
    }

    #[test]
    fn a_sampling_statement_is_recognised() {
        assert_eq!(kinds("y ~ normal(0, 1);"), vec![StatementKind::Sampling]);
    }

    #[test]
    fn a_target_increment_is_recognised() {
        assert_eq!(
            kinds("target += normal_lpdf(y | 0, 1);"),
            vec![StatementKind::TargetIncrement]
        );
        assert_eq!(kinds("target  +=  1;"), vec![StatementKind::TargetIncrement]);
    }

    #[test]
    fn a_declaration_is_recognised() {
        assert_eq!(
            kinds("vector[N] y;\nreal<lower=0> s = 1;"),
            vec![StatementKind::Declaration, StatementKind::Declaration]
        );
    }

    #[test]
    fn anything_else_is_an_ordinary_statement() {
        assert_eq!(
            kinds("x[1] = 2;\nprint(\"hi\");"),
            vec![StatementKind::Statement, StatementKind::Statement]
        );
    }

    #[test]
    fn a_compound_statement_counts_once() {
        assert_eq!(
            kinds("for (i in 1:N) {\n  y[i] ~ normal(0, 1);\n}"),
            vec![StatementKind::Sampling],
            "the `~` inside is what makes the loop a sampling statement"
        );
    }

    #[test]
    fn several_statements_are_split_on_their_semicolons() {
        assert_eq!(kinds("a = 1;\nb = 2;\nc = 3;").len(), 3);
    }

    #[test]
    fn a_semicolon_in_a_string_does_not_split_a_statement() {
        assert_eq!(kinds("print(\"a;b\");").len(), 1);
    }

    #[test]
    fn a_placeholder_token_does_not_look_like_a_brace() {
        assert_eq!(kinds("${p}_z ~ std_normal();"), vec![StatementKind::Sampling]);
    }

    #[test]
    fn placeholders_survive_classification() {
        assert_eq!(kinds("$p ~ $dist;"), vec![StatementKind::Sampling]);
    }

    #[test]
    fn an_rng_call_is_noticed() {
        let found = statements("real x = normal_rng(0, 1);");
        assert!(found[0].draws_random);
        assert_eq!(found[0].kind, StatementKind::Declaration);
    }

    #[test]
    fn a_name_merely_containing_rng_is_not_a_draw() {
        assert!(!statements("real x = rng_offset + 1;")[0].draws_random);
        assert!(!statements("real x = my_rng;")[0].draws_random);
    }

    #[test]
    fn a_statements_offset_points_at_its_first_real_byte() {
        let body = "\n  y ~ normal(0, 1);\n";
        let found = statements(body);
        assert_eq!(&body[found[0].offset..found[0].offset + 1], "y");
    }

    // ---- legality ----------------------------------------------------

    #[test]
    fn a_sampling_statement_is_only_legal_in_the_model_block() {
        let statement = classify("y ~ normal(0, 1);", 0);
        assert!(statement.is_legal_in(BlockKind::Model));
        for block in [
            BlockKind::Data,
            BlockKind::TransformedData,
            BlockKind::Parameters,
            BlockKind::TransformedParameters,
            BlockKind::GeneratedQuantities,
        ] {
            assert!(!statement.is_legal_in(block), "{block:?}");
        }
    }

    #[test]
    fn a_target_increment_is_only_legal_in_the_model_block() {
        let statement = classify("target += 1;", 0);
        assert!(statement.is_legal_in(BlockKind::Model));
        assert!(!statement.is_legal_in(BlockKind::GeneratedQuantities));
    }

    #[test]
    fn a_plain_statement_is_not_legal_where_only_declarations_go() {
        let statement = classify("x = 1;", 0);
        assert!(!statement.is_legal_in(BlockKind::Data));
        assert!(!statement.is_legal_in(BlockKind::Parameters));
        assert!(statement.is_legal_in(BlockKind::TransformedData));
        assert!(statement.is_legal_in(BlockKind::Model));
        assert!(statement.is_legal_in(BlockKind::GeneratedQuantities));
    }

    #[test]
    fn a_declaration_is_legal_in_every_program_block() {
        let statement = classify("real x;", 0);
        for block in [
            BlockKind::Data,
            BlockKind::TransformedData,
            BlockKind::Parameters,
            BlockKind::TransformedParameters,
            BlockKind::Model,
            BlockKind::GeneratedQuantities,
        ] {
            assert!(statement.is_legal_in(block), "{block:?}");
        }
    }

    #[test]
    fn an_rng_draw_is_only_legal_where_it_is_not_part_of_the_density() {
        let statement = classify("real x = normal_rng(0, 1);", 0);
        assert!(statement.is_legal_in(BlockKind::GeneratedQuantities));
        assert!(statement.is_legal_in(BlockKind::TransformedData));
        assert!(!statement.is_legal_in(BlockKind::Model));
        assert!(!statement.is_legal_in(BlockKind::TransformedParameters));
    }

    #[test]
    fn a_rejection_says_which_rule_was_broken() {
        let sampling = classify("y ~ normal(0, 1);", 0);
        assert_eq!(
            sampling.rejection(BlockKind::GeneratedQuantities),
            "a `~` statement is not legal in `generated quantities`"
        );

        let draw = classify("real x = normal_rng(0, 1);", 0);
        assert_eq!(
            draw.rejection(BlockKind::Model),
            "a `_rng` function cannot be called in `model`"
        );
    }

    #[test]
    fn classification_is_deterministic() {
        let body = "$p ~ $dist;\nreal ${p}_z = 0;";
        assert_eq!(statements(body), statements(body));
    }
}
