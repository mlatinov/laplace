//! Shallow scan of a `.stan` file's top-level function signatures and their
//! preceding `// @laplace` doc comments.
//!
//! This runs both on installed packages (whole `.stan` files that are just a
//! sequence of function definitions) and on the user's own `functions { }`
//! block content. It never inspects statements *inside* a function body other
//! than to find where the body ends.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::brace_match::CodeMask;
use super::functional::{parse_functional_param, FunctionalError, FunctionalParam};
use super::types::{self, TypeCategory};

/// One parameter of a function signature: `(param_name, param_type)`.
pub type Param = (String, String);

/// A top-level function signature, with its doc comment if one was attached.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionSig {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: String,
    pub doc: Option<Doc>,
    /// Byte offset of the first non-whitespace byte of the signature
    /// header (the return type) within the scanned text. Used to attach a
    /// `pub` marker to the item that follows it.
    ///
    /// Not part of `docs.json`: offsets describe one particular parse of
    /// one particular text, so persisting them would only let them go
    /// stale. They are also excluded from equality, so two signatures
    /// compare equal when they say the same thing about a function.
    #[serde(skip)]
    pub header_offset: usize,
    /// Byte offset of the start of the *line* the item begins on --
    /// its attached `//` doc-comment block if it has one, otherwise the
    /// signature header. Provenance comments are inserted here, above
    /// the doc comment rather than between it and the function.
    #[serde(skip)]
    pub item_offset: usize,
    /// The size expressions stripped off the return type, if it carried
    /// a laplace size annotation (`vector[2] to_pair(real x)`).
    /// `return_type` is always the bare type Stan sees.
    #[serde(skip)]
    pub return_sizes: Vec<String>,
    /// Byte range of the return type's `[...]` annotation, so codegen
    /// can cut it out of the emitted text.
    #[serde(skip)]
    pub return_size_span: Option<Range<usize>>,
    /// Parameters whose type is a `func(...) -> ...` shape. A function
    /// with any of these is a higher-order function and is specialized
    /// rather than emitted as written.
    #[serde(skip)]
    pub functional_params: Vec<FunctionalParam>,
    /// Byte range of the function's body, from its opening `{` through
    /// its closing `}` inclusive. With [`FunctionSig::item_offset`] this
    /// delimits the whole definition, which is what it takes to move a
    /// higher-order function's body into a specialized copy and delete
    /// the original.
    #[serde(skip)]
    pub body_span: Option<Range<usize>>,
    /// Functional parameters that failed to parse. Collected rather than
    /// returned, so this scanner stays total -- doc extraction must not
    /// break on a malformed shape. [`crate::monomorphize`] reports them
    /// with the location of the function they belong to.
    #[serde(skip)]
    pub functional_errors: Vec<FunctionalError>,
}

impl FunctionSig {
    /// Byte range of the whole definition: doc comment, signature and
    /// body. `None` only for a signature built by hand rather than
    /// scanned.
    pub fn definition_span(&self) -> Option<Range<usize>> {
        self.body_span
            .as_ref()
            .map(|body| self.item_offset..body.end)
    }

    /// Whether this is a higher-order function: it takes at least one
    /// functional parameter, so it has no valid Stan form of its own.
    pub fn is_higher_order(&self) -> bool {
        !self.functional_params.is_empty() || !self.functional_errors.is_empty()
    }

    /// The parameters Stan sees, with the functional ones dropped --
    /// the specialized copy's parameter list.
    pub fn value_params(&self) -> Vec<&Param> {
        self.params
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.functional_params.iter().any(|f| f.index == *i))
            .map(|(_, p)| p)
            .collect()
    }
}

impl PartialEq for FunctionSig {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.params == other.params
            && self.return_type == other.return_type
            && self.doc == other.doc
    }
}

impl Eq for FunctionSig {}

/// Structured content of a `// @laplace` doc comment block.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Doc {
    pub brief: Option<String>,
    /// `(param_name, description)`, in the order they appear in the comment.
    pub params: Vec<Param>,
    pub return_doc: Option<String>,
    pub example: Option<String>,
    /// Raw LaTeX, verbatim -- never parsed or validated, just stored and
    /// passed through to docs.json / render / --html output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub math: Option<String>,
}

/// Extract every top-level function signature from `source`.
///
/// Functions without a `// @laplace`-tagged doc comment directly above them
/// still get their signature extracted, with `doc: None` — a doc comment is
/// never required.
pub fn extract_signatures(source: &str) -> Vec<FunctionSig> {
    let mask = CodeMask::new(source);
    let mut sigs = Vec::new();
    let mut segment_start = 0usize;

    while let Some(open_brace) = mask.find_real(source, segment_start, b'{') {
        let Some(close_brace) = mask.match_closing_brace(source, open_brace) else {
            // Unbalanced braces: nothing sensible to do in a shallow scan,
            // stop rather than mis-parse the remainder.
            break;
        };

        if let Some(mut sig) =
            parse_function_header(&source[segment_start..open_brace], segment_start)
        {
            sig.body_span = Some(open_brace..close_brace + 1);
            sigs.push(sig);
        }

        segment_start = close_brace + 1;
    }

    sigs
}

/// Parse the text preceding a function body's `{` into a `FunctionSig`.
/// Returns `None` if it doesn't look like `<return_type> <name>(<params>)`.
///
/// `base` is `pre`'s byte offset within the whole scanned text, so the
/// offsets recorded on the returned signature are absolute.
fn parse_function_header(pre: &str, base: usize) -> Option<FunctionSig> {
    let split = split_header_and_comment(pre.trim_end());
    let HeaderSplit {
        header_lines,
        comment_lines,
        header_line_start,
        item_line_start,
    } = split;
    let header = header_lines.join(" ");
    let header = header.trim();

    let open_paren = header.find('(')?;
    let close_paren = find_matching_paren(header, open_paren)?;
    if !header[close_paren + 1..].trim().is_empty() {
        return None;
    }

    let mut pre_paren_tokens: Vec<&str> = header[..open_paren].split_whitespace().collect();
    let name = pre_paren_tokens.pop()?.to_string();
    if !is_identifier(&name) {
        return None;
    }
    let return_type = pre_paren_tokens.join(" ");
    if return_type.is_empty() {
        return None;
    }

    let params = parse_params(&header[open_paren + 1..close_paren]);
    let doc = parse_doc_block(&comment_lines);

    // The first header line's indentation is not part of the header, so
    // `header_offset` points at the return type itself.
    let indent = pre[header_line_start..]
        .bytes()
        .take_while(|b| matches!(b, b' ' | b'\t'))
        .count();
    let header_start = header_line_start + indent;

    // A laplace size annotation on the return type is read off here and
    // cut from the output later: `vector[2] f(...)` is laplace source,
    // `vector f(...)` is what Stan gets.
    let return_ty = types::parse_type(&return_type);
    let strip_sizes = return_ty.is_sized() && return_ty.category != TypeCategory::Array;
    let return_size_span = strip_sizes
        .then(|| return_size_span(pre, header_start))
        .flatten();
    let (return_type, return_sizes) = if strip_sizes {
        (return_ty.bare.clone(), return_ty.sizes.clone())
    } else {
        (return_type, Vec::new())
    };

    let mut functional_params = Vec::new();
    let mut functional_errors = Vec::new();
    for (index, (param_name, param_type)) in params.iter().enumerate() {
        match parse_functional_param(param_name, param_type, index) {
            None => {}
            Some(Ok(param)) => functional_params.push(param),
            Some(Err(error)) => functional_errors.push(error),
        }
    }

    Some(FunctionSig {
        name,
        params,
        return_type,
        doc,
        header_offset: base + header_start,
        item_offset: base + item_line_start,
        return_sizes,
        return_size_span: return_size_span.map(|r| base + r.start..base + r.end),
        // Filled in by the caller, which is where the body's braces are
        // already known.
        body_span: None,
        functional_params,
        functional_errors,
    })
}

/// Byte range, within `pre`, of the `[...]` on a return type whose
/// header starts at `header_start`.
///
/// Located in the source text rather than in the reconstructed header
/// string, because a header spanning several lines is joined with single
/// spaces and its offsets no longer line up with the file.
fn return_size_span(pre: &str, header_start: usize) -> Option<Range<usize>> {
    let after = &pre[header_start..];
    // The return type ends where the parameter list begins.
    let params_at = after.find('(')?;
    let open = after[..params_at].find('[')?;
    let close = types::matching_bracket(after, open)?;
    Some(header_start + open..header_start + close + 1)
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn find_matching_paren(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_params(params_text: &str) -> Vec<Param> {
    // Commas nest inside array dimensions (`array[N, M] real x`) and
    // inside a functional parameter's argument list
    // (`func(real, int) -> real f`), so splitting is bracket- and
    // paren-aware -- see `types::split_top_level_args`.
    types::split_top_level_args(params_text)
        .into_iter()
        .filter_map(|chunk| {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                return None;
            }
            let mut tokens: Vec<&str> = chunk.split_whitespace().collect();
            let name = tokens.pop()?.to_string();
            let ty = tokens.join(" ");
            if ty.is_empty() {
                return None;
            }
            Some((name, ty))
        })
        .collect()
}

/// Split `pre` (everything between the previous function's `}` and this
/// function's `{`) into the header lines (the signature itself, normally one
/// line) and the doc comment lines directly above it, if any.
///
/// A comment block only counts as "directly above" if there is no blank line
/// between its last line and the header — matching the "immediately
/// preceded" requirement.
fn split_header_and_comment(text: &str) -> HeaderSplit {
    let lines: Vec<&str> = text.lines().collect();
    let starts = line_starts(text);
    // The offset just past the end: what an empty split points at.
    let past_end = text.len();
    let start_of = |idx: usize| starts.get(idx).copied().unwrap_or(past_end);

    let mut idx = lines.len();

    let mut header_lines = Vec::new();
    while idx > 0 {
        let t = lines[idx - 1].trim();
        if t.is_empty() || t.starts_with("//") {
            break;
        }
        header_lines.push(lines[idx - 1].to_string());
        idx -= 1;
    }
    header_lines.reverse();
    let header_line_start = start_of(idx);

    let mut comment_lines = Vec::new();
    if idx > 0 && lines[idx - 1].trim().starts_with("//") {
        while idx > 0 {
            let t = lines[idx - 1].trim();
            if !t.starts_with("//") {
                break;
            }
            comment_lines.push(lines[idx - 1].to_string());
            idx -= 1;
        }
        comment_lines.reverse();
    }

    HeaderSplit {
        header_lines,
        comment_lines,
        header_line_start,
        item_line_start: start_of(idx),
    }
}

/// [`split_header_and_comment`]'s result: the two line groups plus where
/// each group starts, in bytes, within the text it was split from.
struct HeaderSplit {
    header_lines: Vec<String>,
    comment_lines: Vec<String>,
    header_line_start: usize,
    item_line_start: usize,
}

/// Byte offset of the start of every line in `text`, in order. One entry
/// per line as `str::lines` counts them, so the two can be indexed
/// together.
fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    if !text.is_empty() {
        starts.push(0);
    }
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' && i + 1 < text.len() {
            starts.push(i + 1);
        }
    }
    starts
}

/// If `comment_lines` contains a `// @laplace` marker, parse everything after
/// the last such marker into a `Doc`. Ordinary comment lines glued on above
/// the marker (a section banner, a maintainer's note) are not documentation
/// and are ignored. A comment block with no marker at all is an ordinary
/// comment and yields `None`.
fn parse_doc_block(comment_lines: &[String]) -> Option<Doc> {
    let stripped: Vec<String> = comment_lines
        .iter()
        .map(|l| strip_comment_prefix(l))
        .collect();

    let marker = stripped.iter().rposition(|s| s.trim() == "@laplace")?;
    Some(parse_doc_tags(&stripped[marker + 1..]))
}

fn strip_comment_prefix(line: &str) -> String {
    let rest = line.trim_start().strip_prefix("//").unwrap_or(line);
    rest.strip_prefix(' ').unwrap_or(rest).to_string()
}

enum ActiveField {
    None,
    Brief,
    Param,
    Return,
    Example,
    Math,
}

fn parse_doc_tags(lines: &[String]) -> Doc {
    let mut doc = Doc::default();
    let mut active = ActiveField::None;

    for raw in lines {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("@brief") {
            doc.brief = Some(rest.trim().to_string());
            active = ActiveField::Brief;
        } else if let Some(rest) = line.strip_prefix("@param") {
            let rest = rest.trim();
            let (name, desc) = match rest.split_once(char::is_whitespace) {
                Some((n, d)) => (n.to_string(), d.trim().to_string()),
                None => (rest.to_string(), String::new()),
            };
            doc.params.push((name, desc));
            active = ActiveField::Param;
        } else if let Some(rest) = line.strip_prefix("@return") {
            doc.return_doc = Some(rest.trim().to_string());
            active = ActiveField::Return;
        } else if let Some(rest) = line.strip_prefix("@example") {
            doc.example = Some(rest.trim().to_string());
            active = ActiveField::Example;
        } else if let Some(rest) = line.strip_prefix("@math") {
            doc.math = Some(rest.trim().to_string());
            active = ActiveField::Math;
        } else if !line.is_empty() {
            match active {
                ActiveField::Brief => append_opt(&mut doc.brief, line),
                ActiveField::Param => {
                    if let Some(last) = doc.params.last_mut() {
                        append_str(&mut last.1, line);
                    }
                }
                ActiveField::Return => append_opt(&mut doc.return_doc, line),
                // @example and @math are code/LaTeX, not prose: preserve
                // line breaks verbatim instead of collapsing to one line,
                // since whitespace-sensitive content (multi-statement Stan
                // code, LaTeX `cases` environments) would be silently
                // corrupted by joining with a space.
                ActiveField::Example => append_opt_lines(&mut doc.example, line),
                ActiveField::Math => append_opt_lines(&mut doc.math, line),
                ActiveField::None => {}
            }
        }
    }

    doc
}

fn append_opt(field: &mut Option<String>, extra: &str) {
    match field {
        Some(s) => append_str(s, extra),
        None => *field = Some(extra.to_string()),
    }
}

fn append_opt_lines(field: &mut Option<String>, extra: &str) {
    match field {
        Some(s) => {
            s.push('\n');
            s.push_str(extra);
        }
        None => *field = Some(extra.to_string()),
    }
}

fn append_str(field: &mut String, extra: &str) {
    if !field.is_empty() {
        field.push(' ');
    }
    field.push_str(extra);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `gps::rbf_cov` example referenced throughout laplace-project-plan.md.
    const RBF_COV_PACKAGE: &str = r#"
// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @param x Vector of input locations.
// @param alpha Marginal standard deviation of the GP.
// @param rho Length-scale of the GP.
// @return An N x N positive semi-definite covariance matrix.
// @example rbf_cov(x, 1.0, 0.5)
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}

// Not a laplace doc comment, just a regular note for maintainers.
real jitter(real epsilon) {
  return epsilon;
}

matrix rbf_cov_jittered(vector x, real alpha, real rho, real epsilon) {
  matrix[rows(x), rows(x)] k = rbf_cov(x, alpha, rho);
  for (i in 1:rows(x)) {
    k[i, i] += jitter(epsilon);
  }
  return k;
}
"#;

    #[test]
    fn documented_function_extracts_full_doc() {
        let sigs = extract_signatures(RBF_COV_PACKAGE);
        let rbf_cov = sigs.iter().find(|s| s.name == "rbf_cov").unwrap();

        assert_eq!(rbf_cov.return_type, "matrix");
        assert_eq!(
            rbf_cov.params,
            vec![
                ("x".to_string(), "vector".to_string()),
                ("alpha".to_string(), "real".to_string()),
                ("rho".to_string(), "real".to_string()),
            ]
        );

        let doc = rbf_cov.doc.as_ref().expect("rbf_cov should have a doc");
        assert_eq!(
            doc.brief.as_deref(),
            Some("Squared exponential (RBF) covariance matrix.")
        );
        assert_eq!(
            doc.params,
            vec![
                ("x".to_string(), "Vector of input locations.".to_string()),
                (
                    "alpha".to_string(),
                    "Marginal standard deviation of the GP.".to_string()
                ),
                ("rho".to_string(), "Length-scale of the GP.".to_string()),
            ]
        );
        assert_eq!(
            doc.return_doc.as_deref(),
            Some("An N x N positive semi-definite covariance matrix.")
        );
        assert_eq!(doc.example.as_deref(), Some("rbf_cov(x, 1.0, 0.5)"));
    }

    #[test]
    fn undocumented_function_still_extracts_signature() {
        let sigs = extract_signatures(RBF_COV_PACKAGE);
        let jitter = sigs.iter().find(|s| s.name == "jitter").unwrap();

        assert_eq!(jitter.return_type, "real");
        assert_eq!(
            jitter.params,
            vec![("epsilon".to_string(), "real".to_string())]
        );
        assert_eq!(jitter.doc, None);
    }

    #[test]
    fn plain_comment_above_function_is_not_treated_as_doc() {
        // `jitter` is preceded by a `//` comment that doesn't start with
        // `@laplace` -- it must not be picked up as documentation.
        let sigs = extract_signatures(RBF_COV_PACKAGE);
        let jitter = sigs.iter().find(|s| s.name == "jitter").unwrap();
        assert!(jitter.doc.is_none());
    }

    #[test]
    fn extracts_every_top_level_function_in_order() {
        let sigs = extract_signatures(RBF_COV_PACKAGE);
        let names: Vec<&str> = sigs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["rbf_cov", "jitter", "rbf_cov_jittered"]);
    }

    #[test]
    fn braces_inside_string_literals_do_not_confuse_body_matching() {
        let source = r#"
real noisy(real x) {
  print("debug: { not a real brace }");
  return x;
}

real after(real y) {
  return y;
}
"#;
        let sigs = extract_signatures(source);
        let names: Vec<&str> = sigs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["noisy", "after"]);
    }

    #[test]
    fn braces_inside_line_comments_do_not_confuse_body_matching() {
        let source = r#"
real noisy(real x) {
  // unmatched brace in a comment: {
  return x;
}

real after(real y) {
  return y;
}
"#;
        let sigs = extract_signatures(source);
        let names: Vec<&str> = sigs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["noisy", "after"]);
    }

    #[test]
    fn array_typed_params_split_correctly_on_top_level_commas() {
        let source = r#"
// @laplace
// @brief Sum an array of reals.
// @param xs Values to sum.
// @return The sum.
real array_sum(array[N] real xs) {
  return sum(xs);
}
"#;
        let sigs = extract_signatures(source);
        assert_eq!(sigs.len(), 1);
        assert_eq!(
            sigs[0].params,
            vec![("xs".to_string(), "array[N] real".to_string())]
        );
    }

    #[test]
    fn multiple_params_with_array_dimensions_are_not_split_on_inner_commas() {
        let source = r#"
matrix combine(array[N, M] real xs, matrix[N, N] k) {
  return k;
}
"#;
        let sigs = extract_signatures(source);
        assert_eq!(
            sigs[0].params,
            vec![
                ("xs".to_string(), "array[N, M] real".to_string()),
                ("k".to_string(), "matrix[N, N]".to_string()),
            ]
        );
    }

    #[test]
    fn blank_line_between_comment_and_function_means_no_doc() {
        let source = r#"
// @laplace
// @brief This comment is not attached, there's a blank line below.

real detached(real x) {
  return x;
}
"#;
        let sigs = extract_signatures(source);
        assert_eq!(sigs[0].doc, None);
    }

    #[test]
    fn plain_comment_glued_above_the_marker_does_not_hide_the_doc() {
        let source = r#"
// 16 OU kernel ==========================================
// @laplace
// @brief Ornstein-Uhlenbeck kernel.
// @param x Input locations.
real ou(real x) {
  return x;
}
"#;
        let doc = extract_signatures(source)[0]
            .doc
            .clone()
            .expect("doc attached");
        assert_eq!(doc.brief.as_deref(), Some("Ornstein-Uhlenbeck kernel."));
        assert_eq!(
            doc.params,
            vec![("x".to_string(), "Input locations.".to_string())]
        );
    }

    #[test]
    fn multiline_tag_descriptions_are_joined() {
        let source = r#"
// @laplace
// @brief A brief that
// wraps onto a second line.
// @param x The input,
//   continued on another line.
// @return Nothing interesting.
real wrapped(real x) {
  return x;
}
"#;
        let sigs = extract_signatures(source);
        let doc = sigs[0].doc.as_ref().unwrap();
        assert_eq!(
            doc.brief.as_deref(),
            Some("A brief that wraps onto a second line.")
        );
        assert_eq!(
            doc.params,
            vec![(
                "x".to_string(),
                "The input, continued on another line.".to_string()
            )]
        );
    }

    #[test]
    fn header_offset_points_at_the_return_type_and_item_offset_at_the_doc_block() {
        let source =
            "real a() {\n  return 1;\n}\n\n// @laplace\n// @brief B.\nreal b() {\n  return 2;\n}\n";
        let sigs = extract_signatures(source);

        let a = &sigs[0];
        assert_eq!(&source[a.header_offset..a.header_offset + 4], "real");
        assert_eq!(a.item_offset, a.header_offset, "no doc comment attached");

        let b = &sigs[1];
        assert_eq!(&source[b.header_offset..b.header_offset + 6], "real b");
        assert!(
            source[b.item_offset..].starts_with("// @laplace"),
            "item_offset should point at the start of the attached doc block, got {:?}",
            &source[b.item_offset..b.item_offset + 12]
        );
    }

    #[test]
    fn header_offset_skips_indentation() {
        // How a function inside a `functions { }` wrapper looks.
        let source = "  real indented(real x) {\n    return x;\n  }\n";
        let sigs = extract_signatures(source);
        assert_eq!(sigs[0].header_offset, 2);
        assert_eq!(
            sigs[0].item_offset, 0,
            "the line start, indentation included"
        );
    }

    #[test]
    fn offsets_are_excluded_from_signature_equality() {
        let one = extract_signatures("real f(real x) {\n  return x;\n}\n");
        let two = extract_signatures("\n\nreal f(real x) {\n  return x;\n}\n");
        assert_ne!(one[0].header_offset, two[0].header_offset);
        assert_eq!(
            one[0], two[0],
            "equality compares what a signature says, not where it is"
        );
    }

    // ---- sized return types + functional parameters -----------------

    #[test]
    fn a_sized_return_type_is_split_off_and_its_span_recorded() {
        let source = "vector[2] to_pair(real x) {\n  return [x, x * 2]';\n}\n";
        let sigs = extract_signatures(source);
        assert_eq!(sigs[0].return_type, "vector", "Stan sees the bare type");
        assert_eq!(sigs[0].return_sizes, vec!["2"]);
        let span = sigs[0].return_size_span.clone().unwrap();
        assert_eq!(&source[span], "[2]");
    }

    #[test]
    fn a_parameter_dependent_return_size_is_kept_verbatim() {
        let source = "vector[K] basis(real t, int K) {\n  return rep_vector(t, K);\n}\n";
        let sigs = extract_signatures(source);
        assert_eq!(sigs[0].return_sizes, vec!["K"]);
        assert_eq!(sigs[0].return_type, "vector");
    }

    #[test]
    fn a_two_dimensional_return_size_keeps_both() {
        let source = "matrix[R, C] grid(int R, int C) {\n  return rep_matrix(0, R, C);\n}\n";
        let sigs = extract_signatures(source);
        assert_eq!(sigs[0].return_sizes, vec!["R", "C"]);
    }

    #[test]
    fn an_unsized_return_type_records_no_sizes_and_no_span() {
        let sigs = extract_signatures("vector f(real x) {\n  return [x]';\n}\n");
        assert!(sigs[0].return_sizes.is_empty());
        assert!(sigs[0].return_size_span.is_none());
    }

    #[test]
    fn an_array_return_type_is_left_alone() {
        // `array[] real` is ordinary Stan: the brackets are the rank, not
        // a laplace size annotation, and must survive into the output.
        let sigs = extract_signatures("array[] real f(real x) {\n  return {x};\n}\n");
        assert_eq!(sigs[0].return_type, "array[] real");
        assert!(sigs[0].return_size_span.is_none());
    }

    #[test]
    fn a_functional_parameter_is_recognised_and_kept_in_order() {
        let source = "real apply_twice(real x, func(real) -> real f) {\n  return f(f(x));\n}\n";
        let sigs = extract_signatures(source);
        let sig = &sigs[0];
        assert!(sig.is_higher_order());
        assert_eq!(sig.params.len(), 2, "the functional param is still a param");
        assert_eq!(sig.functional_params.len(), 1);
        assert_eq!(sig.functional_params[0].name, "f");
        assert_eq!(sig.functional_params[0].index, 1);
        assert_eq!(sig.functional_params[0].shape(), "func(real) -> real");
        // The specialized copy keeps only the value parameters.
        assert_eq!(
            sig.value_params()
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
    }

    #[test]
    fn a_functional_parameters_own_commas_do_not_split_the_parameter_list() {
        let source = "real h(real x, func(real, int) -> vector f, int k) {\n  return x;\n}\n";
        let sigs = extract_signatures(source);
        assert_eq!(
            sigs[0]
                .params
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["x", "f", "k"]
        );
        assert_eq!(sigs[0].functional_params[0].arg_types.len(), 2);
        assert_eq!(sigs[0].functional_params[0].index, 1);
    }

    #[test]
    fn two_functional_parameters_are_both_recorded_in_order() {
        let source = "real h(func(real) -> real f, func(real) -> real g) {\n  return f(g(1));\n}\n";
        let sigs = extract_signatures(source);
        assert_eq!(
            sigs[0]
                .functional_params
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["f", "g"]
        );
        assert!(sigs[0].value_params().is_empty());
    }

    #[test]
    fn an_ordinary_function_is_not_higher_order() {
        let sigs = extract_signatures("real f(real x) {\n  return x;\n}\n");
        assert!(!sigs[0].is_higher_order());
        assert!(sigs[0].functional_params.is_empty());
    }

    #[test]
    fn a_malformed_functional_shape_is_collected_not_fatal() {
        // The scanner must stay total: doc extraction runs over the same
        // text and cannot be allowed to fail on a bad shape.
        let sigs = extract_signatures("real h(real x, func(real) real f) {\n  return x;\n}\n");
        assert_eq!(sigs.len(), 1);
        assert!(sigs[0].functional_params.is_empty());
        assert_eq!(sigs[0].functional_errors.len(), 1);
        assert!(sigs[0].is_higher_order(), "still not emittable as written");
    }

    #[test]
    fn the_definition_span_covers_the_doc_comment_signature_and_body() {
        let source = "real a() {\n  return 1;\n}\n\n// @laplace\n// @brief B.\nreal b(real x) {\n  return x;\n}\n";
        let sigs = extract_signatures(source);

        let a = sigs[0].definition_span().unwrap();
        assert_eq!(&source[a], "real a() {\n  return 1;\n}");

        let b = sigs[1].definition_span().unwrap();
        assert_eq!(
            &source[b],
            "// @laplace\n// @brief B.\nreal b(real x) {\n  return x;\n}"
        );
    }

    #[test]
    fn the_body_span_is_just_the_braces() {
        let source = "real f(real x) {\n  return x;\n}\n";
        let span = extract_signatures(source)[0].body_span.clone().unwrap();
        assert_eq!(&source[span], "{\n  return x;\n}");
    }

    #[test]
    fn empty_source_yields_no_signatures() {
        assert_eq!(extract_signatures(""), vec![]);
    }

    #[test]
    fn zero_argument_function() {
        let source = "real constant_one() {\n  return 1;\n}\n";
        let sigs = extract_signatures(source);
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "constant_one");
        assert!(sigs[0].params.is_empty());
    }

    #[test]
    fn function_without_math_tag_has_no_math() {
        let sigs = extract_signatures(RBF_COV_PACKAGE);
        let rbf_cov = sigs.iter().find(|s| s.name == "rbf_cov").unwrap();
        assert_eq!(rbf_cov.doc.as_ref().unwrap().math, None);
    }

    #[test]
    fn multiline_math_preserves_line_breaks_verbatim() {
        let source = r#"
// @laplace
// @brief RBF kernel.
// @math k(x, x') = \alpha^2 \exp\left(
//   -\frac{(x - x')^2}{2 \rho^2}
// \right)
// @param x Vector of input locations.
matrix rbf_cov(vector x) {
  return x;
}
"#;
        let sigs = extract_signatures(source);
        let doc = sigs[0].doc.as_ref().unwrap();
        assert_eq!(
            doc.math.as_deref(),
            Some("k(x, x') = \\alpha^2 \\exp\\left(\n-\\frac{(x - x')^2}{2 \\rho^2}\n\\right)")
        );
    }

    #[test]
    fn math_with_latex_special_characters_is_not_mistaken_for_a_new_tag() {
        // Backslashes, ampersands, and `\\` line breaks inside a `cases`
        // environment must never be parsed as a new `@tag` boundary.
        let source = r#"
// @laplace
// @brief Piecewise function.
// @math f(x) = \begin{cases}
//   x^2 & \text{if } x \geq 0 \\
//   -x^2 & \text{if } x < 0
// \end{cases}
real piecewise(real x) {
  return x;
}
"#;
        let sigs = extract_signatures(source);
        let doc = sigs[0].doc.as_ref().unwrap();
        assert_eq!(
            doc.math.as_deref(),
            Some(
                "f(x) = \\begin{cases}\nx^2 & \\text{if } x \\geq 0 \\\\\n-x^2 & \\text{if } x < 0\n\\end{cases}"
            )
        );
    }

    #[test]
    fn multiline_example_preserves_line_breaks() {
        let source = r#"
// @laplace
// @brief Fits a model in two steps.
// @example real mu = compute_mean(x);
//   real sigma = compute_sd(x);
real fit(vector x) {
  return x[1];
}
"#;
        let sigs = extract_signatures(source);
        let doc = sigs[0].doc.as_ref().unwrap();
        assert_eq!(
            doc.example.as_deref(),
            Some("real mu = compute_mean(x);\nreal sigma = compute_sd(x);")
        );
    }

    #[test]
    fn example_as_last_tag_does_not_swallow_function_signature() {
        // @example directly followed by the function declaration, with no
        // trailing @tag -- the most common real case. Capture must stop at
        // the comment block boundary, not spill into the header.
        let source = r#"
// @laplace
// @brief Sums an array.
// @example array_sum({1.0, 2.0, 3.0})
real array_sum(array[N] real xs) {
  return sum(xs);
}
"#;
        let sigs = extract_signatures(source);
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "array_sum");
        let doc = sigs[0].doc.as_ref().unwrap();
        assert_eq!(doc.example.as_deref(), Some("array_sum({1.0, 2.0, 3.0})"));
    }
}
