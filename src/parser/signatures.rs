//! Shallow scan of a `.stan` file's top-level function signatures and their
//! preceding `// @laplace` doc comments.
//!
//! This runs both on installed packages (whole `.stan` files that are just a
//! sequence of function definitions) and on the user's own `functions { }`
//! block content. It never inspects statements *inside* a function body other
//! than to find where the body ends.

use serde::{Deserialize, Serialize};

use super::brace_match::CodeMask;

/// One parameter of a function signature: `(param_name, param_type)`.
pub type Param = (String, String);

/// A top-level function signature, with its doc comment if one was attached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionSig {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: String,
    pub doc: Option<Doc>,
}

/// Structured content of a `// @laplace` doc comment block.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Doc {
    pub brief: Option<String>,
    /// `(param_name, description)`, in the order they appear in the comment.
    pub params: Vec<Param>,
    pub return_doc: Option<String>,
    pub example: Option<String>,
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

        if let Some(sig) = parse_function_header(&source[segment_start..open_brace]) {
            sigs.push(sig);
        }

        segment_start = close_brace + 1;
    }

    sigs
}

/// Parse the text preceding a function body's `{` into a `FunctionSig`.
/// Returns `None` if it doesn't look like `<return_type> <name>(<params>)`.
fn parse_function_header(pre: &str) -> Option<FunctionSig> {
    let (header_lines, comment_lines) = split_header_and_comment(pre.trim_end());
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

    Some(FunctionSig {
        name,
        params,
        return_type,
        doc,
    })
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

/// Split a parameter list on top-level commas, ignoring commas inside `[ ]`
/// array-dimension brackets (e.g. `array[N, M] real x`).
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '[' => {
                depth += 1;
                current.push(c);
            }
            ']' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => out.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    out.push(current);
    out
}

fn parse_params(params_text: &str) -> Vec<Param> {
    split_top_level_commas(params_text)
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
fn split_header_and_comment(text: &str) -> (Vec<String>, Vec<String>) {
    let lines: Vec<&str> = text.lines().collect();
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

    (header_lines, comment_lines)
}

/// If `comment_lines` is a `// @laplace`-tagged block, parse it into a `Doc`.
/// Any other (untagged) comment block directly above a function is treated
/// as an ordinary comment, not documentation, and yields `None`.
fn parse_doc_block(comment_lines: &[String]) -> Option<Doc> {
    let stripped: Vec<String> = comment_lines
        .iter()
        .map(|l| strip_comment_prefix(l))
        .collect();

    if stripped.first().map(|s| s.trim()) != Some("@laplace") {
        return None;
    }

    Some(parse_doc_tags(&stripped[1..]))
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
        } else if !line.is_empty() {
            match active {
                ActiveField::Brief => append_opt(&mut doc.brief, line),
                ActiveField::Param => {
                    if let Some(last) = doc.params.last_mut() {
                        append_str(&mut last.1, line);
                    }
                }
                ActiveField::Return => append_opt(&mut doc.return_doc, line),
                ActiveField::Example => append_opt(&mut doc.example, line),
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
        assert_eq!(jitter.params, vec![("epsilon".to_string(), "real".to_string())]);
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
            vec![("x".to_string(), "The input, continued on another line.".to_string())]
        );
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
}
