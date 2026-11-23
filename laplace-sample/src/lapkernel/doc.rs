//! `laplace doc pkg::kernel` / `laplace sampler doc` (Phase 4.3): a kernel's
//! doc comments, its tuning values with their `@adapt` tags, and whether its
//! step ends in `propose()` or `move()`.
//!
//! Declarations are shown as the author wrote them (the source text of each
//! tuning declaration), so the doc never disagrees with the file.

use super::ast::{Expr, ExprKind, Span, Stmt};
use super::kernel::LapKernelFile;

/// One tuning value: its declaration (without the tag), its `@adapt` tag,
/// and its `//@param` text.
#[derive(Clone, Debug, PartialEq)]
pub struct TuningDoc {
    pub name: String,
    pub decl: String,
    pub adapt: Option<String>,
    pub doc: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelDoc {
    /// What the user writes: `langevin::mala`, or the file path.
    pub title: String,
    /// `langevin 0.1.0, kernels/mala.lapkernel`, or the file path.
    pub origin: String,
    pub brief: Option<String>,
    pub tuning: Vec<TuningDoc>,
    /// `//@param` entries naming no tuning value (shown, so nothing is lost).
    pub other_params: Vec<(String, String)>,
    /// The terminals the step can end in, in source order: `propose`, `move`.
    pub ends_in: Vec<&'static str>,
}

/// The source text from `span` to the next `;` (exclusive).
fn decl_text(source: &str, span: Span) -> String {
    let start: usize = source
        .split_inclusive('\n')
        .take(span.line.saturating_sub(1))
        .map(str::len)
        .sum::<usize>()
        + span.col.saturating_sub(1);
    let rest = &source[start.min(source.len())..];
    let text = rest.split(';').next().unwrap_or(rest);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn collect_terminals(s: &Stmt, out: &mut Vec<&'static str>) {
    match s {
        Stmt::Expr(Expr {
            kind: ExprKind::Call { name, .. },
            ..
        }) => {
            for t in ["propose", "move"] {
                if name == t && !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        Stmt::Block(b) => b.stmts.iter().for_each(|x| collect_terminals(x, out)),
        Stmt::If { then, els, .. } => {
            collect_terminals(then, out);
            if let Some(e) = els {
                collect_terminals(e, out);
            }
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => collect_terminals(body, out),
        _ => {}
    }
}

pub fn kernel_doc(file: &LapKernelFile, title: &str, origin: &str) -> KernelDoc {
    let k = &file.parsed;
    let docs = &k.docs;
    let tuning: Vec<TuningDoc> = k
        .tuning
        .iter()
        .map(|d| {
            let text = decl_text(&file.source, d.span);
            let (decl, adapt) = match text.split_once("@adapt") {
                Some((a, b)) => (
                    a.trim().to_string(),
                    Some(format!("@adapt{}", b.trim_end())),
                ),
                None => (text.clone(), None),
            };
            TuningDoc {
                name: d.name.clone(),
                decl,
                adapt,
                doc: docs
                    .params
                    .iter()
                    .find(|(n, _)| *n == d.name)
                    .map(|(_, t)| t.clone()),
            }
        })
        .collect();
    let other_params = docs
        .params
        .iter()
        .filter(|(n, _)| !k.tuning.iter().any(|d| &d.name == n))
        .cloned()
        .collect();
    let mut ends_in = vec![];
    if let Some(step) = &k.step {
        step.stmts
            .iter()
            .for_each(|s| collect_terminals(s, &mut ends_in));
    }
    KernelDoc {
        title: title.to_string(),
        origin: origin.to_string(),
        brief: docs.brief.clone(),
        tuning,
        other_params,
        ends_in,
    }
}

pub fn render(d: &KernelDoc) -> String {
    let mut out = format!("{}  ({})\n\n", d.title, d.origin);
    match &d.brief {
        Some(b) => out += &format!("  {b}\n\n"),
        None => out += "  (no //@brief doc comment)\n\n",
    }
    if d.tuning.is_empty() {
        out += "Tuning values: none\n\n";
    } else {
        out += "Tuning values (set as `name = value` after the kernel, e.g. in a .lapsampler):\n";
        for t in &d.tuning {
            out += &format!("  {}", t.decl);
            if let Some(a) = &t.adapt {
                out += &format!("   {a}");
            }
            out += "\n";
            if let Some(doc) = &t.doc {
                out += &format!("      {doc}\n");
            }
        }
        out += "\n";
    }
    for (n, t) in &d.other_params {
        out += &format!("  (//@param {n}: {t} -- no tuning value of that name)\n");
    }
    let step = match d.ends_in.as_slice() {
        ["propose"] => {
            "ends in propose(): a Metropolis-Hastings proposal; laplace-sample accepts or \
                        rejects it (one uniform)"
        }
        ["move"] => {
            "ends in move(): a direct move, always taken; the kernel itself must leave the \
                     target invariant"
        }
        [] => "has no propose()/move() (it does not check)",
        _ => "ends in propose() on some paths and move() on others",
    };
    out += &format!("Step: {step}.\n");
    let adapts: Vec<&str> = d
        .tuning
        .iter()
        .filter(|t| t.adapt.is_some())
        .map(|t| t.name.as_str())
        .collect();
    out += &if adapts.is_empty() {
        "Adaptation: nothing to adapt; use it without adapt().\n".to_string()
    } else {
        format!(
            "Adaptation: adapt({}) tunes {} in warmup, then freezes them.\n",
            d.title,
            adapts.join(" and ")
        )
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped(name: &str) -> LapKernelFile {
        LapKernelFile::load(std::path::Path::new(&format!(
            "{}/kernels/{name}.lapkernel",
            env!("CARGO_MANIFEST_DIR")
        )))
        .unwrap()
    }

    #[test]
    fn the_shipped_mala_documents_itself() {
        let d = kernel_doc(
            &shipped("mala"),
            "langevin::mala",
            "langevin 0.1.0, kernels/mala.lapkernel",
        );
        assert_eq!(
            d.brief.as_deref(),
            Some("Metropolis-adjusted Langevin algorithm with a diagonal metric")
        );
        assert_eq!(d.tuning.len(), 2);
        assert_eq!(d.tuning[0].decl, "real<lower=0> eps = 1");
        assert_eq!(
            d.tuning[0].adapt.as_deref(),
            Some("@adapt(step_size, target_accept = 0.574)")
        );
        assert_eq!(
            d.tuning[0].doc.as_deref(),
            Some("Step size (adapted toward 0.574 acceptance)")
        );
        assert_eq!(
            d.tuning[1].decl,
            "vector<lower=0>[dim] m = rep_vector(1, dim)"
        );
        assert_eq!(d.tuning[1].adapt.as_deref(), Some("@adapt(metric)"));
        assert_eq!(d.ends_in, vec!["propose"]);
        let r = render(&d);
        assert!(
            r.starts_with("langevin::mala  (langevin 0.1.0, kernels/mala.lapkernel)\n"),
            "{r}"
        );
        assert!(r.contains("Step: ends in propose()"), "{r}");
        assert!(
            r.contains("adapt(langevin::mala) tunes eps and m in warmup"),
            "{r}"
        );
    }

    #[test]
    fn a_move_kernel_without_docs_or_tags() {
        let f = LapKernelFile::from_source(
            "k.lapkernel",
            "kernel k {\n  tuning {\n    real h = 0.1;\n  }\n  step {\n    if (h > 0) {\n      move(q);\n    } else {\n      \
             propose(q, 0);\n    }\n  }\n}\n"
                .into(),
        )
        .unwrap();
        let d = kernel_doc(&f, "k.lapkernel", "k.lapkernel");
        assert_eq!(d.ends_in, vec!["move", "propose"]);
        let r = render(&d);
        assert!(r.contains("(no //@brief doc comment)"), "{r}");
        assert!(r.contains("  real h = 0.1\n"), "{r}");
        assert!(
            r.contains("propose() on some paths and move() on others"),
            "{r}"
        );
        assert!(r.contains("nothing to adapt"), "{r}");
        let only_move = LapKernelFile::from_source(
            "m.lapkernel",
            "kernel m {\n  step {\n    move(q);\n  }\n}\n".into(),
        )
        .unwrap();
        assert!(render(&kernel_doc(&only_move, "m", "m")).contains("ends in move(): a direct move"));
    }
}
