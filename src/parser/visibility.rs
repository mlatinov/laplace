//! `pub` visibility for library items.
//!
//! In a `.laplacelib` file every item is private to its package unless it
//! is marked `pub`:
//!
//! ```stan
//! pub real mean_(vector x) {       // part of the package's public API
//!   return sum_values(x) / num_elements(x);
//! }
//!
//! real sum_values(vector x) {      // private helper
//!   return sum(x);
//! }
//! ```
//!
//! Private is an *access* rule, not hiding: a private item is still
//! compiled into the output (public items call it), it simply cannot be
//! named from outside its own package.
//!
//! `pub` never reaches the generated `.stan` file -- it is stripped while
//! the file is parsed, which is why this module reports a *cut* range per
//! marker rather than rewriting anything itself.
//!
//! # Items, not functions
//!
//! Visibility attaches to a [`LibraryItem`], of which functions are
//! currently the only [`ItemKind`]. Templates and macros become further
//! kinds without changing how visibility is scanned, attached or checked.

use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::{is_ident_char, CodeMask};
use crate::parser::signatures::FunctionSig;

/// The keyword, spelled once.
pub const PUB_KEYWORD: &str = "pub";

/// Whether an item is part of its package's public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

impl Visibility {
    pub fn is_public(self) -> bool {
        matches!(self, Visibility::Public)
    }

    /// How the visibility reads in a provenance comment.
    pub fn label(self) -> &'static str {
        match self {
            Visibility::Public => "pub",
            Visibility::Private => "private",
        }
    }
}

/// The kinds of item a library file can define. Templates (`@template`)
/// and macros (`@macro`) join this enum in later patches; everything that
/// consumes [`LibraryItem`] is written against the enum rather than
/// against functions specifically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Function,
    Template,
    Macro,
}

impl ItemKind {
    pub fn describe(self) -> &'static str {
        match self {
            ItemKind::Function => "function",
            ItemKind::Template => "template",
            ItemKind::Macro => "macro",
        }
    }
}

/// One named thing a library file defines, with its visibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryItem {
    pub name: String,
    pub kind: ItemKind,
    pub visibility: Visibility,
    /// Byte offset of the item's first real byte in the parsed body.
    pub header_offset: usize,
    /// Byte offset of the start of the line the item begins on, doc
    /// comment included.
    pub item_offset: usize,
}

/// A `pub` keyword found in a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PubMarker {
    /// The keyword plus the whitespace separating it from the item: the
    /// range to cut so that `pub` never reaches the output.
    pub cut: Range<usize>,
    /// Where the item itself starts, in the original source.
    pub item_start: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VisibilityError {
    #[error("`pub` must be followed by a library item")]
    DanglingPub { offset: usize },
}

impl VisibilityError {
    /// Byte offset in the file the error is about.
    pub fn offset(&self) -> usize {
        match self {
            VisibilityError::DanglingPub { offset } => *offset,
        }
    }

    /// The `help:` line to print under the error.
    pub fn help(&self) -> &'static str {
        match self {
            VisibilityError::DanglingPub { .. } => {
                "write `pub` directly before a function definition, e.g. `pub real mean_(vector x)`"
            }
        }
    }
}

/// Find every `pub` marker inside `regions` of `source`.
///
/// `regions` are the byte ranges where item definitions live: in a
/// `.laplacelib` file, the text outside every top-level block plus the
/// *body* of a `functions { }` wrapper. A `pub` is only a marker at a
/// region's own brace depth zero, so the word appearing inside a function
/// body is left alone.
pub fn find_pub_markers(
    source: &str,
    regions: &[Range<usize>],
) -> Result<Vec<PubMarker>, VisibilityError> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut markers = Vec::new();

    for region in regions {
        let mut depth = 0usize;
        let mut i = region.start;
        while i < region.end.min(bytes.len()) {
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

            if depth > 0 || !is_ident_char(bytes[i]) {
                i += 1;
                continue;
            }

            let word_end = end_of_identifier(bytes, i);
            let is_word_start = i == 0 || !is_ident_char(bytes[i - 1]);
            if is_word_start && &source[i..word_end] == PUB_KEYWORD {
                let item_start = skip_whitespace(bytes, word_end);
                if item_start == word_end || item_start >= region.end {
                    return Err(VisibilityError::DanglingPub { offset: i });
                }
                markers.push(PubMarker {
                    cut: i..item_start,
                    item_start,
                });
            }
            i = word_end;
        }
    }

    markers.sort_by_key(|m| m.cut.start);
    Ok(markers)
}

fn end_of_identifier(bytes: &[u8], from: usize) -> usize {
    let mut end = from;
    while end < bytes.len() && is_ident_char(bytes[end]) {
        end += 1;
    }
    end
}

fn skip_whitespace(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    while at < bytes.len() && bytes[at].is_ascii_whitespace() {
        at += 1;
    }
    at
}

/// Pair each item defined in `body` with its visibility.
///
/// `pub_offsets` are where the stripped `pub` markers ended up in `body`
/// -- the [`PubMarker::item_start`] offsets mapped through the cuts, which
/// is exactly the first byte of the item they marked. A marker that lands
/// on no item's header is a `pub` that was attached to something this
/// version of laplace does not recognise as an item.
pub fn resolve_items(
    sigs: &[FunctionSig],
    pub_offsets: &[usize],
) -> Result<Vec<LibraryItem>, VisibilityError> {
    let items: Vec<LibraryItem> = sigs
        .iter()
        .map(|sig| LibraryItem {
            name: sig.name.clone(),
            kind: ItemKind::Function,
            visibility: if pub_offsets.contains(&sig.header_offset) {
                Visibility::Public
            } else {
                Visibility::Private
            },
            header_offset: sig.header_offset,
            item_offset: sig.item_offset,
        })
        .collect();

    if let Some(stray) = pub_offsets
        .iter()
        .find(|at| !items.iter().any(|item| item.header_offset == **at))
    {
        return Err(VisibilityError::DanglingPub { offset: *stray });
    }

    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::origin::apply_cuts;
    use crate::parser::signatures::extract_signatures;

    /// The whole file is one region: bare function definitions, no blocks.
    fn whole(source: &str) -> Vec<Range<usize>> {
        let whole = 0..source.len();
        vec![whole]
    }

    /// Strip the markers the way the `.laplacelib` parser does, then
    /// resolve items against the stripped body.
    fn items_of(source: &str) -> Vec<LibraryItem> {
        let markers = find_pub_markers(source, &whole(source)).unwrap();
        let cuts: Vec<Range<usize>> = markers.iter().map(|m| m.cut.clone()).collect();
        let starts: Vec<usize> = markers.iter().map(|m| m.item_start).collect();
        let cut = apply_cuts(source, &cuts, &starts);
        resolve_items(&extract_signatures(&cut.text), &cut.markers).unwrap()
    }

    fn public_names(source: &str) -> Vec<String> {
        items_of(source)
            .into_iter()
            .filter(|i| i.visibility.is_public())
            .map(|i| i.name)
            .collect()
    }

    #[test]
    fn items_are_private_unless_marked_pub() {
        let source = "pub real a(real x) {\n  return b(x);\n}\n\nreal b(real x) {\n  return x;\n}\n";
        let items = items_of(source);
        assert_eq!(
            items
                .iter()
                .map(|i| (i.name.as_str(), i.visibility))
                .collect::<Vec<_>>(),
            vec![("a", Visibility::Public), ("b", Visibility::Private)]
        );
        assert!(items.iter().all(|i| i.kind == ItemKind::Function));
    }

    #[test]
    fn a_file_with_no_pub_has_no_public_items() {
        let source = "real a() {\n  return 1;\n}\nreal b() {\n  return 2;\n}\n";
        assert!(public_names(source).is_empty());
    }

    #[test]
    fn pub_is_stripped_from_the_body() {
        let source = "pub real a() {\n  return 1;\n}\n";
        let markers = find_pub_markers(source, &whole(source)).unwrap();
        let cuts: Vec<Range<usize>> = markers.iter().map(|m| m.cut.clone()).collect();
        let text = apply_cuts(source, &cuts, &[]).text;
        assert_eq!(text, "real a() {\n  return 1;\n}\n");
        assert!(!text.contains("pub"));
    }

    #[test]
    fn a_doc_comment_above_pub_still_attaches_to_the_function() {
        let source = "// @laplace\n// @brief Mean.\npub real mean_(vector x) {\n  return 1;\n}\n";
        let markers = find_pub_markers(source, &whole(source)).unwrap();
        let cuts: Vec<Range<usize>> = markers.iter().map(|m| m.cut.clone()).collect();
        let starts: Vec<usize> = markers.iter().map(|m| m.item_start).collect();
        let cut = apply_cuts(source, &cuts, &starts);
        let sigs = extract_signatures(&cut.text);
        let items = resolve_items(&sigs, &cut.markers).unwrap();

        assert_eq!(items[0].visibility, Visibility::Public);
        assert_eq!(
            sigs[0].doc.as_ref().unwrap().brief.as_deref(),
            Some("Mean."),
            "stripping `pub` must not detach the doc comment"
        );
    }

    #[test]
    fn the_word_pub_inside_a_function_body_is_not_a_marker() {
        let source = "real f() {\n  real pub = 1;\n  return pub;\n}\n";
        assert!(find_pub_markers(source, &whole(source)).unwrap().is_empty());
        assert!(public_names(source).is_empty());
    }

    #[test]
    fn the_word_pub_in_a_comment_or_string_is_not_a_marker() {
        let source = "// pub real a() {}\nreal b() {\n  print(\"pub real c()\");\n}\n";
        assert!(find_pub_markers(source, &whole(source)).unwrap().is_empty());
    }

    #[test]
    fn an_identifier_merely_starting_with_pub_is_not_a_marker() {
        let source = "real publish(real x) {\n  return x;\n}\n";
        assert!(find_pub_markers(source, &whole(source)).unwrap().is_empty());
    }

    #[test]
    fn pub_separated_by_a_newline_still_marks_the_next_item() {
        let source = "pub\nreal a() {\n  return 1;\n}\n";
        assert_eq!(public_names(source), vec!["a"]);
    }

    #[test]
    fn pub_with_nothing_after_it_is_an_error() {
        let source = "real a() {\n  return 1;\n}\npub\n";
        let err = find_pub_markers(source, &whole(source)).unwrap_err();
        assert!(matches!(err, VisibilityError::DanglingPub { .. }), "{err:?}");
        assert_eq!(&source[err.offset()..err.offset() + 3], "pub");
    }

    #[test]
    fn pub_attached_to_something_that_is_not_an_item_is_an_error() {
        // `pub` in front of a trailing comment: nothing laplace recognises
        // as a library item follows it.
        let source = "real a() {\n  return 1;\n}\n\npub\n// trailing note\n";
        let markers = find_pub_markers(source, &whole(source)).unwrap();
        let cuts: Vec<Range<usize>> = markers.iter().map(|m| m.cut.clone()).collect();
        let starts: Vec<usize> = markers.iter().map(|m| m.item_start).collect();
        let cut = apply_cuts(source, &cuts, &starts);
        let err = resolve_items(&extract_signatures(&cut.text), &cut.markers).unwrap_err();
        assert!(matches!(err, VisibilityError::DanglingPub { .. }), "{err:?}");
    }

    #[test]
    fn markers_outside_the_given_regions_are_ignored() {
        let source = "pub real a() {\n  return 1;\n}\npub real b() {\n  return 2;\n}\n";
        let cutoff = source.find("pub real b").unwrap();
        let region = 0..cutoff;
        let markers = find_pub_markers(source, std::slice::from_ref(&region)).unwrap();
        assert_eq!(markers.len(), 1);
    }

    #[test]
    fn visibility_labels_are_what_provenance_comments_print() {
        assert_eq!(Visibility::Public.label(), "pub");
        assert_eq!(Visibility::Private.label(), "private");
    }

    #[test]
    fn scanning_is_deterministic() {
        let source = "pub real a() {\n  return 1;\n}\nreal b() {\n  return 2;\n}\npub real c() {\n  return 3;\n}\n";
        assert_eq!(items_of(source), items_of(source));
        assert_eq!(public_names(source), vec!["a", "c"]);
    }
}
