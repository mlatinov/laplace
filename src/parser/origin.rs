//! Mapping a byte offset in *transformed* source text back to the file and
//! line it was written on.
//!
//! laplace rewrites library sources before emitting them: a `.laplacelib`
//! file loses its `library { }` block, any `functions { }` wrapper, and its
//! `pub` markers, and a package's files are then concatenated into one
//! body. Provenance comments and error messages still have to name the file
//! and line a human would open, so every transformation that drops bytes
//! records what survived.
//!
//! Every transformation here is a *cut*: whole byte ranges are removed and
//! nothing is inserted or reordered. That makes the mapping a list of
//! verbatim segments, and a line number inside a segment is just the
//! segment's starting line plus the newlines seen since -- no need to keep
//! the original text around.

use std::ops::Range;

/// A span of text copied verbatim out of one source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineSegment {
    /// Byte range within the transformed text.
    pub range: Range<usize>,
    /// 1-indexed line, in the original file, that `range.start` came from.
    pub original_line: usize,
    /// Byte offset, in the original file, that `range.start` came from.
    /// Lets a transformation be applied in two passes: the first pass's
    /// result is measured, mapped back here, and re-cut from the original.
    pub original_offset: usize,
}

/// What [`apply_cuts`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutResult {
    /// The surviving text, cuts removed.
    pub text: String,
    /// Where each surviving run of bytes came from, in order.
    pub segments: Vec<LineSegment>,
    /// Where each requested marker offset ended up in `text`, in the order
    /// the markers were given. A marker inside a cut maps to the start of
    /// the text that follows the cut.
    pub markers: Vec<usize>,
}

/// Remove `cuts` from `source`.
///
/// Cuts may arrive in any order and may overlap; they are normalized
/// first. `markers` are byte offsets in `source` whose position in the
/// result the caller needs to know -- used to find where a stripped `pub`
/// keyword's item begins.
pub fn apply_cuts(source: &str, cuts: &[Range<usize>], markers: &[usize]) -> CutResult {
    let merged = merge_ranges(cuts, source.len());

    let mut text = String::with_capacity(source.len());
    let mut segments: Vec<LineSegment> = Vec::new();
    let mut marker_positions = vec![0usize; markers.len()];

    // Line number, in `source`, of the byte at `cursor`.
    let mut cursor = 0usize;
    let mut line = 1usize;

    let copy_to = |text: &mut String,
                   segments: &mut Vec<LineSegment>,
                   cursor: &mut usize,
                   line: &mut usize,
                   end: usize| {
        if end <= *cursor {
            return;
        }
        let chunk = &source[*cursor..end];
        let start = text.len();
        text.push_str(chunk);
        segments.push(LineSegment {
            range: start..text.len(),
            original_line: *line,
            original_offset: *cursor,
        });
        *line += chunk.bytes().filter(|&b| b == b'\n').count();
        *cursor = end;
    };

    // Markers are resolved as the copy sweeps past them, so they stay
    // correct however the cuts are arranged.
    let mut pending: Vec<(usize, usize)> = markers.iter().copied().enumerate().collect();
    pending.sort_by_key(|&(_, at)| at);
    let mut next_marker = 0usize;

    for cut in &merged {
        while next_marker < pending.len() && pending[next_marker].1 <= cut.start {
            let (idx, at) = pending[next_marker];
            copy_to(&mut text, &mut segments, &mut cursor, &mut line, at);
            marker_positions[idx] = text.len();
            next_marker += 1;
        }
        copy_to(&mut text, &mut segments, &mut cursor, &mut line, cut.start);
        // Skip the cut itself, counting the lines it swallowed so later
        // segments still report the right original line.
        line += source[cut.clone()].bytes().filter(|&b| b == b'\n').count();
        cursor = cut.end;
        // A marker *inside* this cut lands wherever the text resumes.
        while next_marker < pending.len() && pending[next_marker].1 < cut.end {
            marker_positions[pending[next_marker].0] = text.len();
            next_marker += 1;
        }
    }

    while next_marker < pending.len() {
        let (idx, at) = pending[next_marker];
        let at = at.min(source.len());
        copy_to(&mut text, &mut segments, &mut cursor, &mut line, at);
        marker_positions[idx] = text.len();
        next_marker += 1;
    }
    copy_to(
        &mut text,
        &mut segments,
        &mut cursor,
        &mut line,
        source.len(),
    );

    CutResult {
        text,
        segments,
        markers: marker_positions,
    }
}

impl CutResult {
    /// The offset in the original source that `offset` in [`CutResult::text`]
    /// came from. The end of the text maps to the end of the segment it
    /// closes, so a half-open range can be mapped back whole.
    pub fn to_original(&self, offset: usize) -> Option<usize> {
        let segment = self
            .segments
            .iter()
            .find(|s| s.range.contains(&offset) || s.range.end == offset)?;
        Some(segment.original_offset + (offset - segment.range.start))
    }
}

/// Sort, clamp and coalesce overlapping or touching ranges.
fn merge_ranges(ranges: &[Range<usize>], len: usize) -> Vec<Range<usize>> {
    let mut sorted: Vec<Range<usize>> = ranges
        .iter()
        .map(|r| r.start.min(len)..r.end.min(len))
        .filter(|r| r.start < r.end)
        .collect();
    sorted.sort_by_key(|r| (r.start, r.end));

    let mut merged: Vec<Range<usize>> = Vec::with_capacity(sorted.len());
    for range in sorted {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

/// One source file's contribution to a package's concatenated body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOrigin {
    /// File name relative to the package directory, as a reader would type
    /// it -- never an absolute path, which would vary by machine and break
    /// build determinism.
    pub file: String,
    /// The bytes of the package body this file contributed.
    pub body_range: Range<usize>,
    /// Segments, relative to `body_range.start`.
    pub segments: Vec<LineSegment>,
}

/// Where a byte in a package body was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location<'a> {
    pub file: &'a str,
    /// 1-indexed line in `file`.
    pub line: usize,
}

/// Every file that went into one package's concatenated source body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackageOrigin {
    pub files: Vec<FileOrigin>,
}

impl PackageOrigin {
    /// A package whose whole body is one verbatim file -- a plain `.stan`
    /// package file, and the shape test fixtures use.
    pub fn single_file(file: impl Into<String>, len: usize) -> Self {
        PackageOrigin {
            files: vec![FileOrigin {
                file: file.into(),
                body_range: 0..len,
                segments: vec![LineSegment {
                    range: 0..len,
                    original_line: 1,
                    original_offset: 0,
                }],
            }],
        }
    }

    /// Which file and line the byte at `offset` of `body` was written on.
    ///
    /// `body` must be the same concatenated text the origins were built
    /// from. `None` for an offset in no file's range, or in a gap between
    /// segments (the separator newlines the concatenation adds).
    pub fn locate<'a>(&'a self, body: &str, offset: usize) -> Option<Location<'a>> {
        let file = self.files.iter().find(|f| f.body_range.contains(&offset))?;
        let relative = offset - file.body_range.start;
        let segment = file.segments.iter().find(|s| s.range.contains(&relative))?;

        let from = file.body_range.start + segment.range.start;
        let seen = body
            .get(from..offset)
            .map(|text| text.bytes().filter(|&b| b == b'\n').count())
            .unwrap_or(0);
        Some(Location {
            file: &file.file,
            line: segment.original_line + seen,
        })
    }
}

/// 1-indexed line and column of `offset` in `text`, for an error message.
/// The column counts characters, not bytes, so it lines up with what an
/// editor shows.
pub fn line_col(text: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(text.len());
    let line_start = text[..offset].rfind('\n').map_or(0, |i| i + 1);
    let line = text[..line_start].bytes().filter(|&b| b == b'\n').count() + 1;
    let column = text[line_start..offset].chars().count() + 1;
    (line, column)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_of(result: &CutResult, offset: usize) -> usize {
        let origin = PackageOrigin {
            files: vec![FileOrigin {
                file: "f.laplacelib".to_string(),
                body_range: 0..result.text.len(),
                segments: result.segments.clone(),
            }],
        };
        origin.locate(&result.text, offset).unwrap().line
    }

    #[test]
    fn no_cuts_copies_everything_and_keeps_line_numbers() {
        let source = "a\nb\nc\n";
        let result = apply_cuts(source, &[], &[]);
        assert_eq!(result.text, source);
        assert_eq!(line_of(&result, 0), 1);
        assert_eq!(line_of(&result, 2), 2);
        assert_eq!(line_of(&result, 4), 3);
    }

    #[test]
    fn a_cut_shifts_following_lines_back_to_their_original_numbers() {
        // Line 1-3 cut away: what used to be line 4 is now line 1 of the
        // text, but must still report as line 4.
        let source = "library {\n  import stats\n}\nreal f() {\n  return 1;\n}\n";
        let cut = 0..source.find("real").unwrap();
        let result = apply_cuts(source, &[cut], &[]);
        assert_eq!(result.text, "real f() {\n  return 1;\n}\n");
        assert_eq!(line_of(&result, 0), 4);
        assert_eq!(line_of(&result, result.text.find("return").unwrap()), 5);
    }

    #[test]
    fn several_cuts_each_shift_what_follows() {
        let source = "1\n2\n3\n4\n5\n6\n";
        // Remove "2\n" and "4\n".
        let result = apply_cuts(source, &[2..4, 6..8], &[]);
        assert_eq!(result.text, "1\n3\n5\n6\n");
        assert_eq!(line_of(&result, 0), 1);
        assert_eq!(line_of(&result, 2), 3);
        assert_eq!(line_of(&result, 4), 5);
        assert_eq!(line_of(&result, 6), 6);
    }

    #[test]
    fn a_marker_reports_where_the_text_after_it_landed() {
        let source = "pub real f() {\n  return 1;\n}\n";
        // Cut "pub " and ask where what followed it ended up.
        let cut = 0..4;
        let result = apply_cuts(source, std::slice::from_ref(&cut), &[4]);
        assert_eq!(result.text, "real f() {\n  return 1;\n}\n");
        assert_eq!(result.markers, vec![0]);
    }

    #[test]
    fn a_marker_inside_a_cut_lands_where_the_text_resumes() {
        let source = "xxpub real f() {}\n";
        let cut = 2..6;
        let result = apply_cuts(source, std::slice::from_ref(&cut), &[3]);
        assert_eq!(result.text, "xxreal f() {}\n");
        assert_eq!(result.markers, vec![2]);
    }

    #[test]
    fn markers_are_returned_in_the_order_they_were_given() {
        let source = "aXbXc";
        let result = apply_cuts(source, &[1..2, 3..4], &[4, 2]);
        assert_eq!(result.text, "abc");
        // `4` ("c") is index 2 in the output; `2` ("b") is index 1.
        assert_eq!(result.markers, vec![2, 1]);
    }

    #[test]
    fn overlapping_and_unsorted_cuts_are_normalized() {
        let source = "abcdefgh";
        let result = apply_cuts(source, &[4..6, 1..3, 2..5], &[]);
        assert_eq!(result.text, "agh");
    }

    #[test]
    fn a_cut_past_the_end_is_clamped() {
        let source = "abc";
        let cut = 2..99;
        let result = apply_cuts(source, std::slice::from_ref(&cut), &[]);
        assert_eq!(result.text, "ab");
    }

    #[test]
    fn to_original_maps_the_result_back_so_a_second_pass_can_re_cut() {
        let source = "library {\n}\n\nreal f() {\n  return 1;\n}\n";
        let cut = 0..source.find("\n\nreal").unwrap() + 1;
        let first = apply_cuts(source, std::slice::from_ref(&cut), &[]);
        assert_eq!(first.text, "\nreal f() {\n  return 1;\n}\n");
        // The blank first line of the result, mapped back, is the blank
        // line the cut stopped at in the original.
        let back = first.to_original(1).unwrap();
        assert_eq!(&source[back..back + 4], "real");
    }

    #[test]
    fn locate_finds_the_right_file_in_a_concatenated_package_body() {
        // Two files concatenated, each followed by a separator newline.
        let first = "real a() {\n  return 1;\n}\n";
        let second = "real b() {\n  return 2;\n}\n";
        let body = format!("{first}\n{second}\n");

        let origin = PackageOrigin {
            files: vec![
                FileOrigin {
                    file: "a.stan".to_string(),
                    body_range: 0..first.len(),
                    segments: vec![LineSegment {
                        range: 0..first.len(),
                        original_line: 1,
                        original_offset: 0,
                    }],
                },
                FileOrigin {
                    file: "b.stan".to_string(),
                    body_range: first.len() + 1..first.len() + 1 + second.len(),
                    segments: vec![LineSegment {
                        range: 0..second.len(),
                        original_line: 1,
                        original_offset: 0,
                    }],
                },
            ],
        };

        let a = origin.locate(&body, 0).unwrap();
        assert_eq!((a.file, a.line), ("a.stan", 1));

        let b_at = body.find("real b").unwrap();
        let b = origin.locate(&body, b_at).unwrap();
        assert_eq!((b.file, b.line), ("b.stan", 1));

        let b_return = body.find("return 2").unwrap();
        assert_eq!(origin.locate(&body, b_return).unwrap().line, 2);
    }

    #[test]
    fn locate_returns_none_for_a_separator_between_files() {
        let origin = PackageOrigin {
            files: vec![FileOrigin {
                file: "a.stan".to_string(),
                body_range: 0..3,
                segments: vec![LineSegment {
                    range: 0..3,
                    original_line: 1,
                    original_offset: 0,
                }],
            }],
        };
        assert_eq!(origin.locate("abc\n", 3), None);
    }

    #[test]
    fn line_col_counts_from_one() {
        let text = "ab\ncde\n";
        assert_eq!(line_col(text, 0), (1, 1));
        assert_eq!(line_col(text, 1), (1, 2));
        assert_eq!(line_col(text, 3), (2, 1));
        assert_eq!(line_col(text, 5), (2, 3));
    }

    #[test]
    fn line_col_columns_count_characters_not_bytes() {
        let text = "// é marks the spot\nreal f() {}\n";
        let at = text.find("marks").unwrap();
        // 'é' is two bytes but one column.
        assert_eq!(line_col(text, at), (1, 6));
    }
}
