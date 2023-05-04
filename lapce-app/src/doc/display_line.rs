use std::{borrow::Cow, ops::Range};

use lapce_core::{buffer::Buffer, word::WordCursor};
use smallvec::SmallVec;

use crate::doc::phantom_text::PhantomText;

use super::phantom_text::{PhantomTextLine, PhantomTextProvider};

/// A line within the display of the editor, rather than an actual line in the buffer.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DisplayLine(usize);
impl DisplayLine {
    /// Construct a display line without checking that it is a valid display line.  
    /// Though, this is the primary way to construct [`DisplayLine`]s manually.
    pub fn new_unchecked(line: usize) -> Self {
        Self(line)
    }

    pub fn get(self) -> usize {
        self.0
    }

    /// Iterator over (display line start, extra line count (that was added to display line start),
    /// real line)
    fn display_line_iter<'a>(
        phantom_prov: &'a impl PhantomTextProvider,
        buffer: &Buffer,
    ) -> impl Iterator<Item = (DisplayLine, usize, usize)> + 'a {
        let mut acc_line = 0;
        (0..buffer.num_lines()).flat_map(move |line| {
            if line != 0 {
                acc_line += 1;
            }

            let phantom = phantom_prov.phantom_text(line);

            let phantom_line_count: usize = phantom
                .offset_size_iter()
                .map(|(_, _, _, t)| t.extra_line_count())
                .sum();

            let p_iter = phantom.offset_size_iter();

            let initial = (DisplayLine::new_unchecked(acc_line), 0, line);

            let mut cur_acc = acc_line;
            // Tracks the number of lines, which is needed for ensuring that the line shift from
            // real line is valid. Ex: if the first phantom text has one line, and the second has
            // two, then we need that for the `x + accum_line_count` below to ensure it includes
            // the first phantom text's phantom line.
            let mut accum_line_count = 0;
            let p_iter = p_iter
                .filter_map(|(_, _, _, text)| {
                    let line_count = text.extra_line_count();

                    if line_count == 0 {
                        None
                    } else {
                        Some(line_count)
                    }
                })
                .flat_map(move |line_count| {
                    let res = (1..=line_count).map(move |x| {
                        (
                            DisplayLine::new_unchecked(cur_acc + x),
                            x + accum_line_count,
                            line,
                        )
                    });

                    cur_acc += line_count;
                    accum_line_count += line_count;

                    res
                });

            let res = std::iter::once(initial).chain(p_iter);

            acc_line += phantom_line_count;

            res
        })
    }

    /// Convert an offset into the relevant (display line, display column).  
    pub fn display_line_col_of_offset(
        phantom_prov: &impl PhantomTextProvider,
        buffer: &Buffer,
        offset: usize,
    ) -> (DisplayLine, usize) {
        let (line, col) = buffer.offset_to_line_col(offset);
        Self::display_line_col_of_line_col(phantom_prov, buffer, line, col)
    }

    /// Convert a (line, col) in the buffer into the relevant (display line, display column).  
    /// Note that if you already have a [`DisplayLineInfo`] instance, you should use
    /// [`DisplayLineInfo::col_to_display_col`] instead, as we just call out to that.
    pub fn display_line_col_of_line_col(
        phantom_prov: &impl PhantomTextProvider,
        buffer: &Buffer,
        line: usize,
        col: usize,
    ) -> (DisplayLine, usize) {
        // TODO: couldn't display line iter provide decent column info?
        let iter = Self::display_line_iter(phantom_prov, buffer)
            .filter(|(_, _, real_line)| *real_line == line);
        let mut last_dline = None;
        for (dline, _, _) in iter {
            let info = DisplayLineInfo::new(phantom_prov, buffer, dline);
            let dcol = info.col_to_display_col(col);

            last_dline = Some((dline, info));

            if let Some(dcol) = dcol {
                return (dline, dcol);
            }
        }

        if let Some((last_dline, info)) = last_dline {
            // If we found a display line but the column was too far, then we give the last dcolumn
            return (last_dline, info.max_display_col());
        } else {
            // Otherwise, we didn't find any display lines at all, so we're probably past the end
            // so we return the last display line and column.
            let last = Self::last_display_line(phantom_prov, buffer);
            let info = DisplayLineInfo::new(phantom_prov, buffer, last);

            (last, info.max_display_col())
        }
    }

    pub fn last_display_line(
        phantom_prov: &impl PhantomTextProvider,
        buffer: &Buffer,
    ) -> Self {
        Self::display_line_iter(phantom_prov, buffer)
            .last()
            .map(|(a, _, _)| a)
            .unwrap_or_else(|| Self::new_unchecked(0))
    }

    /// Returns the (real line, line shift) for this display line.  
    /// The line shift is how many lines it is away from that real line.
    pub fn find_real_line(
        self,
        phantom_prov: &impl PhantomTextProvider,
        buffer: &Buffer,
    ) -> (usize, usize) {
        let v = DisplayLine::display_line_iter(phantom_prov, buffer)
            .find(|(l, _, _)| *l == self)
            .map(|(_, line_shift, real_line)| (real_line, line_shift));

        if let Some(v) = v {
            return v;
        }

        // Otherwise fall back to the last line / line shift
        let last = Self::last_display_line(phantom_prov, buffer);
        if last == self {
            unreachable!("find_real_line tried 'falling back' to the last display line, but ending up appearing to recurse forever. This is a bug.");
        }

        last.find_real_line(phantom_prov, buffer)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DisplayLineInfo {
    /// The real line that this is related to.  
    /// For phantom text, this means the phantom text exists on this line.  
    /// For linewrapping, this means it started on this line, but then potentially wrapped.
    pub line: usize,
    /// The shifted amount away from the real line.  
    /// Ex: A multiline phantom text would have `0` for the first line, and then `1` for the line
    /// after the newline.  
    /// Ex: A linewrapped line would have `0` for the first line, and then `1` for the wrapped line.
    pub line_shift: usize,
    /// The columns of the underlying real line that are on this display line.
    /// The phantom text is applied on top of this.
    pub range: Range<usize>,
    pub phantom: PhantomTextLine,
}
impl DisplayLineInfo {
    pub fn new(
        phantom_prov: &impl PhantomTextProvider,
        buffer: &Buffer,
        dline: DisplayLine,
    ) -> DisplayLineInfo {
        let (line, line_shift) = dline.find_real_line(phantom_prov, buffer);
        let phantom = phantom_prov.phantom_text(line);

        // The phantom text is for the entire buffer line. This means that we need to find the
        // pieces that are on our display line (aka `line_shift` amount of lines in).
        // Then cut out the pieces that aren't on the display line. We'll have to shift the columns
        // around.

        let mut cur_line_shift = 0;
        let mut start_col: Option<usize> = None;
        let mut end_col: Option<usize> = None;
        let mut first_col: Option<usize> = None;

        let mut new_phantoms: SmallVec<[PhantomText; 6]> = SmallVec::new();

        for (_, size, col, text) in phantom.offset_size_iter() {
            let line_count = text.extra_line_count();

            if cur_line_shift > line_shift {
                // We're past the display line. Just stop here.
                break;
            } else if cur_line_shift + line_count < line_shift {
                // We're not on the display line yet.
                cur_line_shift += line_count;
                continue;
            }

            // We're on the display line, and/or partially past it.

            // The number of lines starting from the display line
            let line_ontop = cur_line_shift + line_count - line_shift;

            // We need to get the index into the text that is on the display line.
            let line_idx = line_shift - cur_line_shift;

            let line_text = text.text.lines().nth(line_idx).unwrap();

            let new_col = if let Some(start_col) = start_col {
                // We have a start col so this is not the first phantom text on the display line.
                // If the phantom text has a newline, then we can set the end_col to the col of the
                // phantom text.
                if line_count != 0 {
                    end_col = Some(col);
                }

                // We have a start col, so we need to shift the col by that amount so that it
                // is positioned properly on this display line.
                col - start_col
            } else {
                // There was no start col set. This means this is the first phantom text on the
                // display line. So it simply starts at zero.
                // We store the col that the phantom text started in the real line, so we can
                // shift future phantom text's columns.
                // This also serves as the start in the real line for the relevant text.
                let res = if line_shift == 0 {
                    // This is the first line, so we do not shift.
                    start_col = Some(0);
                    col
                } else {
                    // This is not the first line, so we need to store the start col to shift by.
                    start_col = Some(col);
                    0
                };

                if line_shift == 0 && line_count != 0 {
                    end_col = Some(col);
                } else if line_shift == 0 || line_ontop == 0 {
                    // this is the first phantom line but there can be more phantom text
                    // so we can't deduce the end col yet. See the `if` above for the logic of this.
                } else {
                    // this is the first and last phantom line since it goes for multiple lines
                    // so we can deduce the end col. It is just start_col, which gives us an empty
                    // range.
                    end_col = start_col;
                }

                res
            };

            let mut new_text = text.clone();
            new_text.col = new_col;
            new_text.text = line_text.to_string();

            new_phantoms.push(new_text);

            cur_line_shift += line_count;
        }

        let new_phantom = PhantomTextLine {
            text: new_phantoms,
            max_severity: phantom.max_severity,
        };

        if end_col.is_none() {
            // TODO: we shouldn't need to get the whole line content. We just need the last few chars, so we could just use the offsets. That would avoid any potential allocations.
            let underlying_line = buffer.line_content(line);
            let end = underlying_line.len();
            end_col = if underlying_line.ends_with("\r\n") {
                Some(end - 2)
            } else if underlying_line.ends_with("\n") {
                Some(end - 1)
            } else {
                Some(end)
            };
        }

        DisplayLineInfo {
            line,
            line_shift,
            range: start_col.unwrap_or(0)..end_col.unwrap(),
            phantom: new_phantom,
        }
    }

    // TODO: test to see if this results in the expected value. Might need before_cursor to be an arg
    pub fn max_display_col(&self) -> usize {
        let range_len = self.range.end - self.range.start;
        self.phantom.col_after(range_len, false)
    }

    /// Get the maximum *buffer* column on the display line.
    pub fn max_col(&self) -> usize {
        self.range.end
    }

    /// Get the display column of the end of the line.
    pub fn line_end_display_col(&self, caret: bool) -> usize {
        // TODO: handle caret!
        self.max_display_col()
    }

    /// Convert a col in the buffer for this line into the relevant display column.
    pub fn col_to_display_col(&self, col: usize) -> Option<usize> {
        if col < self.range.start {
            // The col is before the display line, so it is not visible.
            None
        } else if col >= self.range.end {
            // The col is after the display line, so it is not visible.
            None
        } else {
            // The col is on the display line, so we can convert it.
            let shifted_col = col - self.range.start;
            Some(self.phantom.col_at(shifted_col))
        }
    }

    /// Convert a col in the buffer for this line into the relevant display column.
    /// If the col is not on the display line, then it is clamped to the start or end of the
    /// display line.
    pub fn col_to_display_col_clamp(&self, col: usize) -> usize {
        if col < self.range.start {
            // The col is before the display line, so it is not visible.
            self.phantom.col_at(0)
        } else if col >= self.range.end {
            // The col is after the display line, so it is not visible.
            self.phantom.col_at(self.range.end - self.range.start)
        } else {
            // The col is on the display line, so we can convert it.
            let shifted_col = col - self.range.start;
            self.phantom.col_at(shifted_col)
        }
    }

    /// Convert a display col to a buffer col (in the real line).
    pub fn display_col_to_col(&self, col: usize) -> usize {
        let real_col = self.phantom.before_col(col);
        self.range.start + real_col
    }

    /// Get the display line content merged with the phantom text.  
    /// If `soften_newlines` is true, then we replace `\r\n` or `\n` by spaces for each whitespace.
    pub fn line_content(&self, buffer: &Buffer, soften_newlines: bool) -> String {
        let line_content = buffer.line_content(self.line);
        let line_content = &line_content[self.range.clone()];

        let line_content = if soften_newlines {
            if let Some(s) = line_content.strip_suffix("\r\n") {
                Cow::Owned(format!("{s}  "))
            } else if let Some(s) = line_content.strip_suffix("\n") {
                Cow::Owned(format!("{s} "))
            } else {
                Cow::Borrowed(line_content)
            }
        } else {
            Cow::Borrowed(line_content)
        };

        // TODO: if phantom was empty then this could just return a Cow<str> in some cases
        // TODO: combine_with_text could allow Cow<str> to be given
        self.phantom.combine_with_text(line_content.to_string())
    }

    // TODO: test
    /// Get the *buffer* column of the first non-blank character on the actual line.
    pub fn first_non_blank_character_on_line(&self, buffer: &Buffer) -> usize {
        let start_offset = buffer.offset_of_line(self.line) + self.range.start;
        WordCursor::new(buffer.text(), start_offset).next_non_blank_char()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use smallvec::smallvec;

    use crate::doc::phantom_text::{PhantomText, PhantomTextKind};

    use super::*;

    // We use various functions to make some tests simpler/less repetitive

    struct Bundle {
        buffer: Buffer,
        phantom_prov: HashMap<usize, PhantomTextLine>,
    }
    impl Bundle {
        /// Test the output of [`DisplayLine::display_line_iter`] against the expected
        /// (display line, line_shift, real line), with the display line being automatically
        /// constructed.
        #[track_caller]
        fn test_display_line_iter(&self, lines: &[(usize, usize, usize)]) {
            let mut iter =
                DisplayLine::display_line_iter(&self.phantom_prov, &self.buffer);
            let lines = lines
                .iter()
                .copied()
                .map(|(l, s, r)| (DisplayLine::new_unchecked(l), s, r));

            for expected in lines {
                assert_eq!(iter.next(), Some(expected));
            }

            assert_eq!(iter.next(), None, "Expected last entry to be None");
        }

        /// Assert that the last line is equal to the given dline.
        #[track_caller]
        fn assert_last_line_eq(&self, dline: usize) {
            assert_eq!(
                DisplayLine::last_display_line(&self.phantom_prov, &self.buffer),
                DisplayLine::new_unchecked(dline)
            );
        }

        #[track_caller]
        fn assert_real_line(&self, dline: usize, exp: (usize, usize)) {
            assert_eq!(
                DisplayLine::new_unchecked(dline)
                    .find_real_line(&self.phantom_prov, &self.buffer),
                exp
            );
        }

        #[track_caller]
        fn assert_line_col(
            &self,
            line: usize,
            col: usize,
            (exp_dline, exp_dcol): (usize, usize),
        ) {
            assert_eq!(
                DisplayLine::display_line_col_of_line_col(
                    &self.phantom_prov,
                    &self.buffer,
                    line,
                    col
                ),
                (DisplayLine::new_unchecked(exp_dline), exp_dcol),
                "line: {line}, col: {col} did not convert into display line: {exp_dline} display col: {exp_dcol}"
            );
        }

        #[track_caller]
        fn assert_line_cols(&self, v: &[(usize, usize, (usize, usize))]) {
            for (line, col, v) in v.iter().copied() {
                self.assert_line_col(line, col, v);
            }
        }

        /// Get the display line info for a display line.
        fn line_info(&self, dline: usize) -> DisplayLineInfo {
            DisplayLineInfo::new(
                &self.phantom_prov,
                &self.buffer,
                DisplayLine::new_unchecked(dline),
            )
        }

        /// Get the phantom text for a specific line and combine it with the buffer text.
        fn combined(&self, line: usize) -> String {
            let text = self.buffer.line_content(line);
            let text = text.strip_suffix("\n").unwrap_or(&text);
            self.phantom_prov
                .get(&line)
                .unwrap()
                .combine_with_text(text.to_string())
        }
    }

    #[test]
    fn test_basic_display_line_info() {
        let bun = Bundle {
            buffer: Buffer::new("test thing"),
            phantom_prov: HashMap::new(),
        };

        bun.test_display_line_iter(&[(0, 0, 0)]);

        bun.assert_last_line_eq(0);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("text thing".len()),
                phantom: PhantomTextLine::default()
            }
        );
        // aka out of bounds is last line
        assert_eq!(
            bun.line_info(1),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("text thing".len()),
                phantom: PhantomTextLine::default()
            }
        );

        let basic_lc = &[
            (0, 0, (0, 0)),   // t
            (0, 1, (0, 1)),   // e
            (0, 2, (0, 2)),   // s
            (0, 3, (0, 3)),   // t
            (0, 7, (0, 7)),   // i
            (0, 8, (0, 8)),   // n
            (0, 9, (0, 9)),   // g
            (0, 10, (0, 10)), // at the end
            (0, 11, (0, 10)), // past the end
            (0, 12, (0, 10)),
        ];
        // These are equivalent to the normal buffer line col
        bun.assert_line_cols(basic_lc);

        let bun = Bundle {
            buffer: Buffer::new("test thing\nhi"),
            phantom_prov: HashMap::new(),
        };

        bun.test_display_line_iter(&[(0, 0, 0), (1, 0, 1)]);
        bun.assert_last_line_eq(1);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("text thing".len()),
                phantom: PhantomTextLine::default()
            }
        );
        assert_eq!(
            bun.line_info(1),
            DisplayLineInfo {
                line: 1,
                line_shift: 0,
                range: 0..("hi".len()),
                phantom: PhantomTextLine::default()
            }
        );
        // aka out of bounds is last line
        assert_eq!(bun.line_info(2), bun.line_info(1));

        bun.assert_line_cols(basic_lc);
        bun.assert_line_cols(&[
            (1, 0, (1, 0)), // h
            (1, 1, (1, 1)), // i
            (1, 2, (1, 2)), // at the end
            (1, 3, (1, 2)), // past the end
            (1, 4, (1, 2)),
        ]);

        let bun = Bundle {
            buffer: Buffer::new("test thing\nhi\nhello"),
            phantom_prov: HashMap::new(),
        };

        bun.test_display_line_iter(&[(0, 0, 0), (1, 0, 1), (2, 0, 2)]);

        bun.assert_last_line_eq(2);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("text thing".len()),
                phantom: PhantomTextLine::default()
            }
        );
        assert_eq!(
            bun.line_info(1),
            DisplayLineInfo {
                line: 1,
                line_shift: 0,
                range: 0..("hi".len()),
                phantom: PhantomTextLine::default()
            }
        );
        assert_eq!(
            bun.line_info(2),
            DisplayLineInfo {
                line: 2,
                line_shift: 0,
                range: 0..("hello".len()),
                phantom: PhantomTextLine::default()
            }
        );
        // aka out of bounds is last line
        assert_eq!(bun.line_info(3), bun.line_info(2));

        bun.assert_line_cols(basic_lc);
        bun.assert_line_cols(&[
            (1, 0, (1, 0)), // h
            (1, 1, (1, 1)), // i
            (1, 2, (1, 2)), // at the end
            (1, 3, (1, 2)), // past the end
            (1, 4, (1, 2)),
            (2, 0, (2, 0)), // h
            (2, 1, (2, 1)), // e
        ]);
    }

    // TODO: remember to test phantom text fg/bg to make sure they're kept
    // can just add those as random values, really.
    #[test]
    fn test_phantom_display_line_info() {
        let mut bun = Bundle {
            buffer: Buffer::new("test thing"),
            phantom_prov: HashMap::new(),
        };
        let p = PhantomTextLine {
            text: smallvec![PhantomText {
                kind: PhantomTextKind::InlayHint,
                col: 3,
                text: "blah".to_string(),
                font_size: Some(12),
                fg: None,
                bg: None,
                under_line: None
            }],
            max_severity: None,
        };
        bun.phantom_prov.insert(0, p.clone());

        assert_eq!(bun.combined(0), "tesblaht thing");

        bun.test_display_line_iter(&[(0, 0, 0)]);

        bun.assert_last_line_eq(0);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("test thing".len()),
                phantom: p.clone()
            }
        );
        // aka out of bounds is last line
        assert_eq!(bun.line_info(1), bun.line_info(0));

        bun.assert_line_cols(&[
            (0, 0, (0, 0)),   // t
            (0, 1, (0, 1)),   // e
            (0, 2, (0, 2)),   // s
            (0, 3, (0, 7)), // t which gets shifted forward by the 'blah' phantom text
            (0, 4, (0, 8)), // space
            (0, 5, (0, 9)), // t
            (0, 6, (0, 10)), // h
            (0, 7, (0, 11)), // i
            (0, 8, (0, 12)), // n
            (0, 9, (0, 13)), // g
            (0, 10, (0, 14)), // end
            (0, 11, (0, 14)), // saturate
            (0, 12, (0, 14)), // saturate
            // test weird extremes
            (0, 15, (0, 14)), // saturate
            (0, 16, (0, 14)), // saturate
        ]);

        let mut bun = Bundle {
            buffer: Buffer::new("test thing"),
            phantom_prov: HashMap::new(),
        };

        bun.phantom_prov.insert(
            0,
            PhantomTextLine {
                text: smallvec![PhantomText {
                    kind: PhantomTextKind::InlayHint,
                    col: 3,
                    text: "blah\naa".to_string(),
                    font_size: Some(12),
                    fg: None,
                    bg: None,
                    under_line: None
                }],
                max_severity: None,
            },
        );

        assert_eq!(bun.combined(0), "tesblah\naat thing");

        bun.test_display_line_iter(&[(0, 0, 0), (1, 1, 0)]);

        bun.assert_last_line_eq(1);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("tes".len()),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 3,
                        text: "blah".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(1),
            DisplayLineInfo {
                line: 0,
                line_shift: 1,
                range: 3..("t thing".len() + 3),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 0,
                        text: "aa".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        // aka out of bounds is last line
        assert_eq!(bun.line_info(2), bun.line_info(1));

        bun.assert_line_cols(&[
            (0, 0, (0, 0)),  // t
            (0, 1, (0, 1)),  // e
            (0, 2, (0, 2)),  // s
            (0, 3, (1, 2)), // t, goes to next line due to phantom text, shifted by 'aa'
            (0, 4, (1, 3)), // space
            (0, 5, (1, 4)), // t
            (0, 6, (1, 5)), // h
            (0, 7, (1, 6)), // i
            (0, 8, (1, 7)), // n
            (0, 9, (1, 8)), // g
            (0, 10, (1, 9)), // end
            (0, 11, (1, 9)), // saturate
            (0, 12, (1, 9)), // saturate
        ]);

        let mut bun = Bundle {
            buffer: Buffer::new("test thing"),
            phantom_prov: HashMap::new(),
        };

        // multiple multiline phantoms
        bun.phantom_prov.insert(
            0,
            PhantomTextLine {
                text: smallvec![
                    PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 3,
                        text: "blah\naa".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    },
                    PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 7,
                        text: "t\nabc\nme".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }
                ],
                max_severity: None,
            },
        );

        assert_eq!(bun.combined(0), "tesblah\naat tht\nabc\nmeing");
        // roughly: "tes[blah\naa]t th[t\nabc\nme]ing"

        bun.test_display_line_iter(&[(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)]);

        bun.assert_last_line_eq(3);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("tes".len()),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 3,
                        text: "blah".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(1),
            DisplayLineInfo {
                line: 0,
                line_shift: 1,
                range: 3..("t th".len() + 3),
                phantom: PhantomTextLine {
                    text: smallvec![
                        PhantomText {
                            kind: PhantomTextKind::InlayHint,
                            col: 0,
                            text: "aa".to_string(),
                            font_size: Some(12),
                            fg: None,
                            bg: None,
                            under_line: None
                        },
                        PhantomText {
                            kind: PhantomTextKind::InlayHint,
                            col: 4,
                            text: "t".to_string(),
                            font_size: Some(12),
                            fg: None,
                            bg: None,
                            under_line: None
                        }
                    ],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(2),
            DisplayLineInfo {
                line: 0,
                line_shift: 2,
                range: 7..7, // completely phantom line
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 0,
                        text: "abc".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(3),
            DisplayLineInfo {
                line: 0,
                line_shift: 3,
                range: 7..("ing".len() + 7),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 0,
                        text: "me".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        // aka out of bounds is last line
        assert_eq!(bun.line_info(4), bun.line_info(3));

        bun.assert_line_cols(&[
            (0, 0, (0, 0)),  // t
            (0, 1, (0, 1)),  // e
            (0, 2, (0, 2)),  // s
            (0, 3, (1, 2)),  // t, shifted to next line and then forward by 'aa'
            (0, 4, (1, 3)),  // space
            (0, 5, (1, 4)),  // t
            (0, 6, (1, 5)),  // h
            (0, 7, (3, 2)),  // i, shifted two lines and then forward by 'me'
            (0, 8, (3, 3)),  // n
            (0, 9, (3, 4)),  // g
            (0, 10, (3, 5)), // end
            (0, 11, (3, 5)), // saturate
            (0, 12, (3, 5)), // saturate
        ]);

        let mut bun = Bundle {
            buffer: Buffer::new("test thing\nhi"),
            phantom_prov: HashMap::new(),
        };

        bun.phantom_prov.insert(
            0,
            PhantomTextLine {
                text: smallvec![PhantomText {
                    kind: PhantomTextKind::InlayHint,
                    col: 3,
                    text: "blah\nthing".to_string(),
                    font_size: Some(12),
                    fg: None,
                    bg: None,
                    under_line: None
                }],
                max_severity: None,
            },
        );

        // roughly: "tes[blah\nthing]t thing\nhi"

        bun.test_display_line_iter(&[(0, 0, 0), (1, 1, 0), (2, 0, 1)]);

        let mut bun = Bundle {
            buffer: Buffer::new("test thing\nhi"),
            phantom_prov: HashMap::new(),
        };

        bun.phantom_prov.insert(
            0,
            PhantomTextLine {
                text: smallvec![PhantomText {
                    kind: PhantomTextKind::InlayHint,
                    col: 3,
                    text: "blah\nthing".to_string(),
                    font_size: Some(12),
                    fg: None,
                    bg: None,
                    under_line: None
                }],
                max_severity: None,
            },
        );
        bun.phantom_prov.insert(
            1,
            PhantomTextLine {
                text: smallvec![PhantomText {
                    kind: PhantomTextKind::InlayHint,
                    col: 1,
                    text: "aa\n111".to_string(),
                    font_size: Some(12),
                    fg: None,
                    bg: None,
                    under_line: None
                }],
                max_severity: None,
            },
        );

        // roughly: "tes[blah\nthing]t thing\nh[aa\n111]i"
        assert_eq!(bun.combined(0), "tesblah\nthingt thing");
        assert_eq!(bun.combined(1), "haa\n111i");

        bun.test_display_line_iter(&[(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)]);

        bun.assert_last_line_eq(3);

        assert_eq!(
            bun.line_info(0),
            DisplayLineInfo {
                line: 0,
                line_shift: 0,
                range: 0..("tes".len()),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 3,
                        text: "blah".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(1),
            DisplayLineInfo {
                line: 0,
                line_shift: 1,
                range: 3..("t thing".len() + 3),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 0,
                        text: "thing".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    },],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(2),
            DisplayLineInfo {
                line: 1,
                line_shift: 0,
                range: 0..("h".len()),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 1,
                        text: "aa".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );
        assert_eq!(
            bun.line_info(3),
            DisplayLineInfo {
                line: 1,
                line_shift: 1,
                range: 1..("i".len() + 1),
                phantom: PhantomTextLine {
                    text: smallvec![PhantomText {
                        kind: PhantomTextKind::InlayHint,
                        col: 0,
                        text: "111".to_string(),
                        font_size: Some(12),
                        fg: None,
                        bg: None,
                        under_line: None
                    }],
                    max_severity: None,
                }
            }
        );

        bun.assert_line_cols(&[
            (0, 0, (0, 0)),   // t
            (0, 1, (0, 1)),   // e
            (0, 2, (0, 2)),   // s
            (0, 3, (1, 5)),   // t moved down a line, then shifted foward by 'thing'
            (0, 4, (1, 6)),   // space
            (0, 5, (1, 7)),   // t
            (0, 6, (1, 8)),   // h
            (0, 7, (1, 9)),   // i
            (0, 8, (1, 10)),  // n
            (0, 9, (1, 11)),  // g
            (0, 10, (1, 12)), // end of line
            (0, 11, (1, 12)), // saturate
            (0, 12, (1, 12)), // saturate
            (1, 0, (2, 0)),   // h goes to next (real) line
            (1, 1, (3, 3)),   // i move down a line, then shifted forward by '111'
            (1, 2, (3, 4)),   // end
            (1, 2, (3, 4)),   // saturate
        ]);
    }

    #[test]
    fn test_find_real_line() {
        let bun = Bundle {
            buffer: Buffer::new("test thing"),
            phantom_prov: HashMap::new(),
        };

        bun.assert_real_line(0, (0, 0));
        bun.assert_real_line(1, (0, 0));

        let mut bun = Bundle {
            buffer: Buffer::new("test thing\nhi"),
            phantom_prov: HashMap::new(),
        };

        bun.assert_real_line(0, (0, 0));
        bun.assert_real_line(1, (1, 0));
        bun.assert_real_line(2, (1, 0));

        bun.phantom_prov.insert(
            0,
            PhantomTextLine {
                text: smallvec![PhantomText {
                    kind: PhantomTextKind::InlayHint,
                    col: 3,
                    text: "blah\nthing".to_string(),
                    font_size: Some(12),
                    fg: None,
                    bg: None,
                    under_line: None
                }],
                max_severity: None,
            },
        );

        bun.assert_real_line(0, (0, 0));
        bun.assert_real_line(1, (0, 1));
        bun.assert_real_line(2, (1, 0));
        bun.assert_real_line(3, (1, 0));
    }
}
