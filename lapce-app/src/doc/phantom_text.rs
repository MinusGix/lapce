use std::sync::Arc;

use floem::peniko::Color;
use lsp_types::DiagnosticSeverity;
use smallvec::SmallVec;

/// `PhantomText` is for text that is not in the actual document, but should be rendered with it.  
/// Ex: Inlay hints, IME text, error lens' diagnostics, etc
#[derive(Debug, Clone, PartialEq)]
pub struct PhantomText {
    /// The kind is currently used for sorting the phantom text on a line
    pub kind: PhantomTextKind,
    /// Column on the line that the phantom text should be displayed at
    pub col: usize,
    pub text: String,
    pub font_size: Option<usize>,
    // font_family: Option<FontFamily>,
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub under_line: Option<Color>,
}
impl PhantomText {
    /// The amount of lines beyond the expected `1` that the phantom text will take up
    pub fn extra_line_count(&self) -> usize {
        self.text.lines().count().saturating_sub(1)
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum PhantomTextKind {
    /// Input methods
    Ime,
    /// Completion lens
    Completion,
    /// Inlay hints supplied by an LSP/PSP (like type annotations)
    InlayHint,
    /// Error lens
    Diagnostic,
}

/// Information about the phantom text on a specific line.  
/// This has various utility functions for transforming a coordinate (typically a column) into the
/// resulting coordinate after the phantom text is combined with the line's real content.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PhantomTextLine {
    /// This uses a smallvec because most lines rarely have more than a couple phantom texts
    pub text: SmallVec<[PhantomText; 6]>,
    /// Maximum diagnostic severity, so that we can color the background as an error if there is a
    /// warning and error on the line. (For error lens)
    pub max_severity: Option<DiagnosticSeverity>,
}

impl PhantomTextLine {
    /// Translate a column position into the text into what it would be after combining
    pub fn col_at(&self, pre_col: usize) -> usize {
        let mut last = pre_col;
        for (col_shift, size, col, _) in self.offset_size_iter() {
            if pre_col >= col {
                last = pre_col + col_shift + size;
            }
        }

        last
    }

    /// Translate a column position into the text into what it would be after combining
    /// If before_cursor is false and the cursor is right at the start then it will stay there
    pub fn col_after(&self, pre_col: usize, before_cursor: bool) -> usize {
        let mut last = pre_col;
        for (col_shift, size, col, _) in self.offset_size_iter() {
            if pre_col > col || (pre_col == col && before_cursor) {
                last = pre_col + col_shift + size;
            }
        }

        last
    }

    /// Translate a column position into the position it would be before combining
    pub fn before_col(&self, col: usize) -> usize {
        let mut last = col;
        for (col_shift, size, hint_col, _) in self.offset_size_iter() {
            let shifted_start = hint_col + col_shift;
            let shifted_end = shifted_start + size;
            if col >= shifted_start {
                if col >= shifted_end {
                    last = col - col_shift - size;
                } else {
                    last = hint_col;
                }
            }
        }
        last
    }

    /// Insert the hints at their positions in the text
    pub fn combine_with_text(&self, text: String) -> String {
        let mut text = text;
        let mut col_shift = 0;

        for phantom in self.text.iter() {
            let location = phantom.col + col_shift;

            // Stop iterating if the location is bad
            if text.get(location..).is_none() {
                return text;
            }

            text.insert_str(location, &phantom.text);
            col_shift += phantom.text.len();
        }

        text
    }

    /// Iterator over (col_shift, size, phantom.col, phantom)
    /// Note that this only iterates over the ordered text, since those depend on the text for where
    /// they'll be positioned
    pub fn offset_size_iter(
        &self,
    ) -> impl Iterator<Item = (usize, usize, usize, &PhantomText)> + '_ {
        let mut col_shift = 0;

        self.text.iter().map(move |phantom| {
            let pre_col_shift = col_shift;
            col_shift += phantom.text.len();
            (
                pre_col_shift,
                col_shift - pre_col_shift,
                phantom.col,
                phantom,
            )
        })
    }

    /// Iterator over (col_shift, size, phantom.col, phantom)
    /// Note that this only iterates over the ordered text, since those depend on the text for where
    /// they'll be positioned
    pub fn offset_size_into_iter(self) -> PhantomOffsetSizeIntoIter {
        PhantomOffsetSizeIntoIter {
            iter: self.text.into_iter(),
            col_shift: 0,
        }
    }
}

pub struct PhantomOffsetSizeIntoIter {
    iter: smallvec::IntoIter<[PhantomText; 6]>,
    col_shift: usize,
}
impl Iterator for PhantomOffsetSizeIntoIter {
    type Item = (usize, usize, usize, PhantomText);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|phantom| {
            let pre_col_shift = self.col_shift;
            self.col_shift += phantom.text.len();
            (
                pre_col_shift,
                self.col_shift - pre_col_shift,
                phantom.col,
                phantom,
            )
        })
    }
}

pub trait PhantomTextProvider {
    fn phantom_text(&self, line: usize) -> Arc<PhantomTextLine>;
}

#[cfg(test)]
impl PhantomTextProvider for std::collections::HashMap<usize, PhantomTextLine> {
    fn phantom_text(&self, line: usize) -> Arc<PhantomTextLine> {
        // We wrap this in an Arc here to make the tests simpler to write, since the tests
        // are cheap.
        Arc::new(
            self.get(&line)
                .cloned()
                .unwrap_or_else(PhantomTextLine::default),
        )
    }
}
