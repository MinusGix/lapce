// Use `Breaks` for phantom text line breaks, there'd just be multiple at one position which should be fine.
// We can have DisplayLine be more of a (real line, line shift) pair, rather than a single usize since that makes it easier to update it. We'd just supply an iterator over them for things that need that.
// And textlayoutcache can just have a SmallVec<[TextLayout; 2] or something.
// This make it nicer to clear cached lines and avoid issues of 'is this displayline old or not?'

// This is based on xi-editor's line-wrap logic

use std::{cmp::Ordering, ops::Range};

use lapce_core::buffer::InvalLines;
use lapce_xi_rope::{
    breaks::{BreakBuilder, Breaks, BreaksInfo, BreaksMetric},
    Cursor, Interval, LinesMetric, Rope, RopeDelta, RopeInfo,
};

pub trait WidthCalc {}

/// The visual width of the buffer for the purpose of word wrapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WrapWidth {
    /// No wrapping in effect.
    None,

    /// Width in bytes (utf-8 code units).
    ///
    /// Only works well for ASCII, will probably not be maintained long-term.
    Bytes(usize),

    /// Width in px units, requiring measurement by the front-end.
    Width(f64),
    // TODO: setting for whether phantom text should be included in wrap length logic
    // (ex: if you're just fitting a style guide, like limit of 100 chars per line, then bytes works and shouldn't count phantom text; but if you're wanting it so that the text always fits on your screen, then you want to include phantom text)
    // TODO: some users would want the width to automatically fit the editor size, though that should be managed outside of this code.
}

impl Default for WrapWidth {
    fn default() -> Self {
        WrapWidth::None
    }
}

impl WrapWidth {
    fn differs_in_kind(self, other: WrapWidth) -> bool {
        use self::WrapWidth::*;
        match (self, other) {
            (None, None) | (Bytes(_), Bytes(_)) | (Width(_), Width(_)) => false,
            _else => true,
        }
    }
}

pub struct VisualLine {
    pub interval: Interval,
    /// The buffer line number for this line. This is only set for the first visual line of a
    /// buffer line.
    pub line: Option<usize>,
}
impl VisualLine {
    fn new<I: Into<Interval>, L: Into<Option<usize>>>(iv: I, line: L) -> Self {
        VisualLine {
            interval: iv.into(),
            line: line.into(),
        }
    }
}

/// Detailed information about changes to linebreaks, used to generate
/// invalidation information. The logic behind invalidation is different
/// depending on whether we're updating breaks after an edit, or continuing
/// a bulk rewrapping task. This is shared between the two cases, and contains
/// the information relevant to both of them.
struct WrapSummary {
    start_line: usize,
    /// Total number of invalidated lines; this is meaningless in the after_edit case
    inval_count: usize,
    /// The total number of new (hard + soft) breaks in the wrapped region.
    new_count: usize,
    /// The number of new soft breaks
    new_soft: usize,
}

/// A range that needs to be rewrapped.
type Task = Interval;

/// Tracks state related to visual lines.
#[derive(Default)]
pub struct Lines {
    /// Tracks linebreak information
    breaks: Breaks,
    wrap: WrapWidth,
    /// Ranges of lines that need be wrapped.
    work: Vec<Task>,
}
impl Lines {
    pub fn set_wrap_width(&mut self, text: &Rope, wrap: WrapWidth) {
        // We have to rewrap everything
        self.work.clear();
        self.add_task(0..text.len());
        if self.breaks.is_empty() || self.wrap.differs_in_kind(wrap) {
            // we keep breaks while resizing, for more efficient invalidation
            self.breaks = Breaks::new_no_break(text.len());
        }
    }

    /// Add an interval to be (re)wrapped, merging it with existing work.
    fn add_task<T: Into<Interval>>(&mut self, iv: T) {
        let iv = iv.into();
        if iv.is_empty() {
            return;
        }

        // Get the index that work that intersects with our interval starts at
        let Some(relevant_work_start) = self.work.iter().position(|&t| !t.intersect(iv).is_empty()) else {
            // There was no relevant work. This means we don't need to merge anything.
            self.work.push(iv);
            return;
        };

        // TODO: I feel that there should be some way to do this without allocating a new vec most
        // of the time,
        // or at least using something like a smallvec.
        let to_update = self.work.split_off(relevant_work_start);

        // The unmerged intervals that still need to be added
        let mut new_task = Some(iv);
        for work in to_update.iter() {
            match new_task.take() {
                // If the new_task intersects the found work, then we union them and continue on.
                // This will slowly build up a larger union, until we find some disjoint work.
                Some(new) if !work.intersect(new).is_empty() => {
                    new_task = Some(work.union(new))
                }
                // Disjoint, so we push the new_task and the work rather than joining them
                Some(new) => {
                    self.work.push(new);
                    self.work.push(*work);
                }
                // TODO: minor but we could stop whenever new_task is None and just use something like vec extend, which would be slightly faster
                // There's no new_task, so we just push the work
                None => self.work.push(*work),
            }
        }

        if let Some(end) = new_task.take() {
            self.work.push(end);
        }
    }

    /// Check if the wrapping is finished
    pub fn is_converged(&self) -> bool {
        self.wrap == WrapWidth::None || self.work.is_empty()
    }

    /// Check if the interval intersects with any work that needs to be done.
    pub fn interval_needs_wrap(&self, iv: Interval) -> bool {
        self.work.iter().any(|t| !t.intersect(iv).is_empty())
    }

    pub fn visual_line_of_offset(&self, text: &Rope, offset: usize) -> usize {
        let mut line = text.line_of_offset(offset);
        if self.wrap != WrapWidth::None {
            line += self.breaks.count::<BreaksMetric>(offset);
        }
        line
    }

    pub fn offset_of_visual_line(&self, text: &Rope, line: usize) -> usize {
        todo!()
    }

    /// Iterator over [`VisualLine`]s, starting at `start_line`
    pub fn iter_lines<'a>(
        &'a self,
        text: &'a Rope,
        start_line: usize,
    ) -> impl Iterator<Item = VisualLine> + 'a {
        todo!();
        std::iter::empty() // silence err
    }

    /// Returns the next task, prioritizing the currently visible region.  
    /// Does not modify the task list; this is done after the task is ran.
    fn get_next_task(&self, visible_offset: usize) -> Option<Task> {
        // the first task `t` where `t.end > visible_offset` is the only task the might contain the
        // visible region
        self.work
            .iter()
            .find(|t| t.end > visible_offset)
            .map(|t| Task::new(t.start.max(visible_offset), t.end))
            .or(self.work.last().cloned())
    }

    /// Update the Task intervals after we finish wrapped some interval.
    fn update_tasks_after_wrap<T: Into<Interval>>(&mut self, wrapped_iv: T) {
        if self.work.is_empty() {
            return;
        }

        let wrapped_iv = wrapped_iv.into();

        // TODO: couldn't this just do some sort of iterator to replace elements as you go? to
        // avoid extra allocs
        let mut work = Vec::new();
        for task in &self.work {
            // If the task is strictly before/after the wrapped interval, then we can ignore it
            if task.is_before(wrapped_iv.start) || task.is_after(wrapped_iv.end) {
                work.push(*task);
                continue;
            }

            // Push the part at the start of the task that wasn't wrapped
            if wrapped_iv.start > task.start {
                work.push(task.prefix(wrapped_iv));
            }

            // Push the part at the end of the task that wasn't wrapped
            if wrapped_iv.end < task.end {
                work.push(task.suffix(wrapped_iv));
            }
        }

        self.work = work;
    }

    /// Adjust offsets for any tasks after an edit.
    fn patchup_tasks<T: Into<Interval>>(&mut self, iv: T, new_len: usize) {
        let iv = iv.into();
        let mut new_work = Vec::new();

        for task in &self.work {
            if task.is_before(iv.start) {
                new_work.push(*task);
            } else if task.contains(iv.start) {
                let head = task.prefix(iv);
                let tail_end =
                    iv.start.max((task.end + new_len).saturating_sub(iv.size()));
                let tail = Interval::new(iv.start, tail_end);
                new_work.push(head);
                new_work.push(tail);
            } else {
                // take task - our edit interval, then translate it (- old_size, + new_size)
                let tail =
                    task.suffix(iv).translate(new_len).translate_neg(iv.size());
                new_work.push(tail);
            }
        }
        new_work.retain(|iv| !iv.is_empty());
        self.work.clear();
        for task in new_work {
            if let Some(prev) = self.work.last_mut() {
                if prev.end >= task.start {
                    *prev = prev.union(task);
                    continue;
                }
            }
            self.work.push(task);
        }
    }

    /// Do a chunk of wrap work, if any exists.
    pub(crate) fn rewrap_chunk(
        &mut self,
        text: &Rope,
        width_cache: &mut impl WidthCalc,
        visible_lines: Range<usize>,
    ) -> Option<InvalLines> {
        if self.is_converged() {
            None
        } else {
            let summary = self.do_wrap_task(text, width_cache, visible_lines, None);
            let WrapSummary {
                start_line,
                inval_count,
                new_count,
                ..
            } = summary;
            Some(InvalLines {
                start_line,
                inval_count,
                new_count,
            })
        }
    }

    /// Updates breaks after an edit. Returns `InvalLines`, for minimal invalidation,
    /// when possible.
    pub(crate) fn after_edit(
        &mut self,
        text: &Rope,
        old_text: &Rope,
        delta: &RopeDelta,
        width_cache: &mut impl WidthCalc,
        visible_lines: Range<usize>,
    ) -> Option<InvalLines> {
        let (iv, newlen) = delta.summary();

        let logical_start_line = text.line_of_offset(iv.start);
        let old_logical_end_line = old_text.line_of_offset(iv.end) + 1;
        let new_logical_end_line = text.line_of_offset(iv.start + newlen) + 1;
        let old_logical_end_offset = old_text.offset_of_line(old_logical_end_line);
        let old_hard_count = old_logical_end_line - logical_start_line;
        let new_hard_count = new_logical_end_line - logical_start_line;

        //TODO: we should be able to avoid wrapping the whole para in most cases,
        // but the logic is trickier.
        let prev_break = text.offset_of_line(logical_start_line);
        let next_hard_break = text.offset_of_line(new_logical_end_line);

        // count the soft breaks in the region we will rewrap, before we update them.
        let inval_soft = self.breaks.count::<BreaksMetric>(old_logical_end_offset)
            - self.breaks.count::<BreaksMetric>(prev_break);

        // update soft breaks, adding empty spans in the edited region
        let mut builder = BreakBuilder::new();
        builder.add_no_break(newlen);
        self.breaks.edit(iv, builder.build());
        self.patchup_tasks(iv, newlen);

        if self.wrap == WrapWidth::None {
            return Some(InvalLines {
                start_line: logical_start_line,
                inval_count: old_hard_count,
                new_count: new_hard_count,
            });
        }

        let new_task = prev_break..next_hard_break;
        self.add_task(new_task);

        // possible if the whole buffer is deleted, e.g
        if !self.work.is_empty() {
            let summary = self.do_wrap_task(text, width_cache, visible_lines, None);
            let WrapSummary {
                start_line,
                new_soft,
                ..
            } = summary;
            // if we haven't converged after this update we can't do minimal invalidation
            // because we don't have complete knowledge of the new breaks state.
            if self.is_converged() {
                let inval_count = old_hard_count + inval_soft;
                let new_count = new_hard_count + new_soft;
                Some(InvalLines {
                    start_line,
                    inval_count,
                    new_count,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn do_wrap_task(
        &mut self,
        text: &Rope,
        width_cache: &mut impl WidthCalc,
        visible_lines: Range<usize>,
        max_lines: Option<usize>,
    ) -> WrapSummary {
        use self::WrapWidth::*;
        // 'line' is a poor unit here; could do some fancy Duration thing?
        const MAX_LINES_PER_BATCH: usize = 500;

        let mut cursor = MergedBreaks::new(text, &self.breaks);
        let visible_off = cursor.offset_of_line(visible_lines.start);
        let logical_off = text.offset_of_line(text.line_of_offset(visible_off));

        // task.start is a hard break; task.end is a boundary or EOF.
        let task = self.get_next_task(logical_off).unwrap();
        cursor.set_offset(task.start);
        debug_assert_eq!(
            cursor.offset, task.start,
            "task_start must be valid offset"
        );

        let mut ctx = match self.wrap {
            Bytes(b) => RewrapCtx::new(
                text,
                &CodepointMono,
                b as f64,
                width_cache,
                task.start,
            ),
            Width(w) => RewrapCtx::new(text, w, width_cache, task.start),
            None => unreachable!(),
        };

        let start_line = cursor.cur_line;
        let max_lines = max_lines.unwrap_or(MAX_LINES_PER_BATCH);
        // always wrap at least a screen worth of lines (unless we converge earlier)
        let batch_size = max_lines.max(visible_lines.end - visible_lines.start);

        let mut builder = BreakBuilder::new();
        let mut lines_wrapped = 0;
        let mut pos = task.start;
        let mut old_next_maybe = cursor.next();

        loop {
            if let Some(new_next) = ctx.wrap_one_line(pos) {
                while let Some(old_next) = old_next_maybe {
                    if old_next >= new_next {
                        break; // just advance old cursor and continue
                    }
                    old_next_maybe = cursor.next();
                }

                let is_hard = cursor.offset == new_next && cursor.is_hard_break();
                if is_hard {
                    builder.add_no_break(new_next - pos);
                } else {
                    builder.add_break(new_next - pos);
                }
                lines_wrapped += 1;
                pos = new_next;
                if pos == task.end || (lines_wrapped > batch_size && is_hard) {
                    break;
                }
            } else {
                // EOF
                builder.add_no_break(text.len() - pos);
                break;
            }
        }

        let breaks = builder.build();
        let end = task.start + breaks.len();

        // this is correct *only* when an edit has not occured.
        let inval_soft = self.breaks.count::<BreaksMetric>(end)
            - self.breaks.count::<BreaksMetric>(task.start);

        let hard_count =
            1 + text.line_of_offset(end) - text.line_of_offset(task.start);

        let inval_count = inval_soft + hard_count;
        let new_soft = breaks.measure::<BreaksMetric>();
        let new_count = new_soft + hard_count;

        let iv = Interval::new(task.start, end);
        self.breaks.edit(iv, breaks);
        self.update_tasks_after_wrap(iv);

        WrapSummary {
            start_line,
            inval_count,
            new_count,
            new_soft,
        }
    }

    pub fn logical_line_range(&self, text: &Rope, line: usize) -> (usize, usize) {
        let mut cursor = MergedBreaks::new(text, &self.breaks);
        let offset = cursor.offset_of_line(line);
        let logical_line = text.line_of_offset(offset);
        let start_logical_line_offset = text.offset_of_line(logical_line);
        let end_logical_line_offset = text.offset_of_line(logical_line + 1);
        (start_logical_line_offset, end_logical_line_offset)
    }

    #[cfg(test)]
    fn for_testing(text: &Rope, wrap: WrapWidth) -> Lines {
        let mut lines = Lines::default();
        lines.set_wrap_width(text, wrap);
        lines
    }

    #[cfg(test)]
    fn rewrap_all(&mut self, text: &Rope, width_cache: &mut impl WidthCalc) {
        if !self.is_converged() {
            self.do_wrap_task(text, width_cache, 0..10, Some(usize::max_value()));
        }
    }
}

/// A potential opportunity to insert a break. In this representation, the widths
/// have been requested (in a batch request) but are not necessarily known until
/// the request is issued.
struct PotentialBreak {
    /// The offset within the text of the end of the word.
    pos: usize,
    /// A token referencing the width of the word, to be resolved in the width cache.
    tok: Token,
    /// Whether the break is a hard break or a soft break.
    hard: bool,
}

/// State for a rewrap in progress
struct RewrapCtx<'a> {
    text: &'a Rope,
    lb_cursor: LineBreakCursor<'a>,
    lb_cursor_pos: usize,
    width_cache: &'a mut dyn WidthCalc,
    pot_breaks: Vec<PotentialBreak>,
    /// Index within `pot_breaks`
    pot_break_ix: usize,
    max_width: f64,
}

// This constant should be tuned so that the RPC takes about 1ms. Less than that,
// RPC overhead becomes significant. More than that, interactivity suffers.
const MAX_POT_BREAKS: usize = 10_000;

impl<'a> RewrapCtx<'a> {
    fn new(
        text: &'a Rope,
        max_width: f64,
        width_cache: &'a mut dyn WidthCalc,
        start: usize,
    ) -> RewrapCtx<'a> {
        let lb_cursor_pos = start;
        let lb_cursor = LineBreakCursor::new(text, start);
        RewrapCtx {
            text,
            lb_cursor,
            lb_cursor_pos,
            width_cache,
            pot_breaks: Vec::new(),
            pot_break_ix: 0,
            max_width,
        }
    }

    fn refill_pot_breaks(&mut self) {
        self.pot_breaks.clear();
        self.pot_break_ix = 0;
        let mut pos = self.lb_cursor_pos;
        while pos < self.text.len() && self.pot_breaks.len() < MAX_POT_BREAKS {
            let (next, hard) = self.lb_cursor.next();
            let word = self.text.slice_to_cow(pos..next);
            let tok = req.request(N_RESERVED_STYLES, &word);
            pos = next;
            self.pot_breaks.push(PotentialBreak { pos, tok, hard });
        }
        self.lb_cursor_pos = pos;
    }

    /// Compute the next break, assuming `start` is a valid break.
    ///
    /// Invariant: `start` corresponds to the start of the word referenced by `pot_break_ix`.
    fn wrap_one_line(&mut self, start: usize) -> Option<usize> {
        let mut line_width = 0.0;
        let mut pos = start;
        while pos < self.text.len() {
            if self.pot_break_ix >= self.pot_breaks.len() {
                self.refill_pot_breaks();
            }
            let pot_break = &self.pot_breaks[self.pot_break_ix];
            let width = self.width_cache.resolve(pot_break.tok);
            if !pot_break.hard {
                if line_width == 0.0 && width >= self.max_width {
                    // we don't care about soft breaks at EOF
                    if pot_break.pos == self.text.len() {
                        return None;
                    }
                    self.pot_break_ix += 1;
                    return Some(pot_break.pos);
                }
                line_width += width;
                if line_width > self.max_width {
                    return Some(pos);
                }
                self.pot_break_ix += 1;
                pos = pot_break.pos;
            } else if line_width != 0. && width + line_width > self.max_width {
                // if this is a hard break but we would have broken at the previous
                // pos otherwise, we still break at the previous pos.
                return Some(pos);
            } else {
                self.pot_break_ix += 1;
                return Some(pot_break.pos);
            }
        }
        None
    }
}

struct LineBreakCursor<'a> {
    inner: Cursor<'a, RopeInfo>,
    lb_iter: LineBreakLeafIter,
    last_byte: u8,
}

impl<'a> LineBreakCursor<'a> {
    fn new(text: &'a Rope, pos: usize) -> LineBreakCursor<'a> {
        let inner = Cursor::new(text, pos);
        let lb_iter = match inner.get_leaf() {
            Some((s, offset)) => LineBreakLeafIter::new(s.as_str(), offset),
            _ => LineBreakLeafIter::default(),
        };
        LineBreakCursor {
            inner,
            lb_iter,
            last_byte: 0,
        }
    }

    // position and whether break is hard; up to caller to stop calling after EOT
    fn next(&mut self) -> (usize, bool) {
        let mut leaf = self.inner.get_leaf();
        loop {
            match leaf {
                Some((s, offset)) => {
                    let (next, hard) = self.lb_iter.next(s.as_str());
                    if next < s.len() {
                        return (self.inner.pos() - offset + next, hard);
                    }
                    if !s.is_empty() {
                        self.last_byte = s.as_bytes()[s.len() - 1];
                    }
                    leaf = self.inner.next_leaf();
                }
                // A little hacky but only reports last break as hard if final newline
                None => return (self.inner.pos(), self.last_byte == b'\n'),
            }
        }
    }
}

struct VisualLines<'a> {
    cursor: MergedBreaks<'a>,
    offset: usize,
    /// The current logical line number.
    logical_line: usize,
    len: usize,
    eof: bool,
}

impl<'a> Iterator for VisualLines<'a> {
    type Item = VisualLine;

    fn next(&mut self) -> Option<VisualLine> {
        let line_num = if self.cursor.is_hard_break() {
            Some(self.logical_line)
        } else {
            None
        };
        let next_end_bound = match self.cursor.next() {
            Some(b) => b,
            None if self.eof => return None,
            _else => {
                self.eof = true;
                self.len
            }
        };
        let result = VisualLine::new(self.offset..next_end_bound, line_num);
        if self.cursor.is_hard_break() {
            self.logical_line += 1;
        }
        self.offset = next_end_bound;
        Some(result)
    }
}

/// A cursor over both hard and soft breaks. Hard breaks are retrieved from
/// the rope; the soft breaks are stored independently; this interleaves them.
///
/// # Invariants:
///
/// `self.offset` is always a valid break in one of the cursors, unless
/// at 0 or EOF.
///
/// `self.offset == self.text.pos().min(self.soft.pos())`.
struct MergedBreaks<'a> {
    text: Cursor<'a, RopeInfo>,
    soft: Cursor<'a, BreaksInfo>,
    offset: usize,
    /// Starting from zero, how many calls to `next` to get to `self.offset`?
    cur_line: usize,
    total_lines: usize,
    /// Total length, in base units
    len: usize,
}

impl<'a> Iterator for MergedBreaks<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.text.pos() == self.offset && !self.at_eof() {
            // don't iterate past EOF, or we can't get the leaf and check for \n
            self.text.next::<LinesMetric>();
        }
        if self.soft.pos() == self.offset {
            self.soft.next::<BreaksMetric>();
        }
        let prev_off = self.offset;
        self.offset = self.text.pos().min(self.soft.pos());

        let eof_without_newline =
            self.offset > 0 && self.at_eof() && self.eof_without_newline();
        if self.offset == prev_off || eof_without_newline {
            None
        } else {
            self.cur_line += 1;
            Some(self.offset)
        }
    }
}

// arrived at this by just trying out a bunch of values ¯\_(ツ)_/¯
/// how far away a line can be before we switch to a binary search
const MAX_LINEAR_DIST: usize = 20;

impl<'a> MergedBreaks<'a> {
    fn new(text: &'a Rope, breaks: &'a Breaks) -> Self {
        debug_assert_eq!(text.len(), breaks.len());
        let text = Cursor::new(text, 0);
        let soft = Cursor::new(breaks, 0);
        let total_lines = text.root().measure::<LinesMetric>()
            + soft.root().measure::<BreaksMetric>()
            + 1;
        let len = text.total_len();
        MergedBreaks {
            text,
            soft,
            offset: 0,
            cur_line: 0,
            total_lines,
            len,
        }
    }

    /// Sets the `self.offset` to the first valid break immediately at or preceding `offset`,
    /// and restores invariants.
    fn set_offset(&mut self, offset: usize) {
        self.text.set(offset);
        self.soft.set(offset);
        if offset > 0 {
            if self.text.at_or_prev::<LinesMetric>().is_none() {
                self.text.set(0);
            }
            if self.soft.at_or_prev::<BreaksMetric>().is_none() {
                self.soft.set(0);
            }
        }

        // self.offset should be at the first valid break immediately preceding `offset`, or 0.
        // the position of the non-break cursor should be > than that of the break cursor, or EOF.
        match self.text.pos().cmp(&self.soft.pos()) {
            Ordering::Less => {
                self.text.next::<LinesMetric>();
            }
            Ordering::Greater => {
                self.soft.next::<BreaksMetric>();
            }
            Ordering::Equal => assert!(self.text.pos() == 0),
        }

        self.offset = self.text.pos().min(self.soft.pos());
        self.cur_line =
            merged_line_of_offset(self.text.root(), self.soft.root(), self.offset);
    }

    fn offset_of_line(&mut self, line: usize) -> usize {
        match line {
            0 => 0,
            l if l >= self.total_lines => self.text.total_len(),
            l if l == self.cur_line => self.offset,
            l if l > self.cur_line && l - self.cur_line < MAX_LINEAR_DIST => {
                self.offset_of_line_linear(l)
            }
            other => self.offset_of_line_bsearch(other),
        }
    }

    fn offset_of_line_linear(&mut self, line: usize) -> usize {
        assert!(line > self.cur_line);
        let dist = line - self.cur_line;
        self.nth(dist - 1).unwrap_or(self.len)
    }

    fn offset_of_line_bsearch(&mut self, line: usize) -> usize {
        let mut range = 0..self.len;
        loop {
            let pivot = range.start + (range.end - range.start) / 2;
            self.set_offset(pivot);

            match self.cur_line {
                l if l == line => break self.offset,
                l if l > line => range = range.start..pivot,
                l if line - l > MAX_LINEAR_DIST => range = pivot..range.end,
                _else => break self.offset_of_line_linear(line),
            }
        }
    }

    fn is_hard_break(&self) -> bool {
        self.offset == self.text.pos()
    }

    fn at_eof(&self) -> bool {
        self.offset == self.len
    }

    fn eof_without_newline(&mut self) -> bool {
        debug_assert!(self.at_eof());
        self.text.set(self.len);
        self.text
            .get_leaf()
            .map(|(l, _)| l.as_bytes().last() != Some(&b'\n'))
            .unwrap()
    }
}

fn merged_line_of_offset(text: &Rope, soft: &Breaks, offset: usize) -> usize {
    text.count::<LinesMetric>(offset) + soft.count::<BreaksMetric>(offset)
}

#[cfg(test)]
mod tests {
    use lapce_xi_rope::{
        breaks::{BreakBuilder, BreaksInfo, BreaksMetric},
        tree::Node,
        Cursor,
    };

    #[test]
    fn breaks_multiple() {
        // Tests whether Breaks can have multiple newlines in one spot. Mostly just to ensure it
        // doesn't change behavior.

        let mut b = BreakBuilder::new();
        // breaks are given in length from the previous break
        b.add_break(5);
        b.add_break(0);
        b.add_break(0);

        let bnode = b.build();
        assert_eq!(bnode.len(), 5);

        let mut c = Cursor::new(&bnode, 0);
        assert_eq!(bnode.measure::<BreaksMetric>(), 3);
        assert_eq!(c.next::<BreaksMetric>().unwrap(), 5);
        // because there's no next boundary, even though there's three entries at this boundary
        assert!(c.next::<BreaksMetric>().is_none());
    }
}
