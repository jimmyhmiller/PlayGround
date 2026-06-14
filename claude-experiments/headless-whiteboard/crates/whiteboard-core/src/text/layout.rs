//! Text layout: word wrapping, line layout, and bound-text container fitting.
//!
//! Reimplemented in Rust from Excalidraw's text layout logic, principally
//! `packages/excalidraw/element/textElement.ts` (`wrapText`,
//! `getContainerDims`, `computeBoundTextPosition`,
//! `measureText`/`getBoundTextElementOffset`) and the alignment handling in
//! `packages/excalidraw/element/textWysiwyg.tsx`. No JavaScript is vendored;
//! the algorithms are ported. See `ATTRIBUTION.md`.
//!
//! All glyph measurement goes through the injected [`TextMeasurer`] so the core
//! stays headless. Wrapping is greedy (first-fit) on whitespace, matching
//! Excalidraw: words that do not fit start a new line, words longer than the
//! available width are hard-broken, and explicit `\n` always force a break.

use super::{FontSpec, TextAlign, TextMeasurer, TextMetrics, TextRun, VerticalAlign};
use crate::geometry::{Point, Rect};

/// Padding Excalidraw keeps between a container's edge and its bound text label,
/// on each side. Mirrors `BOUND_TEXT_PADDING` in Excalidraw's constants.
pub const BOUND_TEXT_PADDING: f64 = 5.0;

/// Greedy word-wrap a string to fit within `max_width`, measuring every
/// candidate line through `measurer`.
///
/// Rules (matching Excalidraw's `wrapText`):
/// - Explicit `\n` newlines always force a line break and are preserved as
///   distinct (possibly empty) lines.
/// - Within a paragraph, words are packed greedily; a word that would overflow
///   the current line starts a new line.
/// - A single word wider than `max_width` is hard-broken character-by-character
///   so no returned line exceeds `max_width` (except an unavoidable single
///   character that is itself wider than `max_width`).
/// - Runs of spaces between words collapse to the single separating space used
///   while packing, exactly as Excalidraw's whitespace-split wrapping does.
///
/// `max_width` must be finite and positive; otherwise wrapping is a no-op split
/// on explicit newlines only (there is no width to fit to).
pub fn wrap_text(
    measurer: &dyn TextMeasurer,
    text: &str,
    font: &FontSpec,
    max_width: f64,
) -> Vec<String> {
    let mut lines = Vec::new();

    // Explicit newlines are hard breaks regardless of width.
    for paragraph in text.split('\n') {
        if !max_width.is_finite() || max_width <= 0.0 {
            // No usable width to wrap to: keep the paragraph as a single line.
            lines.push(paragraph.to_string());
            continue;
        }
        wrap_paragraph(measurer, paragraph, font, max_width, &mut lines);
    }

    lines
}

/// Wrap a single newline-free paragraph, appending its resulting lines.
fn wrap_paragraph(
    measurer: &dyn TextMeasurer,
    paragraph: &str,
    font: &FontSpec,
    max_width: f64,
    out: &mut Vec<String>,
) {
    // An empty paragraph (e.g. a blank line) contributes one empty line.
    if paragraph.is_empty() {
        out.push(String::new());
        return;
    }

    let mut current = String::new();

    for word in paragraph.split(' ') {
        // A run like "a  b" splits into ["a", "", "b"]; an empty token here is a
        // second space. Treat it as widening the separator: only emit it if the
        // current line still fits, otherwise it is dropped at a wrap boundary
        // (Excalidraw collapses leading whitespace on wrapped lines).
        if word.is_empty() {
            // Try to append a space to the current line if it fits.
            if !current.is_empty() {
                let candidate = format!("{current} ");
                if fits(measurer, &candidate, font, max_width) {
                    current = candidate;
                }
            }
            continue;
        }

        let candidate = if current.is_empty() {
            (*word).to_string()
        } else {
            format!("{current} {word}")
        };

        if fits(measurer, &candidate, font, max_width) {
            current = candidate;
            continue;
        }

        // Doesn't fit. Flush the current line (if any) and place the word.
        if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }

        if fits(measurer, word, font, max_width) {
            // Word fits on a fresh line.
            current = (*word).to_string();
        } else {
            // Word is too wide even alone: hard-break it by characters.
            let broken = break_long_word(measurer, word, font, max_width);
            // All but the last fragment are complete lines; the last seeds the
            // running line so following words can pack onto it.
            let last = broken.len().saturating_sub(1);
            for (j, frag) in broken.into_iter().enumerate() {
                if j == last {
                    current = frag;
                } else {
                    out.push(frag);
                }
            }
        }
    }

    out.push(current);
}

/// Hard-break a single word that is wider than `max_width` into fragments, each
/// no wider than `max_width` where possible. A single character wider than
/// `max_width` becomes its own fragment (it cannot be split further).
fn break_long_word(
    measurer: &dyn TextMeasurer,
    word: &str,
    font: &FontSpec,
    max_width: f64,
) -> Vec<String> {
    let mut fragments = Vec::new();
    let mut current = String::new();

    for ch in word.chars() {
        let candidate = {
            let mut c = current.clone();
            c.push(ch);
            c
        };
        if current.is_empty() {
            // Always accept at least one character, even if it overflows.
            current = candidate;
        } else if fits(measurer, &candidate, font, max_width) {
            current = candidate;
        } else {
            fragments.push(std::mem::take(&mut current));
            current.push(ch);
        }
    }

    if !current.is_empty() {
        fragments.push(current);
    }

    fragments
}

#[inline]
fn fits(measurer: &dyn TextMeasurer, text: &str, font: &FontSpec, max_width: f64) -> bool {
    measurer.measure(text, font).width <= max_width
}

/// A fully laid-out block of text: the wrapped lines, per-line measured metrics,
/// and the overall bounding size. Produced by [`layout_text`].
#[derive(Debug, Clone, PartialEq)]
pub struct LaidOutText {
    /// The wrapped line strings, top to bottom.
    pub lines: Vec<String>,
    /// Measured metrics for each line, parallel to `lines`.
    pub metrics: Vec<TextMetrics>,
    /// The font the block was laid out in.
    pub font: FontSpec,
    /// Horizontal alignment used when positioning runs.
    pub align: TextAlign,
    /// Width of the widest line.
    pub width: f64,
    /// Total height: `line_count * line_spacing` (Excalidraw uses a fixed
    /// line-height-driven block height, not the sum of per-line glyph heights).
    pub height: f64,
}

impl LaidOutText {
    /// Number of laid-out lines.
    #[inline]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Distance between successive baselines.
    #[inline]
    pub fn line_spacing(&self) -> f64 {
        self.font.line_spacing()
    }

    /// Produce positioned [`TextRun`]s with baseline-left origins.
    ///
    /// `origin` is the top-left of the text block. `vertical` shifts the block
    /// within a box of height `box_height` (if given); without a box height the
    /// block is placed flush at `origin.y` (top-aligned). Horizontal alignment
    /// shifts each line within the block's own [`width`](Self::width).
    ///
    /// Baselines sit one ascent below each line's top, then advance by
    /// [`line_spacing`](Self::line_spacing). This matches how Excalidraw draws
    /// each wrapped line at `y + lineHeightPx * i` with the font's baseline.
    pub fn runs_at(
        &self,
        origin: Point,
        vertical: VerticalAlign,
        box_height: Option<f64>,
    ) -> Vec<TextRun> {
        let line_spacing = self.line_spacing();
        let block_height = self.height;

        // Vertical offset of the block's top inside the box.
        let v_offset = match box_height {
            Some(h) => match vertical {
                VerticalAlign::Top => 0.0,
                VerticalAlign::Middle => (h - block_height) / 2.0,
                VerticalAlign::Bottom => h - block_height,
            },
            None => 0.0,
        };

        let block_top = origin.y + v_offset;

        self.lines
            .iter()
            .zip(self.metrics.iter())
            .enumerate()
            .map(|(i, (line, m))| {
                let line_top = block_top + line_spacing * i as f64;
                // Baseline-left: the baseline is `ascent` below the line top.
                let baseline_y = line_top + m.ascent;

                // Horizontal placement within the block width.
                let x_offset = match self.align {
                    TextAlign::Left => 0.0,
                    TextAlign::Center => (self.width - m.width) / 2.0,
                    TextAlign::Right => self.width - m.width,
                };

                TextRun {
                    text: line.clone(),
                    font: self.font.clone(),
                    origin: Point::new(origin.x + x_offset, baseline_y),
                    align: self.align,
                }
            })
            .collect()
    }
}

/// Lay out `text` in `font`: wrap (if `max_width` is given), measure every line,
/// and compute the overall block size.
///
/// With `max_width = None` the text is split on explicit newlines only and never
/// wrapped. With `Some(w)` it is greedily word-wrapped to width `w` via
/// [`wrap_text`].
pub fn layout_text(
    measurer: &dyn TextMeasurer,
    text: &str,
    font: &FontSpec,
    align: TextAlign,
    max_width: Option<f64>,
) -> LaidOutText {
    let lines = match max_width {
        Some(w) => wrap_text(measurer, text, font, w),
        None => text.split('\n').map(|s| s.to_string()).collect(),
    };

    let metrics: Vec<TextMetrics> = lines.iter().map(|l| measurer.measure(l, font)).collect();

    let width = metrics
        .iter()
        .map(|m| m.width)
        .fold(0.0_f64, |acc, w| acc.max(w));

    // Excalidraw's block height is line count times the line-height-derived
    // spacing, independent of per-glyph ascent/descent.
    let height = font.line_spacing() * lines.len() as f64;

    LaidOutText {
        lines,
        metrics,
        font: font.clone(),
        align,
        width,
        height,
    }
}

/// The computed size and placement of a text label bound inside a container.
///
/// Mirrors Excalidraw's bound-text sizing: the label wraps to the container's
/// inner width (container width minus padding on both sides), and the resulting
/// text block is positioned within the container according to the requested
/// alignments, inset by [`BOUND_TEXT_PADDING`].
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerTextLayout {
    /// The laid-out (wrapped) text.
    pub text: LaidOutText,
    /// Top-left of the text block in scene coordinates.
    pub origin: Point,
    /// Tight bounding rect of the placed text block.
    pub bounds: Rect,
    /// Inner width the text was wrapped to (container width minus padding).
    pub inner_width: f64,
    /// Minimum container height needed to fit the text without clipping
    /// (text block height plus padding top and bottom). Excalidraw grows the
    /// container to at least this height.
    pub required_container_height: f64,
}

/// Compute the wrapped size and placement of a text label bound inside
/// `container`, mirroring Excalidraw's bound-text behavior.
///
/// The text wraps to `container.width - 2 * BOUND_TEXT_PADDING`. Vertical and
/// horizontal placement use `vertical` / `horizontal` within the padded inner
/// box. The returned [`ContainerTextLayout::required_container_height`] reports
/// the height the container must have to fully contain the label; callers that
/// auto-grow containers should expand to at least that value.
pub fn container_text_dimensions(
    measurer: &dyn TextMeasurer,
    container: Rect,
    text: &str,
    font: &FontSpec,
    horizontal: TextAlign,
    vertical: VerticalAlign,
) -> ContainerTextLayout {
    let inner_width = (container.width - 2.0 * BOUND_TEXT_PADDING).max(0.0);

    let laid = layout_text(measurer, text, font, horizontal, Some(inner_width));

    let inner_height = (container.height - 2.0 * BOUND_TEXT_PADDING).max(0.0);

    // Horizontal placement of the block's left edge inside the padded box.
    let h_offset = match horizontal {
        TextAlign::Left => 0.0,
        TextAlign::Center => (inner_width - laid.width) / 2.0,
        TextAlign::Right => inner_width - laid.width,
    };

    // Vertical placement of the block's top inside the padded box.
    let v_offset = match vertical {
        VerticalAlign::Top => 0.0,
        VerticalAlign::Middle => (inner_height - laid.height) / 2.0,
        VerticalAlign::Bottom => inner_height - laid.height,
    };

    let origin = Point::new(
        container.x + BOUND_TEXT_PADDING + h_offset,
        container.y + BOUND_TEXT_PADDING + v_offset,
    );

    let bounds = Rect::new(origin.x, origin.y, laid.width, laid.height);

    let required_container_height = laid.height + 2.0 * BOUND_TEXT_PADDING;

    ContainerTextLayout {
        text: laid,
        origin,
        bounds,
        inner_width,
        required_container_height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{FontFamily, MonospaceMeasurer};

    /// A measurer whose advance is exactly 1 unit per char per font-size unit:
    /// `advance_ratio = 1.0` means a char at size 10 is 10 wide. Makes the
    /// arithmetic in assertions exact and obvious.
    fn unit_measurer() -> MonospaceMeasurer {
        MonospaceMeasurer { advance_ratio: 1.0 }
    }

    /// Font sized so one character is exactly 10 wide with `unit_measurer`.
    fn font10() -> FontSpec {
        FontSpec::new(FontFamily::Code, 10.0)
    }

    #[test]
    fn no_wrap_when_within_width() {
        let m = unit_measurer();
        let f = font10();
        // "abc" -> 30 wide, fits in 100.
        let lines = wrap_text(&m, "abc", &f, 100.0);
        assert_eq!(lines, vec!["abc".to_string()]);
    }

    #[test]
    fn greedy_wrap_packs_words() {
        let m = unit_measurer();
        let f = font10();
        // Each word is 3 chars = 30 wide; "aaa bbb" = 70 wide.
        // max_width 70 -> both fit on one line. 60 -> they split.
        assert_eq!(
            wrap_text(&m, "aaa bbb", &f, 70.0),
            vec!["aaa bbb".to_string()]
        );
        assert_eq!(
            wrap_text(&m, "aaa bbb", &f, 60.0),
            vec!["aaa".to_string(), "bbb".to_string()]
        );
    }

    #[test]
    fn wrap_point_is_exact_at_char_advance() {
        let m = unit_measurer();
        let f = font10();
        // "aa bb cc": words 20 each, separators add a space (10) each.
        // "aa bb" = 50. width 50 fits exactly; "aa bb cc" = 80 doesn't.
        let lines = wrap_text(&m, "aa bb cc", &f, 50.0);
        assert_eq!(lines, vec!["aa bb".to_string(), "cc".to_string()]);

        // width 49 -> "aa bb" (50) no longer fits, so "bb" wraps.
        let lines = wrap_text(&m, "aa bb cc", &f, 49.0);
        assert_eq!(
            lines,
            vec!["aa".to_string(), "bb".to_string(), "cc".to_string()]
        );
    }

    #[test]
    fn preserves_explicit_newlines() {
        let m = unit_measurer();
        let f = font10();
        let lines = wrap_text(&m, "ab\ncd", &f, 1000.0);
        assert_eq!(lines, vec!["ab".to_string(), "cd".to_string()]);
    }

    #[test]
    fn preserves_blank_lines() {
        let m = unit_measurer();
        let f = font10();
        let lines = wrap_text(&m, "a\n\nb", &f, 1000.0);
        assert_eq!(lines, vec!["a".to_string(), String::new(), "b".to_string()]);
    }

    #[test]
    fn breaks_overlong_word() {
        let m = unit_measurer();
        let f = font10();
        // "aaaaa" = 50 wide, max_width 30 -> 3 chars per line: "aaa","aa".
        let lines = wrap_text(&m, "aaaaa", &f, 30.0);
        assert_eq!(lines, vec!["aaa".to_string(), "aa".to_string()]);
    }

    #[test]
    fn single_char_wider_than_width_is_kept() {
        let m = unit_measurer();
        let f = font10();
        // One char is 10 wide; max_width 5 can't fit it, but we never drop it.
        let lines = wrap_text(&m, "ab", &f, 5.0);
        assert_eq!(lines, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn overlong_word_then_following_word_packs() {
        let m = unit_measurer();
        let f = font10();
        // "aaaaa" breaks into "aaa"/"aa"; then "b" should pack onto "aa" -> "aa b"?
        // "aa b" = 40 > 30, so "b" wraps to its own line.
        let lines = wrap_text(&m, "aaaaa b", &f, 30.0);
        assert_eq!(
            lines,
            vec!["aaa".to_string(), "aa".to_string(), "b".to_string()]
        );

        // With width 40, "aa b" (40) fits.
        let lines = wrap_text(&m, "aaaaa b", &f, 40.0);
        assert_eq!(lines, vec!["aaaa".to_string(), "a b".to_string()]);
    }

    #[test]
    fn collapses_double_space_at_wrap() {
        let m = unit_measurer();
        let f = font10();
        // "aa  bb": tokens ["aa","","bb"]. width 20 -> only "aa" fits per line,
        // the extra space is dropped at the wrap boundary.
        let lines = wrap_text(&m, "aa  bb", &f, 20.0);
        assert_eq!(lines, vec!["aa".to_string(), "bb".to_string()]);
    }

    #[test]
    fn no_max_width_splits_on_newlines_only() {
        let m = unit_measurer();
        let f = font10();
        let lines = wrap_text(&m, "a very long line", &f, f64::INFINITY);
        assert_eq!(lines, vec!["a very long line".to_string()]);
    }

    #[test]
    fn layout_computes_width_and_height() {
        let m = unit_measurer();
        let f = font10(); // line_spacing = 10 * 1.25 = 12.5
        let laid = layout_text(&m, "aa\nbbbb", &f, TextAlign::Left, None);
        assert_eq!(laid.line_count(), 2);
        // widest line is "bbbb" = 40.
        assert_eq!(laid.width, 40.0);
        // height = 2 * 12.5 = 25.
        assert_eq!(laid.height, 25.0);
    }

    #[test]
    fn layout_with_wrap_changes_line_count() {
        let m = unit_measurer();
        let f = font10();
        let laid = layout_text(&m, "aa bb cc", &f, TextAlign::Left, Some(50.0));
        // "aa bb" / "cc"
        assert_eq!(laid.line_count(), 2);
        assert_eq!(laid.lines, vec!["aa bb".to_string(), "cc".to_string()]);
    }

    #[test]
    fn runs_baseline_left_origins_top_aligned() {
        let m = unit_measurer();
        let f = font10(); // ascent = 8, line_spacing = 12.5
        let laid = layout_text(&m, "aa\nbb", &f, TextAlign::Left, None);
        let runs = laid.runs_at(Point::new(100.0, 200.0), VerticalAlign::Top, None);
        assert_eq!(runs.len(), 2);
        // First line baseline: top 200 + ascent 8 = 208, x = 100.
        assert_eq!(runs[0].origin, Point::new(100.0, 208.0));
        // Second line top = 200 + 12.5 = 212.5, baseline = 220.5.
        assert_eq!(runs[1].origin, Point::new(100.0, 220.5));
    }

    #[test]
    fn runs_center_align_shifts_x() {
        let m = unit_measurer();
        let f = font10();
        // Lines "a" (10) and "aaa" (30); block width = 30.
        let laid = layout_text(&m, "a\naaa", &f, TextAlign::Center, None);
        assert_eq!(laid.width, 30.0);
        let runs = laid.runs_at(Point::new(0.0, 0.0), VerticalAlign::Top, None);
        // "a": (30 - 10)/2 = 10 offset.
        assert_eq!(runs[0].origin.x, 10.0);
        // "aaa": (30 - 30)/2 = 0 offset.
        assert_eq!(runs[1].origin.x, 0.0);
    }

    #[test]
    fn runs_right_align_shifts_x() {
        let m = unit_measurer();
        let f = font10();
        let laid = layout_text(&m, "a\naaa", &f, TextAlign::Right, None);
        let runs = laid.runs_at(Point::new(0.0, 0.0), VerticalAlign::Top, None);
        // "a": 30 - 10 = 20.
        assert_eq!(runs[0].origin.x, 20.0);
        assert_eq!(runs[1].origin.x, 0.0);
    }

    #[test]
    fn runs_vertical_middle_centers_block() {
        let m = unit_measurer();
        let f = font10(); // line_spacing 12.5
        let laid = layout_text(&m, "aa", &f, TextAlign::Left, None);
        // block height = 12.5. In a box of height 112.5, middle offset =
        // (112.5 - 12.5)/2 = 50. baseline = 0 + 50 + ascent 8 = 58.
        let runs = laid.runs_at(Point::new(0.0, 0.0), VerticalAlign::Middle, Some(112.5));
        assert_eq!(runs[0].origin.y, 58.0);
    }

    #[test]
    fn runs_vertical_bottom_aligns_block() {
        let m = unit_measurer();
        let f = font10();
        let laid = layout_text(&m, "aa", &f, TextAlign::Left, None);
        // block height 12.5, box 100 -> offset 87.5, baseline 87.5 + 8 = 95.5.
        let runs = laid.runs_at(Point::new(0.0, 0.0), VerticalAlign::Bottom, Some(100.0));
        assert_eq!(runs[0].origin.y, 95.5);
    }

    #[test]
    fn container_wraps_to_inner_width() {
        let m = unit_measurer();
        let f = font10();
        // Container width 70 -> inner width 70 - 10 = 60. "aaa bbb" (70) wraps.
        let container = Rect::new(0.0, 0.0, 70.0, 100.0);
        let layout = container_text_dimensions(
            &m,
            container,
            "aaa bbb",
            &f,
            TextAlign::Left,
            VerticalAlign::Top,
        );
        assert_eq!(layout.inner_width, 60.0);
        assert_eq!(
            layout.text.lines,
            vec!["aaa".to_string(), "bbb".to_string()]
        );
    }

    #[test]
    fn container_origin_respects_padding_top_left() {
        let m = unit_measurer();
        let f = font10();
        let container = Rect::new(10.0, 20.0, 200.0, 200.0);
        let layout =
            container_text_dimensions(&m, container, "ab", &f, TextAlign::Left, VerticalAlign::Top);
        // Top-left placement: container origin + padding.
        assert_eq!(
            layout.origin,
            Point::new(10.0 + BOUND_TEXT_PADDING, 20.0 + BOUND_TEXT_PADDING)
        );
    }

    #[test]
    fn container_centers_text_block() {
        let m = unit_measurer();
        let f = font10(); // line_spacing 12.5
        let container = Rect::new(0.0, 0.0, 120.0, 120.0);
        // inner box: width 110, height 110. text "aa" -> width 20, height 12.5.
        let layout = container_text_dimensions(
            &m,
            container,
            "aa",
            &f,
            TextAlign::Center,
            VerticalAlign::Middle,
        );
        // h_offset = (110 - 20)/2 = 45; x = 0 + 5 + 45 = 50.
        assert_eq!(layout.origin.x, 50.0);
        // v_offset = (110 - 12.5)/2 = 48.75; y = 0 + 5 + 48.75 = 53.75.
        assert_eq!(layout.origin.y, 53.75);
    }

    #[test]
    fn container_required_height_includes_padding() {
        let m = unit_measurer();
        let f = font10(); // line_spacing 12.5
        let container = Rect::new(0.0, 0.0, 200.0, 10.0);
        // Two wrapped lines -> block height 25; required = 25 + 10 = 35.
        let layout = container_text_dimensions(
            &m,
            container,
            "aa\nbb",
            &f,
            TextAlign::Left,
            VerticalAlign::Top,
        );
        assert_eq!(layout.text.line_count(), 2);
        assert_eq!(layout.required_container_height, 35.0);
    }

    #[test]
    fn container_bounds_match_block() {
        let m = unit_measurer();
        let f = font10();
        let container = Rect::new(0.0, 0.0, 200.0, 200.0);
        let layout = container_text_dimensions(
            &m,
            container,
            "aaa",
            &f,
            TextAlign::Left,
            VerticalAlign::Top,
        );
        assert_eq!(layout.bounds.x, layout.origin.x);
        assert_eq!(layout.bounds.y, layout.origin.y);
        assert_eq!(layout.bounds.width, layout.text.width);
        assert_eq!(layout.bounds.height, layout.text.height);
    }

    #[test]
    fn integrates_with_default_monospace_measurer() {
        // Sanity check against the shared MonospaceMeasurer (advance 0.6).
        let m = MonospaceMeasurer::default();
        let f = FontSpec::new(FontFamily::Code, 10.0); // char width = 6.
                                                       // "abcd" = 24 wide. max_width 24 fits, 23 forces a break of the word.
        assert_eq!(wrap_text(&m, "abcd", &f, 24.0), vec!["abcd".to_string()]);
        assert_eq!(
            wrap_text(&m, "abcd", &f, 23.0),
            vec!["abc".to_string(), "d".to_string()]
        );
    }
}
