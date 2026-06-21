//! Diagnostics: render a compile error with `file:line:col`, the source line,
//! and a caret underline — the difference between a toy and a real compiler in
//! daily use.
//!
//! ```text
//! error: the trait bound `B: Show` is not satisfied (required by `describe`)
//!   --> app.gcr:6:20
//!    |
//!  6 | fn main() -> i64 { describe(B { v: 42 }) }
//!    |                    ^^^^^^^^^^^^^^^^^^^^^^
//! ```

use crate::lexer::Span;

/// Map a byte offset to a 1-based (line, column).
pub fn line_col(src: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(src.len());
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in src.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// The full text of the line containing `offset` (without its newline), plus the
/// 1-based line number.
fn line_text(src: &str, offset: usize) -> (usize, &str) {
    let offset = offset.min(src.len());
    let line_start = src[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = src[offset..].find('\n').map(|i| offset + i).unwrap_or(src.len());
    let line_no = src[..line_start].bytes().filter(|b| *b == b'\n').count() + 1;
    (line_no, &src[line_start..line_end])
}

/// The embedded standard-library source. Prelude items are lexed from this text
/// separately from the user program, so a span originating in the prelude indexes
/// into here, not the user file. `render` falls back to it when a span lands
/// outside the user source.
const PRELUDE_SRC: &str = include_str!("prelude.gcr");

/// Resolve a byte offset to `(source-label, 1-based line, col)`, choosing which
/// source it belongs to: the user `src`/`file` when the offset is inside it, else
/// the injected prelude (lexed in its OWN offset space, so prelude spans index
/// `PRELUDE_SRC`, not the user file). `None` if it indexes neither — an HONEST
/// "no location" rather than a fabricated one (the bug this fixes: clamping an
/// out-of-range prelude offset to the end of a small user file gave a confidently
/// WRONG `user.gcr:11:1`).
///
/// This is the user+prelude model diagnostics already use. USER code is always
/// resolved correctly (user spans are always `< src.len()`). LIMITATION: a
/// prelude span whose offset happens to be `< src.len()` (only possible with a
/// large user file) is still attributed to the user file, and per-`mod` file
/// sources aren't distinguished — both need a full offset-disjoint SourceMap
/// (the production multi-source fix; see docs/DEBUGGER_DESIGN.md P2).
pub fn resolve_location(file: &str, src: &str, off: usize) -> Option<(String, usize, usize)> {
    if !src.is_empty() && off < src.len() {
        let (l, c) = line_col(src, off);
        return Some((file.to_string(), l, c));
    }
    if off < PRELUDE_SRC.len() {
        let (l, c) = line_col(PRELUDE_SRC, off);
        return Some(("<std>".to_string(), l, c));
    }
    None
}

/// Render a diagnostic for `span` in `src` (named `file`) with `msg`. The span's
/// `start..end` is underlined with carets (clamped to a single line).
pub fn render(file: &str, src: &str, span: Span, msg: &str) -> String {
    // A span may point outside `src` — most commonly into the injected prelude,
    // whose text is lexed separately. Re-render against the prelude source so the
    // user still sees the offending stdlib line with carets.
    if (span.start as usize) >= src.len() {
        if (span.start as usize) < PRELUDE_SRC.len() {
            let inner = render_in("<std>", PRELUDE_SRC, span, msg);
            return format!("{}\n note: this error is in the standard library (prelude)", inner);
        }
        return format!("error: {msg}\n note: no source location available");
    }
    render_in(file, src, span, msg)
}

/// Render a diagnostic for `span` known to index into `src`.
fn render_in(file: &str, src: &str, span: Span, msg: &str) -> String {
    let start = span.start as usize;
    let end = (span.end as usize).max(start + 1).min(src.len());
    let (line, col) = line_col(src, start);
    let (line_no, text) = line_text(src, start);
    debug_assert_eq!(line, line_no);

    // Caret run: from `col` for the span's width, but not past the line's end.
    let line_start_byte = src[..start.min(src.len())].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end_byte = line_start_byte + text.len();
    let caret_end = end.min(line_end_byte);
    let caret_len = src[start..caret_end.max(start)].chars().count().max(1);

    let gutter = format!("{}", line);
    let pad = " ".repeat(gutter.len());
    let underline = format!("{}{}", " ".repeat(col.saturating_sub(1)), "^".repeat(caret_len));

    format!(
        "error: {msg}\n\
         {pad}--> {file}:{line}:{col}\n\
         {pad} |\n\
         {gutter} | {text}\n\
         {pad} | {underline}",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_basic() {
        let src = "abc\ndef\nghi";
        assert_eq!(line_col(src, 0), (1, 1));
        assert_eq!(line_col(src, 4), (2, 1)); // 'd'
        assert_eq!(line_col(src, 6), (2, 3)); // 'f'
        assert_eq!(line_col(src, 8), (3, 1)); // 'g'
    }

    #[test]
    fn renders_caret() {
        let src = "fn main() -> i64 { true }";
        // span over `true` (offset 19..23)
        let out = render("app.gcr", src, Span::new(19, 23), "expected i64, found bool");
        assert!(out.contains("app.gcr:1:20"));
        assert!(out.contains("^^^^"));
        assert!(out.contains("fn main() -> i64 { true }"));
    }

    #[test]
    fn multiline_picks_right_line() {
        let src = "fn a() -> i64 {\n  let x = bad;\n  0\n}";
        let off = src.find("bad").unwrap();
        let out = render("x.gcr", src, Span::new(off, off + 3), "unknown");
        assert!(out.contains("x.gcr:2:"));
        assert!(out.contains("let x = bad;"));
    }

    #[test]
    fn prelude_span_falls_back_to_stdlib_source() {
        // A span past the user source but within the prelude renders against the
        // prelude with a stdlib note (no panic, real context).
        let user = "fn main() -> i64 { 0 }";
        // Pick an offset inside the (larger) prelude text.
        let off = user.len() + 20;
        assert!(off < PRELUDE_SRC.len());
        let out = render("app.gcr", user, Span::new(off, off + 3), "some stdlib error");
        assert!(out.contains("standard library"), "{out}");
    }

    #[test]
    fn span_fully_out_of_range_is_graceful() {
        let user = "fn main() -> i64 { 0 }";
        let huge = PRELUDE_SRC.len() + user.len() + 1000;
        let out = render("app.gcr", user, Span::new(huge, huge + 1), "weird");
        assert!(out.contains("no source location"), "{out}");
    }
}
