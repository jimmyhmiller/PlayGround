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
fn line_col(src: &str, offset: usize) -> (usize, usize) {
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

/// Render a diagnostic for `span` in `src` (named `file`) with `msg`. The span's
/// `start..end` is underlined with carets (clamped to a single line).
pub fn render(file: &str, src: &str, span: Span, msg: &str) -> String {
    let start = span.start as usize;
    let end = (span.end as usize).max(start + 1);
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
}
