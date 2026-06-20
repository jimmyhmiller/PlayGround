//! Source spans and rendered diagnostics.
//!
//! A `Span` is a half-open byte range `[lo, hi)` into the source text. The reader
//! stamps one onto every `Sexp` node; the parser forwards it onto the errors it
//! raises, so a diagnostic can point at the exact offending source. `Span::DUMMY`
//! marks a node with no source location — macro-generated forms, and (for now)
//! forms read from included/imported files, whose offsets are into a *different*
//! source than the one being rendered. Rendering a dummy span falls back to the
//! bare message, which is exactly the pre-spans behaviour (no regression).
//!
//! Errors are carried as a `Diag` (message + span). `Diag` converts *from*
//! `String`/`&str`, so the existing `?`-on-`Result<_, String>` call sites keep
//! working and simply produce a spanless diagnostic; a span is attached only
//! where the reader/parser has the offending node in hand.

/// A half-open byte range `[lo, hi)` into a source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    /// "No source location." Used for synthesized nodes (macro output) and for
    /// forms whose bytes are in a different source than the one we render.
    pub const DUMMY: Span = Span { lo: u32::MAX, hi: u32::MAX };

    pub fn new(lo: usize, hi: usize) -> Span {
        Span { lo: lo as u32, hi: hi as u32 }
    }

    pub fn is_dummy(&self) -> bool {
        *self == Span::DUMMY
    }

    /// The smallest span covering both `self` and `other`. A dummy operand is
    /// ignored, so folding spans over a node's children "just works" even when
    /// some children are synthesized.
    pub fn to(self, other: Span) -> Span {
        if self.is_dummy() {
            return other;
        }
        if other.is_dummy() {
            return self;
        }
        Span { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }
}

/// A diagnostic: a human message plus an optional source location.
#[derive(Debug, Clone)]
pub struct Diag {
    pub msg: String,
    pub span: Span,
}

impl Diag {
    /// A spanless diagnostic (rendered as a bare message).
    pub fn new(msg: impl Into<String>) -> Diag {
        Diag { msg: msg.into(), span: Span::DUMMY }
    }

    /// A diagnostic located at `span`.
    pub fn at(span: Span, msg: impl Into<String>) -> Diag {
        Diag { msg: msg.into(), span }
    }

    /// Attach `span` *if* this diagnostic doesn't already carry one. Lets an
    /// outer parser frame add a location to an inner spanless error without
    /// clobbering a more precise span a deeper frame already set.
    pub fn with_span(mut self, span: Span) -> Diag {
        if self.span.is_dummy() {
            self.span = span;
        }
        self
    }
}

impl From<String> for Diag {
    fn from(s: String) -> Diag {
        Diag::new(s)
    }
}

impl From<&str> for Diag {
    fn from(s: &str) -> Diag {
        Diag::new(s)
    }
}

impl std::fmt::Display for Diag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

/// 1-based line and column of a byte offset, plus the offending line's text and
/// the byte offset at which that line starts. Columns count `char`s (not bytes),
/// so a caret lands under the right glyph for non-ASCII source.
fn locate(src: &str, off: usize) -> (usize, usize, &str, usize) {
    let off = off.min(src.len());
    let line_start = src[..off].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = src[off..].find('\n').map(|i| off + i).unwrap_or(src.len());
    let line = 1 + src[..line_start].bytes().filter(|&b| b == b'\n').count();
    let col = 1 + src[line_start..off].chars().count();
    (line, col, &src[line_start..line_end], line_start)
}

/// Render a diagnostic *body* (no `error:` prefix — the caller owns that) with a
/// `file:line:col`, the offending source line, and a caret/underline. A dummy or
/// out-of-range span falls back to just the message.
///
/// ```text
/// arithmetic on different types (f64 vs i64)
///  --> example.coil:3:8
///   |
/// 3 |   (iadd x 1.0)
///   |        ^^^^^
/// ```
pub fn render(diag: &Diag, src: &str, file: &str) -> String {
    let span = diag.span;
    if span.is_dummy() || span.lo as usize > src.len() {
        return diag.msg.clone();
    }
    let lo = span.lo as usize;
    let (line, col, line_text, line_start) = locate(src, lo);

    // Underline width: the span, clamped to the end of this line, at least 1.
    let hi = (span.hi as usize).min(line_start + line_text.len()).max(lo);
    let caret_chars = src[lo..hi].chars().count().max(1);

    let num = line.to_string();
    let pad = " ".repeat(num.len());
    // Indentation before the caret must match the rendered source line up to the
    // span start, counting chars (tabs render as one space here for alignment).
    let indent = " ".repeat(col - 1);
    let carets = "^".repeat(caret_chars);

    format!(
        "{msg}\n\
         {pad} --> {file}:{line}:{col}\n\
         {pad} |\n\
         {num} | {line_text}\n\
         {pad} | {indent}{carets}",
        msg = diag.msg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_line_col_and_caret() {
        let src = "(defn main []\n  (-> :i64)\n  (iadd x 1.0))\n";
        // Point at `x` on line 3.
        let off = src.find('x').unwrap();
        let d = Diag::at(Span::new(off, off + 1), "unbound variable 'x'");
        let out = render(&d, src, "ex.coil");
        assert!(out.contains("ex.coil:3:"), "got: {out}");
        assert!(out.contains("unbound variable 'x'"), "got: {out}");
        assert!(out.contains("(iadd x 1.0))"), "source line missing:\n{out}");
        assert!(out.contains('^'), "caret missing:\n{out}");
    }

    #[test]
    fn dummy_span_falls_back_to_message() {
        let d = Diag::new("something went wrong");
        assert_eq!(render(&d, "whatever", "f"), "something went wrong");
    }

    #[test]
    fn span_union_ignores_dummy() {
        let a = Span::new(2, 5);
        assert_eq!(a.to(Span::DUMMY), a);
        assert_eq!(Span::DUMMY.to(a), a);
        assert_eq!(Span::new(2, 4).to(Span::new(8, 9)), Span::new(2, 9));
    }
}
