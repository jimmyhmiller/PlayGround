//! Source spans, a multi-source `SourceMap`, and rendered diagnostics.
//!
//! A `Span` is a half-open byte range `[lo, hi)` into *one* registered source —
//! identified by `source`, an index into a [`SourceMap`]. The reader stamps a span
//! (with the file's source id) onto every `Sexp`; the parser/checker forward it onto
//! the diagnostics they raise, so an error can point at the exact offending source —
//! *in the right file*, even across `import`s.
//!
//! `Span::DUMMY` marks a node with no source location (synthesized nodes with no
//! template). Rendering a dummy span falls back to the bare message.
//!
//! `ctxt` is an *expansion context*: `0` is the root (code the user wrote); a nonzero
//! value indexes a [`SourceMap`] expansion record (macro name + call-site + def-site),
//! so a diagnostic in macro-generated code can print an "in expansion of macro …"
//! trace back to the user's call site (Stage C).
//!
//! Errors are carried as a `Diag` (message + span). `Diag` converts *from*
//! `String`/`&str`, so the existing `?`-on-`Result<_, String>` call sites keep
//! working and simply produce a spanless diagnostic; a span is attached only where
//! the reader/parser/checker has the offending node in hand.

/// A half-open byte range `[lo, hi)` into the registered source `source`, in
/// expansion context `ctxt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Index into the [`SourceMap`]; `u32::MAX` for a dummy (no-source) span.
    pub source: u32,
    pub lo: u32,
    pub hi: u32,
    /// Expansion context: `0` = root (user-written); nonzero = a `SourceMap`
    /// expansion record (macro provenance). See [`SourceMap::add_expansion`].
    pub ctxt: u32,
}

impl Span {
    /// "No source location." Used for synthesized nodes with no template origin.
    pub const DUMMY: Span = Span { source: u32::MAX, lo: u32::MAX, hi: u32::MAX, ctxt: 0 };

    /// A root-context span into `source`.
    pub fn new(source: u32, lo: usize, hi: usize) -> Span {
        Span { source, lo: lo as u32, hi: hi as u32, ctxt: 0 }
    }

    pub fn is_dummy(&self) -> bool {
        self.lo == u32::MAX
    }

    /// This span in expansion context `ctxt` (macro provenance, Stage C).
    pub fn with_ctxt(mut self, ctxt: u32) -> Span {
        self.ctxt = ctxt;
        self
    }

    /// The smallest span covering both `self` and `other`. A dummy operand is
    /// ignored, so folding spans over a node's children "just works" even when some
    /// children are synthesized. Spans from *different* sources can't be unioned —
    /// `self` wins (its caret is at least in one real place).
    pub fn to(self, other: Span) -> Span {
        if self.is_dummy() {
            return other;
        }
        if other.is_dummy() || other.source != self.source {
            return self;
        }
        Span {
            source: self.source,
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
            ctxt: self.ctxt,
        }
    }
}

/// One registered source: a display name (file path, or `<source>` for the main
/// in-memory string) and its full text.
pub struct Source {
    pub name: String,
    pub text: String,
}

/// A macro-expansion record: a generated node's `ctxt` resolves to one of these, so
/// a diagnostic can trace back through the macro that produced it.
#[derive(Clone)]
pub struct Expansion {
    /// The macro's (qualified) name.
    pub macro_name: String,
    /// Where the macro was *called* (in the caller's source).
    pub call_site: Span,
    /// Where the macro is *defined* (the `(defn …)` template), so the trace can
    /// point at the code that produced the offending node. `DUMMY` if unknown.
    pub def_site: Span,
    /// The enclosing expansion context, so nested expansions chain (`0` = the
    /// macro was called from user code).
    pub parent: u32,
}

/// The registry of all sources (and expansion records) a compile touches. Built up
/// as the main file + its imports + the prelude are read, and consulted when an
/// error is rendered so the caret lands in the correct file.
#[derive(Default)]
pub struct SourceMap {
    sources: Vec<Source>,
    expansions: Vec<Expansion>,
}

impl SourceMap {
    pub fn new() -> SourceMap {
        SourceMap::default()
    }

    /// Register a source and return its id (an index usable as `Span::source`).
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> u32 {
        let id = self.sources.len() as u32;
        self.sources.push(Source { name: name.into(), text: text.into() });
        id
    }

    /// The number of registered sources (ids `0..len`).
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn text(&self, id: u32) -> Option<&str> {
        self.sources.get(id as usize).map(|s| s.text.as_str())
    }

    pub fn name(&self, id: u32) -> Option<&str> {
        self.sources.get(id as usize).map(|s| s.name.as_str())
    }

    /// Register an expansion record and return its `ctxt` id (`>= 1`; `0` is the
    /// reserved root context).
    pub fn add_expansion(&mut self, e: Expansion) -> u32 {
        self.expansions.push(e);
        self.expansions.len() as u32 // id 1 = expansions[0]
    }

    pub fn expansion(&self, ctxt: u32) -> Option<&Expansion> {
        if ctxt == 0 {
            return None;
        }
        self.expansions.get((ctxt - 1) as usize)
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

    /// Attach `span` *if* this diagnostic doesn't already carry one. Lets an outer
    /// frame add a location to an inner spanless error without clobbering a more
    /// precise span a deeper frame already set.
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

/// 1-based line and column of a byte offset, plus the offending line's text and the
/// byte offset at which that line starts. Columns count `char`s (not bytes), so a
/// caret lands under the right glyph for non-ASCII source.
fn locate(src: &str, off: usize) -> (usize, usize, &str, usize) {
    let off = off.min(src.len());
    let line_start = src[..off].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = src[off..].find('\n').map(|i| off + i).unwrap_or(src.len());
    let line = 1 + src[..line_start].bytes().filter(|&b| b == b'\n').count();
    let col = 1 + src[line_start..off].chars().count();
    (line, col, &src[line_start..line_end], line_start)
}

/// Render one located frame — `file:line:col`, the source line, and a caret — given
/// the resolved source text/name. `label` (e.g. `-->` or `::: `) prefixes the
/// location line. Returns `None` if the span doesn't resolve in `src`.
fn render_frame(span: Span, src: &str, file: &str, msg: &str, arrow: &str) -> Option<String> {
    if span.is_dummy() || span.lo as usize > src.len() {
        return None;
    }
    let lo = span.lo as usize;
    let (line, col, line_text, line_start) = locate(src, lo);
    let hi = (span.hi as usize).min(line_start + line_text.len()).max(lo);
    let caret_chars = src[lo..hi].chars().count().max(1);
    let num = line.to_string();
    let pad = " ".repeat(num.len());
    let indent = " ".repeat(col - 1);
    let carets = "^".repeat(caret_chars);
    let head = if msg.is_empty() {
        format!("{pad} {arrow} {file}:{line}:{col}")
    } else {
        format!("{msg}\n{pad} {arrow} {file}:{line}:{col}")
    };
    Some(format!(
        "{head}\n\
         {pad} |\n\
         {num} | {line_text}\n\
         {pad} | {indent}{carets}",
    ))
}

/// Render a diagnostic *body* (no `error:` prefix — the caller owns that) against a
/// [`SourceMap`]: `file:line:col`, the offending source line, a caret, and — when the
/// span sits in a macro expansion — a "note: in expansion of macro …" trace back to
/// the call site. A dummy or unresolved span falls back to just the message.
///
/// ```text
/// arithmetic on different types (f64 vs i64)
///  --> example.coil:3:8
///   |
/// 3 |   (iadd x 1.0)
///   |        ^^^^^
/// ```
pub fn render(diag: &Diag, sm: &SourceMap) -> String {
    let span = diag.span;
    let src = match sm.text(span.source) {
        Some(s) => s,
        None => return diag.msg.clone(),
    };
    let file = sm.name(span.source).unwrap_or("<source>");
    let primary = match render_frame(span, src, file, &diag.msg, "-->") {
        Some(s) => s,
        None => return diag.msg.clone(),
    };
    // Walk the expansion chain (macro provenance), appending a note per level.
    let mut out = primary;
    let mut ctxt = span.ctxt;
    while let Some(exp) = sm.expansion(ctxt) {
        // Point the note at the macro's definition (the template that produced the
        // offending node) when known, else at the call site.
        let at = if exp.def_site.is_dummy() { exp.call_site } else { exp.def_site };
        let asrc = sm.text(at.source).unwrap_or("");
        let afile = sm.name(at.source).unwrap_or("<source>");
        let note = format!("note: in expansion of macro `{}`", exp.macro_name);
        match render_frame(at, asrc, afile, &note, ":::") {
            Some(frame) => out.push_str(&format!("\n{frame}")),
            None => out.push_str(&format!("\n{note}")),
        }
        ctxt = exp.parent;
    }
    out
}

/// Render several diagnostics against one [`SourceMap`], separated by blank lines,
/// with a trailing `N errors` summary when there is more than one.
pub fn render_all(diags: &[Diag], sm: &SourceMap) -> String {
    if diags.len() <= 1 {
        return diags.first().map(|d| render(d, sm)).unwrap_or_default();
    }
    let mut out = String::new();
    for (i, d) in diags.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\nerror: ");
        }
        out.push_str(&render(d, sm));
    }
    out.push_str(&format!("\n\n{} errors", diags.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_source(src: &str) -> (SourceMap, u32) {
        let mut sm = SourceMap::new();
        let id = sm.add("ex.coil", src);
        (sm, id)
    }

    #[test]
    fn renders_line_col_and_caret() {
        let src = "(defn main []\n  (-> :i64)\n  (iadd x 1.0))\n";
        let (sm, id) = one_source(src);
        let off = src.find('x').unwrap();
        let d = Diag::at(Span::new(id, off, off + 1), "unbound variable 'x'");
        let out = render(&d, &sm);
        assert!(out.contains("ex.coil:3:"), "got: {out}");
        assert!(out.contains("unbound variable 'x'"), "got: {out}");
        assert!(out.contains("(iadd x 1.0))"), "source line missing:\n{out}");
        assert!(out.contains('^'), "caret missing:\n{out}");
    }

    #[test]
    fn dummy_span_falls_back_to_message() {
        let sm = SourceMap::new();
        let d = Diag::new("something went wrong");
        assert_eq!(render(&d, &sm), "something went wrong");
    }

    #[test]
    fn span_union_ignores_dummy_and_crosses_no_source() {
        let a = Span::new(0, 2, 5);
        assert_eq!(a.to(Span::DUMMY), a);
        assert_eq!(Span::DUMMY.to(a), a);
        assert_eq!(Span::new(0, 2, 4).to(Span::new(0, 8, 9)), Span::new(0, 2, 9));
        // different sources: self wins, no bogus cross-file union.
        assert_eq!(Span::new(0, 2, 4).to(Span::new(1, 8, 9)), Span::new(0, 2, 4));
    }

    #[test]
    fn two_sources_render_against_the_right_file() {
        let mut sm = SourceMap::new();
        let _a = sm.add("a.coil", "(main)\n");
        let b = sm.add("b.coil", "line1\nbad here\n");
        let off = sm.text(b).unwrap().find("bad").unwrap();
        let d = Diag::at(Span::new(b, off, off + 3), "boom");
        let out = render(&d, &sm);
        assert!(out.contains("b.coil:2:1"), "got: {out}");
        assert!(out.contains("bad here"), "got: {out}");
    }

    #[test]
    fn expansion_trace_notes_the_macro_definition() {
        let mut sm = SourceMap::new();
        let user = sm.add("user.coil", "(my-macro 1)\n");
        let lib = sm.add("lib.coil", "(defn my-macro ...)\n");
        let call = Span::new(user, 0, 12);
        let def = Span::new(lib, 0, 19); // the `(defn my-macro …)` template
        let ctxt = sm.add_expansion(Expansion {
            macro_name: "foo.my-macro".into(),
            call_site: call,
            def_site: def,
            parent: 0,
        });
        // The offending generated node is located at the call site (a synthesized
        // template node), in the expansion context `ctxt`.
        let d = Diag::at(Span::new(user, 0, 12).with_ctxt(ctxt), "type error");
        let out = render(&d, &sm);
        assert!(out.contains("user.coil:1:1"), "primary at call site:\n{out}");
        assert!(out.contains("in expansion of macro `foo.my-macro`"), "trace:\n{out}");
        assert!(out.contains("lib.coil:1:1"), "note points at the macro definition:\n{out}");
    }
}
