//! Syntax highlighting via tree-sitter.
//!
//! One full reparse per doc change; highlights stored as a flat
//! `(byte_range, kind)` list sorted by start. Per-line spans are
//! extracted on demand by painting a small byte→kind buffer scoped to
//! the line and run-length-encoding it. Later captures overwrite
//! earlier ones, matching the ordering convention in
//! tree-sitter-rust's `highlights.scm`.

use std::ops::Range;

use bevy::prelude::*;
use ropey::Rope;
use style_bevy::{tokens, Theme, ThemeChanged};
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HighlightKind {
    Default,
    Keyword,
    String,
    Comment,
    Function,
    Type,
    Attribute,
    Constant,
    Operator,
    Punctuation,
    Variable,
    Property,
    Label,
    Escape,
    Constructor,
}

impl HighlightKind {
    fn slot(self) -> usize {
        self as usize
    }
}

const PALETTE_SIZE: usize = 15;

/// Per-syntax-kind color palette, refreshed from theme tokens whenever
/// the theme changes. Held as a Bevy `Resource` so it can be read from
/// any system without lifetime juggling on the `Highlighter`.
#[derive(Resource, Clone, Debug)]
pub struct SyntaxPalette {
    colors: [Color; PALETTE_SIZE],
    /// Bumped each time the palette is rewritten from theme. Line
    /// renderers compare this against their stored value to decide
    /// when to re-emit colored spans (otherwise a theme switch
    /// wouldn't visually retone already-rendered lines).
    pub rev: u64,
}

impl Default for SyntaxPalette {
    fn default() -> Self {
        // Match the legacy hardcoded defaults so a startup-before-theme
        // render doesn't flash uncolored.
        Self {
            colors: [
                Color::srgb(0.92, 0.92, 0.94),  // Default
                Color::srgb(0.78, 0.55, 0.90),  // Keyword
                Color::srgb(0.65, 0.87, 0.60),  // String
                Color::srgb(0.48, 0.52, 0.58),  // Comment
                Color::srgb(0.55, 0.78, 1.00),  // Function
                Color::srgb(0.94, 0.82, 0.55),  // Type
                Color::srgb(0.85, 0.70, 0.50),  // Attribute
                Color::srgb(0.95, 0.62, 0.48),  // Constant
                Color::srgb(0.70, 0.75, 0.82),  // Operator
                Color::srgb(0.70, 0.75, 0.82),  // Punctuation
                Color::srgb(0.92, 0.92, 0.94),  // Variable
                Color::srgb(0.85, 0.82, 0.95),  // Property
                Color::srgb(0.95, 0.62, 0.48),  // Label
                Color::srgb(0.95, 0.75, 0.40),  // Escape
                Color::srgb(0.94, 0.82, 0.55),  // Constructor
            ],
            rev: 0,
        }
    }
}

impl SyntaxPalette {
    pub fn color_for(&self, kind: HighlightKind) -> Color {
        self.colors[kind.slot()]
    }
}

/// Convenience for callers that just need the palette singleton. The
/// `World` form keeps lifetime juggling out of `lib.rs` where callers
/// already have a `&World`.
pub fn color_for(palette: &SyntaxPalette, kind: HighlightKind) -> Color {
    palette.color_for(kind)
}

/// Sync `SyntaxPalette` from `Theme`. Called once at startup and on
/// every `ThemeChanged` message so a preset switch retones syntax in
/// the same frame the chrome retones.
pub fn refresh_syntax_palette(theme: Res<Theme>, mut palette: ResMut<SyntaxPalette>) {
    let lin = |id| {
        let c = theme.color(id);
        Color::LinearRgba(c)
    };
    palette.colors = [
        lin(tokens::SYNTAX_DEFAULT),
        lin(tokens::SYNTAX_KEYWORD),
        lin(tokens::SYNTAX_STRING),
        lin(tokens::SYNTAX_COMMENT),
        lin(tokens::SYNTAX_FUNCTION),
        lin(tokens::SYNTAX_TYPE),
        lin(tokens::SYNTAX_ATTRIBUTE),
        lin(tokens::SYNTAX_CONSTANT),
        lin(tokens::SYNTAX_OPERATOR),
        lin(tokens::SYNTAX_PUNCTUATION),
        lin(tokens::SYNTAX_VARIABLE),
        lin(tokens::SYNTAX_PROPERTY),
        lin(tokens::SYNTAX_LABEL),
        lin(tokens::SYNTAX_ESCAPE),
        lin(tokens::SYNTAX_CONSTRUCTOR),
    ];
    palette.rev = palette.rev.wrapping_add(1);
}

fn refresh_on_theme_changed(
    mut events: MessageReader<ThemeChanged>,
    theme: Res<Theme>,
    palette: ResMut<SyntaxPalette>,
) {
    if events.read().last().is_none() {
        return;
    }
    refresh_syntax_palette(theme, palette);
}

pub struct HighlightPlugin;

impl Plugin for HighlightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SyntaxPalette>()
            .add_systems(Startup, refresh_syntax_palette)
            .add_systems(Update, refresh_on_theme_changed);
    }
}

fn kind_from_capture(name: &str) -> HighlightKind {
    // Match the most specific prefix first.
    if name.starts_with("comment") {
        HighlightKind::Comment
    } else if name.starts_with("string") {
        HighlightKind::String
    } else if name.starts_with("escape") {
        HighlightKind::Escape
    } else if name.starts_with("keyword") {
        HighlightKind::Keyword
    } else if name.starts_with("function") {
        HighlightKind::Function
    } else if name.starts_with("constructor") {
        HighlightKind::Constructor
    } else if name.starts_with("type") {
        HighlightKind::Type
    } else if name.starts_with("attribute") {
        HighlightKind::Attribute
    } else if name.starts_with("constant") {
        HighlightKind::Constant
    } else if name.starts_with("operator") {
        HighlightKind::Operator
    } else if name.starts_with("punctuation") {
        HighlightKind::Punctuation
    } else if name.starts_with("property") {
        HighlightKind::Property
    } else if name.starts_with("label") {
        HighlightKind::Label
    } else if name.starts_with("variable") {
        HighlightKind::Variable
    } else {
        HighlightKind::Default
    }
}

#[derive(Resource)]
pub struct Highlighter {
    parser: Parser,
    query: Query,
    /// Captures from the last successful parse, sorted by `start`.
    /// Later entries overwrite earlier ones byte-by-byte when resolving
    /// overlaps at render time.
    spans: Vec<(Range<usize>, HighlightKind)>,
    /// Last rope we parsed. Used to skip reparse when the doc is
    /// unchanged (selection-only edits, mouse drag, etc.). Comparison
    /// is O(n) but avoids a real reparse which is much heavier.
    last_rope: Option<Rope>,
    /// Monotonic revision bumped on each reparse. Line entities compare
    /// this against their own stored rev to decide when to rebuild
    /// spans.
    pub rev: u64,
}

impl Highlighter {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("tree-sitter-rust language must load");
        let query = Query::new(
            &tree_sitter_rust::LANGUAGE.into(),
            tree_sitter_rust::HIGHLIGHTS_QUERY,
        )
        .expect("tree-sitter-rust HIGHLIGHTS_QUERY must compile");
        Self {
            parser,
            query,
            spans: Vec::new(),
            last_rope: None,
            rev: 0,
        }
    }

    /// Reparse if the rope actually changed. Returns whether spans were
    /// updated — callers use this (plus `rev`) to invalidate cached
    /// per-line render state.
    pub fn maybe_reparse(&mut self, rope: &Rope) -> bool {
        if self.last_rope.as_ref().is_some_and(|r| r == rope) {
            return false;
        }
        // One allocation per doc change. Tree-sitter wants contiguous
        // bytes; ropey's chunk API would let us skip this, but the copy
        // is bounded by doc size and dominated by parse time anyway.
        let source = rope.to_string();
        let Some(tree) = self.parser.parse(&source, None) else {
            // Parser bailed (should only happen on a cancellation flag
            // we don't set). Keep prior spans so rendering doesn't
            // flash uncolored.
            return false;
        };

        let names = self.query.capture_names();
        let mut spans: Vec<(Range<usize>, HighlightKind)> = Vec::new();
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&self.query, tree.root_node(), source.as_bytes());
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let name = names[cap.index as usize];
                let kind = kind_from_capture(name);
                if matches!(kind, HighlightKind::Default) {
                    continue;
                }
                spans.push((cap.node.byte_range(), kind));
            }
        }
        spans.sort_by_key(|(r, _)| r.start);
        self.spans = spans;
        self.last_rope = Some(rope.clone());
        self.rev = self.rev.wrapping_add(1);
        true
    }

    /// Emit `(text, kind)` chunks for a subsection of a single doc
    /// line. `line_text` is the slice actually rendered; `byte_offset`
    /// is how many bytes into the line that slice starts, so a
    /// horizontally-scrolled view can still resolve its captures to
    /// the right colors.
    pub fn line_chunks(
        &self,
        rope: &Rope,
        line_idx: usize,
        byte_offset: usize,
        line_text: &str,
    ) -> Vec<(String, HighlightKind)> {
        let line_start = rope.line_to_byte(line_idx) + byte_offset;
        let len = line_text.len();
        if len == 0 {
            return Vec::new();
        }

        // Paint per-byte, later spans winning. `sort_by_key` above is
        // stable by start; we iterate in that order so longer / later
        // captures overwrite earlier coarse ones, mirroring the
        // convention of tree-sitter-rust's highlights.scm.
        let mut kinds = vec![HighlightKind::Default; len];
        // `Vec` is sorted by start; we could binary-search for the
        // first relevant span, but the list is small (one entry per
        // capture in the doc) and lines are scanned only on change.
        let line_end = line_start + len;
        for (range, kind) in &self.spans {
            if range.end <= line_start {
                continue;
            }
            if range.start >= line_end {
                break;
            }
            let lo = range.start.max(line_start) - line_start;
            let hi = range.end.min(line_end) - line_start;
            for k in &mut kinds[lo..hi] {
                *k = *kind;
            }
        }

        // Run-length encode, extending each run to a UTF-8 char
        // boundary so we never split a multi-byte sequence between
        // spans.
        let mut out: Vec<(String, HighlightKind)> = Vec::new();
        let mut i = 0;
        while i < len {
            let kind = kinds[i];
            let mut j = i + 1;
            while j < len && kinds[j] == kind {
                j += 1;
            }
            while j < len && !line_text.is_char_boundary(j) {
                j += 1;
            }
            out.push((line_text[i..j].to_string(), kind));
            i = j;
        }
        out
    }
}
