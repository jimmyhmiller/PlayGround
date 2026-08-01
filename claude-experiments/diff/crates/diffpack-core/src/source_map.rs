//! Honest source-map plumbing: a per-module map produced by the Oxc printer, and
//! the line/column bookkeeping that carries it, without ever guessing, through
//! the text rewrites between a module's lowered code and the bytes of the chunk
//! it lands in.
//!
//! The one rule this module exists to enforce: **a position that cannot be
//! resolved honestly is left UNMAPPED**. A gap in a source map is a legal,
//! meaningful state — a debugger shows the generated line and says nothing about
//! its origin. A *wrong* mapping is worse than no mapping at all: the debugger
//! jumps confidently into unrelated source and the developer reads the wrong
//! code. So every transformation here either accounts for the text it changed
//! exactly, or drops the tokens it can no longer vouch for.

use std::collections::HashMap;
use std::sync::Arc;

use crate::module_graph::DenseModuleId;

/// One mapping from a position in generated text to the position it was printed
/// from. Columns are UTF-16 code units — the unit the source-map spec defines and
/// the unit Oxc's codegen emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapToken {
    pub generated_line: u32,
    pub generated_column: u32,
    pub source_line: u32,
    pub source_column: u32,
    /// Index into [`ModuleSourceMap::names`] — the ORIGINAL identifier at this
    /// position, recorded only where the generated text renamed it.
    pub name: Option<u32>,
}

pub trait ModuleMapLookup {
    fn module_map(&self, module: DenseModuleId) -> Option<(&ModuleSourceMap, &Arc<str>)>;
}

pub struct ResolvedMinifiedToken<'a> {
    pub dense: DenseModuleId,
    pub source_line: u32,
    pub source_column: u32,
    pub name: Option<&'a str>,
}

/// Resolves one minifier token through the readable chunk map into an original
/// module position. `None` is an honestly unmapped bundler-generated region.
pub fn resolve_minified_token<'a>(
    modules: &'a impl ModuleMapLookup,
    minified: &oxc_sourcemap::Token,
    minified_map: &'a oxc_sourcemap::SourceMap,
    readable: &[(MapToken, DenseModuleId)],
    hint: &mut usize,
    source_lines: &mut HashMap<DenseModuleId, Vec<usize>>,
) -> Option<ResolvedMinifiedToken<'a>> {
    let position = (minified.get_src_line(), minified.get_src_col());
    let candidate = partition_point_from_hint(readable, position, *hint);
    *hint = candidate;
    if candidate == 0 {
        return None;
    }
    let (token, dense) = &readable[candidate - 1];
    if token.generated_line != minified.get_src_line() {
        return None;
    }
    let (map, module_source) = modules.module_map(*dense)?;
    let name = match token
        .name
        .and_then(|index| map.names().get(index as usize))
        .filter(|name| is_identifier(name))
    {
        Some(name) => Some(name.as_str()),
        None => minified
            .get_name_id()
            .and_then(|index| minified_map.get_name(index))
            .filter(|candidate| {
                let text = map.source_text(module_source);
                let lines = source_lines
                    .entry(*dense)
                    .or_insert_with(|| line_starts(text));
                identifier_at(text, lines, token.source_line, token.source_column)
                    == Some(*candidate)
            }),
    };
    Some(ResolvedMinifiedToken {
        dense: *dense,
        source_line: token.source_line,
        source_column: token.source_column,
        name,
    })
}

/// Which TEXT a module map's source positions refer to.
///
/// Several stages hand the parser source that is not the bytes on disk: component
/// compilation, macro expansion, virtual splitting, and compatibility rewrites. For those modules
/// the printer's positions describe the REWRITTEN text, so claiming the file on
/// disk would be exactly the class of lie this module removes.
///
/// diffpack labels them instead: the map's `sources` entry names the rewrite and
/// its `sourcesContent` is the rewritten text itself, so a position always refers
/// to the text sitting next to it in the map and a reader cannot be misled about
/// which text that is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapOrigin {
    /// The module's own file, exactly as it is on disk.
    File,
    /// A source diffpack generated from that file. The label names the stage.
    Generated(&'static str),
}

impl MapOrigin {
    /// Narrow `File` to a generated variant; a map that is ALREADY generated keeps
    /// the first (innermost) stage that rewrote it, which is the one whose output
    /// the positions actually index.
    pub fn or_generated(self, stage: &'static str) -> Self {
        match self {
            MapOrigin::File => MapOrigin::Generated(stage),
            already => already,
        }
    }
}

/// A source map over ONE module's generated text, produced by the Oxc printer and
/// therefore real: every token is a position the printer actually emitted, paired
/// with the span of the AST node it printed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleSourceMap {
    origin: MapOrigin,
    /// The exact text the tokens' `source_line`/`source_column` index into, when
    /// that is NOT the module's own stored source (i.e. when a stage generated
    /// it). `None` means the positions index the module's source as the bundler
    /// holds it, which is then what gets inlined as `sourcesContent`. Either way
    /// the map's content is the text its positions were measured against.
    generated_source: Option<Arc<str>>,
    names: Vec<String>,
    /// Sorted by `(generated_line, generated_column)`.
    tokens: Vec<MapToken>,
    /// `line_index[l]` is the offset in `tokens` of the first token on generated
    /// line `l`; `line_index[l + 1]` is one past its last. Built once so a chunk
    /// render can slice a line's tokens without searching.
    line_index: Vec<u32>,
}

impl ModuleSourceMap {
    /// Build a map from tokens the Oxc printer emitted. `tokens` need not be
    /// sorted; `generated_lines` is the line count of the text they describe.
    pub fn new(
        origin: MapOrigin,
        generated_source: Option<Arc<str>>,
        names: Vec<String>,
        mut tokens: Vec<MapToken>,
        generated_lines: usize,
    ) -> Self {
        tokens.sort_by_key(|token| (token.generated_line, token.generated_column));
        tokens.dedup_by_key(|token| (token.generated_line, token.generated_column));
        let mut line_index = vec![0_u32; generated_lines + 1];
        for token in &tokens {
            let line = token.generated_line as usize;
            if line + 1 < line_index.len() {
                line_index[line + 1] += 1;
            }
        }
        for index in 1..line_index.len() {
            line_index[index] += line_index[index - 1];
        }
        Self {
            origin,
            generated_source,
            names,
            tokens,
            line_index,
        }
    }

    pub fn origin(&self) -> MapOrigin {
        self.origin
    }

    /// Record that the module's stored source is itself something diffpack
    /// generated (a compiled component or compatibility rewrite),
    /// so the map's label says so and cannot be read as the file on disk.
    pub fn mark_generated(&mut self, stage: &'static str) {
        self.origin = self.origin.or_generated(stage);
    }

    /// The exact text this map's positions were measured against: the generated
    /// source when a stage produced one, otherwise the module's own source.
    pub fn source_text<'a>(&'a self, module_source: &'a Arc<str>) -> &'a Arc<str> {
        self.generated_source.as_ref().unwrap_or(module_source)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn tokens(&self) -> &[MapToken] {
        &self.tokens
    }

    /// Move this map onto a NEW version of the same generated text, described by
    /// `track`. Whatever `track` could not account for is dropped, so the map
    /// stays true to the text it now describes.
    pub fn rebase(&mut self, track: &LineTrack, generated_lines: usize) {
        let mut tokens = Vec::with_capacity(self.tokens.len());
        track.project(self, 0, &mut tokens);
        *self = Self::new(
            self.origin,
            self.generated_source.clone(),
            std::mem::take(&mut self.names),
            tokens,
            generated_lines,
        );
    }

    /// The tokens on one generated line of the text this map describes. An
    /// out-of-range line has none — never a guessed one.
    pub fn line_tokens(&self, line: u32) -> &[MapToken] {
        let line = line as usize;
        if line + 1 >= self.line_index.len() {
            return &[];
        }
        let start = self.line_index[line] as usize;
        let end = self.line_index[line + 1] as usize;
        &self.tokens[start..end]
    }
}

/// One in-place text rewrite applied to a single line, in UTF-16 columns.
///
/// A token that fell INSIDE `[column, column + removed)` no longer describes any
/// text that survives, so it is dropped; a token after the rewrite keeps its
/// meaning and is shifted by `inserted - removed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnEdit {
    pub column: u32,
    pub removed: u32,
    pub inserted: u32,
}

/// Where each line of a piece of generated text came from in the text a
/// [`ModuleSourceMap`] describes.
///
/// Every stage between the lowered module and the emitted chunk (the flat-module
/// derivation, the export/import shake, the dynamic-import rewrite, the chunk
/// concatenation) either keeps a line verbatim, deletes it, or rewrites part of
/// it. A [`LineTrack`] records exactly that, so the module's real map can be
/// projected onto the chunk's bytes without a single guessed position.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineTrack {
    lines: Vec<LineOrigin>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineOrigin {
    /// Index of this line in the text the module map describes, or `None` when
    /// the line is bundler-synthesized and has no origin at all.
    pub source_line: Option<u32>,
    /// In-place rewrites applied to this line, in increasing column order.
    pub edits: Vec<ColumnEdit>,
}

impl LineOrigin {
    fn verbatim(source_line: u32) -> Self {
        Self {
            source_line: Some(source_line),
            edits: Vec::new(),
        }
    }

    /// The column a token at `column` moved to, or `None` when the rewrite
    /// consumed it (so the token can no longer be vouched for).
    ///
    /// A token exactly AT a rewrite's start survives: the replacement text begins
    /// there and stands for the same construct (a rewritten import binding, a
    /// rewritten dynamic-import specifier). A token strictly INSIDE the replaced
    /// span describes text that no longer exists and is dropped.
    fn remap_column(&self, column: u32) -> Option<u32> {
        let mut shifted = i64::from(column);
        for edit in &self.edits {
            if column <= edit.column {
                break;
            }
            if column < edit.column + edit.removed {
                return None;
            }
            shifted += i64::from(edit.inserted) - i64::from(edit.removed);
        }
        u32::try_from(shifted).ok()
    }
}

impl LineTrack {
    /// A track over text that is exactly the text the module map describes.
    pub fn identity(lines: usize) -> Self {
        Self {
            lines: (0..lines)
                .map(|line| LineOrigin::verbatim(line as u32))
                .collect(),
        }
    }

    /// A track over `lines` lines of bundler-synthesized text with no origin.
    pub fn synthetic(lines: usize) -> Self {
        Self {
            lines: (0..lines)
                .map(|_| LineOrigin {
                    source_line: None,
                    edits: Vec::new(),
                })
                .collect(),
        }
    }

    /// The provenance of one line, for tests and diagnostics.
    pub fn line(&self, index: usize) -> Option<&LineOrigin> {
        self.lines.get(index)
    }

    pub fn push(&mut self, origin: LineOrigin) {
        self.lines.push(origin);
    }

    pub fn extend(&mut self, other: LineTrack) {
        self.lines.extend(other.lines);
    }

    /// A new track over the subsequence of lines `keep` selects (in order). Used
    /// by every stage that deletes whole lines.
    pub fn keep(&self, keep: impl Iterator<Item = usize>) -> Self {
        Self {
            lines: keep
                .map(|index| {
                    self.lines.get(index).cloned().unwrap_or(LineOrigin {
                        source_line: None,
                        edits: Vec::new(),
                    })
                })
                .collect(),
        }
    }

    /// Record an in-place rewrite of `line`. Edits must be recorded in increasing
    /// column order, which is the order every rewrite here applies them in.
    pub fn record_edit(&mut self, line: usize, edit: ColumnEdit) {
        if let Some(origin) = self.lines.get_mut(line) {
            origin.edits.push(edit);
        }
    }

    /// Give up on a line: whatever it says now, this track can no longer prove
    /// where any of it came from, so nothing on it will be mapped.
    pub fn invalidate(&mut self, line: usize) {
        if let Some(origin) = self.lines.get_mut(line) {
            origin.source_line = None;
            origin.edits.clear();
        }
    }

    /// Chain two tracks: `self` describes the CURRENT text in terms of `base`'s
    /// text, and `base` describes that in terms of the module map's text. The
    /// result describes the current text directly in terms of the map.
    ///
    /// Column edits from both stages are expressed in DIFFERENT coordinate
    /// systems (a later stage measured columns in text an earlier stage had
    /// already rewritten). Translating one into the other is only well defined
    /// while no edit lands inside another's insertion, so a line that both stages
    /// rewrote is left UNMAPPED rather than resolved by an assumption.
    pub fn compose(&self, base: &LineTrack) -> Self {
        Self {
            lines: self
                .lines
                .iter()
                .map(|origin| {
                    let Some(middle) = origin.source_line else {
                        return LineOrigin {
                            source_line: None,
                            edits: Vec::new(),
                        };
                    };
                    let Some(under) = base.lines.get(middle as usize) else {
                        return LineOrigin {
                            source_line: None,
                            edits: Vec::new(),
                        };
                    };
                    match (under.edits.is_empty(), origin.edits.is_empty()) {
                        (_, true) => under.clone(),
                        // No earlier rewrite, so this stage's columns already are
                        // the map's columns.
                        (true, false) => LineOrigin {
                            source_line: under.source_line,
                            edits: origin.edits.clone(),
                        },
                        (false, false) => LineOrigin {
                            source_line: None,
                            edits: Vec::new(),
                        },
                    }
                })
                .collect(),
        }
    }

    /// Drop the mapping of every line whose text changed between `before` and
    /// `after`. Used for the in-place rewrites the chunk render applies to
    /// bundler glue lines: they never move a line, so an unchanged line is
    /// provably still the text the map describes and a changed one is not.
    /// Differing line COUNTS mean the rewrite did something this cannot account
    /// for, and the whole track is given up.
    pub fn invalidate_changed_lines(&mut self, before: &str, after: &str) {
        let mut before_lines = before.lines();
        let mut after_lines = after.lines();
        let mut index = 0;
        loop {
            match (before_lines.next(), after_lines.next()) {
                (Some(old), Some(new)) => {
                    if old != new {
                        self.invalidate(index);
                    }
                    index += 1;
                }
                (None, None) => break,
                _ => {
                    for line in 0..self.lines.len() {
                        self.invalidate(line);
                    }
                    break;
                }
            }
        }
    }

    /// Project a module's real map onto the text this track describes, shifting
    /// every generated position by `line_offset` (the line the text starts at in
    /// the chunk). Tokens on deleted or rewritten-over text are dropped.
    pub fn project(&self, map: &ModuleSourceMap, line_offset: u32, out: &mut Vec<MapToken>) {
        for (index, origin) in self.lines.iter().enumerate() {
            let Some(source_line) = origin.source_line else {
                continue;
            };
            let generated_line = line_offset + index as u32;
            for token in map.line_tokens(source_line) {
                let Some(generated_column) = origin.remap_column(token.generated_column) else {
                    continue;
                };
                out.push(MapToken {
                    generated_line,
                    generated_column,
                    source_line: token.source_line,
                    source_column: token.source_column,
                    name: token.name,
                });
            }
        }
    }
}

/// The number of UTF-16 code units in `text` — the source-map column unit.
pub fn utf16_len(text: &str) -> u32 {
    text.chars()
        .map(|character| character.len_utf16() as u32)
        .sum()
}

/// Finds the token partition point by expanding from a caller-supplied hint.
pub fn partition_point_from_hint<T>(
    readable: &[(MapToken, T)],
    position: (u32, u32),
    hint: usize,
) -> usize {
    let at_or_before = |index: usize| {
        let token = &readable[index].0;
        (token.generated_line, token.generated_column) <= position
    };
    let length = readable.len();
    if length == 0 {
        return 0;
    }
    let anchor = hint.min(length - 1);
    let (mut low, mut high);
    if at_or_before(anchor) {
        low = anchor + 1;
        high = length;
        let mut width = 1;
        while anchor + width < length {
            let probe = anchor + width;
            if at_or_before(probe) {
                low = probe + 1;
                width *= 2;
            } else {
                high = probe;
                break;
            }
        }
    } else {
        low = 0;
        high = anchor;
        let mut width = 1;
        while width <= anchor {
            let probe = anchor - width;
            if at_or_before(probe) {
                low = probe + 1;
                break;
            }
            high = probe;
            width *= 2;
        }
    }
    while low < high {
        let middle = low + (high - low) / 2;
        if at_or_before(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

pub fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_alphabetic() || first == '_' || first == '$')
        && characters
            .all(|character| character.is_alphanumeric() || character == '_' || character == '$')
}

pub fn line_count(text: &str) -> u32 {
    text.lines().count() as u32
}

pub fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(text.match_indices('\n').map(|(index, _)| index + 1));
    starts
}

/// Returns the JavaScript identifier beginning at a UTF-16 source position.
pub fn identifier_at<'a>(
    text: &'a str,
    starts: &[usize],
    line: u32,
    column: u32,
) -> Option<&'a str> {
    let start = *starts.get(line as usize)?;
    let end = starts
        .get(line as usize + 1)
        .map_or(text.len(), |next| next - 1);
    let line_text = text.get(start..end)?;
    let mut offset = 0;
    let mut units = 0_u32;
    for character in line_text.chars() {
        if units == column {
            break;
        }
        units += character.len_utf16() as u32;
        offset += character.len_utf8();
    }
    if units != column {
        return None;
    }
    let rest = &line_text[offset..];
    let is_start =
        |character: char| character.is_ascii_alphabetic() || character == '_' || character == '$';
    let is_part =
        |character: char| character.is_ascii_alphanumeric() || character == '_' || character == '$';
    if !rest.starts_with(is_start) {
        return None;
    }
    let length = rest
        .find(|character: char| !is_part(character))
        .unwrap_or(rest.len());
    Some(&rest[..length])
}

/// Replace every occurrence of `needle` with `replacement`, line by line, and
/// record each replacement as a [`ColumnEdit`] on the line it happened on so a
/// map over the ORIGINAL text stays truthful. `needle` must not contain a
/// newline (every caller's needle is a single-line token), which is what makes
/// this equivalent to `str::replace` over the whole text.
///
/// Returns `None` when nothing matched, so a caller can keep borrowing the input.
pub fn replace_tracked(
    text: &str,
    needle: &str,
    replacement: &str,
    track: &mut LineTrack,
) -> Option<String> {
    debug_assert!(
        !needle.contains('\n'),
        "replace_tracked needles are single-line"
    );
    if needle.is_empty() || !text.contains(needle) {
        return None;
    }
    let removed = utf16_len(needle);
    let inserted = utf16_len(replacement);
    let mut output = String::with_capacity(text.len());
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if !line.contains(needle) {
            output.push_str(line);
            continue;
        }
        let mut rest = line;
        let mut column = 0_u32;
        while let Some(position) = rest.find(needle) {
            let (before, after) = rest.split_at(position);
            output.push_str(before);
            output.push_str(replacement);
            column += utf16_len(before);
            track.record_edit(
                index,
                ColumnEdit {
                    column,
                    removed,
                    inserted,
                },
            );
            column += removed;
            rest = &after[needle.len()..];
        }
        output.push_str(rest);
    }
    Some(output)
}

/// Replace every occurrence of any of `pairs`' needles with its replacement, in
/// ONE left-to-right pass, recording each replacement as a [`ColumnEdit`] on the
/// line it happened on.
///
/// The single pass is a CORRECTNESS requirement, not an optimization. A
/// [`LineTrack`]'s edits are all measured against the one text the module map
/// describes, and [`LineOrigin::remap_column`] reads them in increasing column
/// order (see [`LineTrack::record_edit`]). Applying the replacements as N
/// successive passes measures pass 2's columns in the text pass 1 already
/// rewrote, so a single edit list ends up holding two coordinate systems — which
/// does not fail loudly, it silently produces plausible-looking WRONG columns
/// (and segments out of generated-column order, which the source-map format does
/// not allow). Scanning once means every edit is measured against the same text.
///
/// Needles must not contain a newline (every caller's needle is a single-line
/// token). Where several needles match at the same position the LONGEST wins, so
/// a needle that is a prefix of another can never eat the longer match. Returns
/// `None` when nothing matched, so a caller can keep borrowing the input.
pub fn replace_many_tracked(
    text: &str,
    pairs: &[(String, String)],
    track: &mut LineTrack,
) -> Option<String> {
    let present: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(needle, replacement)| (needle.as_str(), replacement.as_str()))
        .filter(|(needle, _)| !needle.is_empty() && text.contains(needle))
        .collect();
    if present.is_empty() {
        return None;
    }
    debug_assert!(
        present.iter().all(|(needle, _)| !needle.contains('\n')),
        "replace_many_tracked needles are single-line"
    );
    let mut output = String::with_capacity(text.len());
    for (index, line) in text.split_inclusive('\n').enumerate() {
        let on_line: Vec<(&str, &str)> = present
            .iter()
            .copied()
            .filter(|(needle, _)| line.contains(needle))
            .collect();
        if on_line.is_empty() {
            output.push_str(line);
            continue;
        }
        let mut rest = line;
        let mut column = 0_u32;
        while let Some((at, needle, replacement)) = on_line
            .iter()
            .filter_map(|(needle, replacement)| {
                rest.find(needle).map(|at| (at, *needle, *replacement))
            })
            .min_by_key(|(at, needle, _)| (*at, std::cmp::Reverse(needle.len())))
        {
            let (before, after) = rest.split_at(at);
            output.push_str(before);
            output.push_str(replacement);
            column += utf16_len(before);
            track.record_edit(
                index,
                ColumnEdit {
                    column,
                    removed: utf16_len(needle),
                    inserted: utf16_len(replacement),
                },
            );
            column += utf16_len(needle);
            rest = &after[needle.len()..];
        }
        output.push_str(rest);
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_of(tokens: Vec<MapToken>, lines: usize) -> ModuleSourceMap {
        ModuleSourceMap::new(MapOrigin::File, None, Vec::new(), tokens, lines)
    }

    fn token(generated_line: u32, generated_column: u32, source_line: u32) -> MapToken {
        MapToken {
            generated_line,
            generated_column,
            source_line,
            source_column: 0,
            name: None,
        }
    }

    #[test]
    fn deleting_lines_renumbers_the_survivors_and_drops_the_rest() {
        let map = map_of(vec![token(0, 0, 10), token(1, 0, 11), token(2, 0, 12)], 3);
        let track = LineTrack::identity(3).keep([0, 2].into_iter());
        let mut tokens = Vec::new();
        track.project(&map, 100, &mut tokens);
        assert_eq!(
            tokens,
            vec![token(100, 0, 10), token(101, 0, 12)],
            "a deleted line contributes nothing and never shifts another line's origin"
        );
    }

    #[test]
    fn a_token_inside_rewritten_text_is_dropped_and_a_later_one_is_shifted() {
        // `__import(ns, "x")` (16 columns) replaced by `x` (1 column) at column 4.
        let map = map_of(
            vec![
                token(0, 0, 5),
                token(0, 4, 5),
                token(0, 8, 5),
                token(0, 20, 5),
            ],
            1,
        );
        let mut track = LineTrack::identity(1);
        track.record_edit(
            0,
            ColumnEdit {
                column: 4,
                removed: 16,
                inserted: 1,
            },
        );
        let mut tokens = Vec::new();
        track.project(&map, 0, &mut tokens);
        assert_eq!(
            tokens,
            vec![token(0, 0, 5), token(0, 4, 5), token(0, 5, 5)],
            "the token AT the rewrite start stands for the replacement; the one at column 8 sat \
             inside the replaced text and is dropped; the one at 20 shifts by -15"
        );
    }

    #[test]
    fn replace_tracked_matches_str_replace_and_records_every_edit() {
        let text = "a __import(n, \"x\") b\nplain\nc __import(n, \"x\") __import(n, \"x\")\n";
        let mut track = LineTrack::identity(3);
        let replaced = replace_tracked(text, "__import(n, \"x\")", "x", &mut track)
            .expect("the needle is present");
        assert_eq!(replaced, text.replace("__import(n, \"x\")", "x"));
        assert_eq!(track.line(0).expect("line 0").edits.len(), 1);
        assert!(track.line(1).expect("line 1").edits.is_empty());
        assert_eq!(track.line(2).expect("line 2").edits.len(), 2);
        // The second edit on line 2 is recorded at its column in the ORIGINAL line.
        assert_eq!(track.line(2).expect("line 2").edits[1].column, 19);
    }

    #[test]
    fn several_replacements_on_one_line_all_land_in_the_map_s_own_coordinates() {
        // Two DIFFERENT bindings rewritten on the same line — the shape that made
        // a per-replacement pass record its columns in the previous pass's text.
        let text = "console.log(__import(n0, \"alpha\"), __import(n1, \"beta\"), __import(n0, \"alpha\"));\n";
        let pairs = vec![
            ("__import(n0, \"alpha\")".to_string(), "alpha".to_string()),
            ("__import(n1, \"beta\")".to_string(), "beta".to_string()),
        ];
        let mut track = LineTrack::identity(1);
        let replaced =
            replace_many_tracked(text, &pairs, &mut track).expect("both needles are present");
        assert_eq!(
            replaced,
            text.replace("__import(n0, \"alpha\")", "alpha")
                .replace("__import(n1, \"beta\")", "beta"),
            "one pass must produce exactly what the successive replaces produced — only the \
             bookkeeping changes, never the bytes"
        );
        let edits = &track.line(0).expect("line 0").edits;
        assert_eq!(
            edits.iter().map(|edit| edit.column).collect::<Vec<_>>(),
            vec![12, 35, 57],
            "every edit is recorded at its column in the ORIGINAL line, in increasing order — \
             the precondition `remap_column` reads them under"
        );

        // A token on each of the three call arguments, in the ORIGINAL text.
        let map = map_of(vec![token(0, 12, 9), token(0, 35, 9), token(0, 57, 9)], 1);
        let mut tokens = Vec::new();
        track.project(&map, 0, &mut tokens);
        let columns = tokens
            .iter()
            .map(|token| token.generated_column)
            .collect::<Vec<_>>();
        assert_eq!(
            columns,
            vec![12, 19, 25],
            "each argument must land on the column it really occupies in the rewritten line"
        );
        for (token, column) in tokens.iter().zip(&columns) {
            let line_length = utf16_len(replaced.lines().next().expect("one line"));
            assert!(
                *column <= line_length,
                "a mapped column must exist on the line it names ({column} > {line_length}): \
                 {token:?}"
            );
        }
        assert!(
            columns.windows(2).all(|pair| pair[0] < pair[1]),
            "segments must stay in increasing generated-column order, got {columns:?}"
        );
    }

    #[test]
    fn an_invalidated_line_maps_nothing() {
        let map = map_of(vec![token(0, 0, 7)], 1);
        let mut track = LineTrack::identity(1);
        track.invalidate(0);
        let mut tokens = Vec::new();
        track.project(&map, 0, &mut tokens);
        assert!(
            tokens.is_empty(),
            "a line whose rewrite could not be accounted for is UNMAPPED, never guessed"
        );
    }

    #[test]
    fn hinted_partition_matches_the_standard_partition_for_every_hint() {
        let readable: Vec<(MapToken, ())> = [(0, 0), (0, 4), (0, 4), (1, 0), (3, 2), (7, 1)]
            .into_iter()
            .map(|(line, column)| {
                (
                    MapToken {
                        generated_line: line,
                        generated_column: column,
                        source_line: 0,
                        source_column: 0,
                        name: None,
                    },
                    (),
                )
            })
            .collect();
        for line in 0..9 {
            for column in 0..11 {
                let position = (line, column);
                let expected = readable.partition_point(|(token, _)| {
                    (token.generated_line, token.generated_column) <= position
                });
                for hint in 0..=readable.len() + 2 {
                    assert_eq!(
                        partition_point_from_hint(&readable, position, hint),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn identifier_lookup_uses_utf16_columns() {
        let text = "😀 value\nnext";
        let starts = line_starts(text);
        assert_eq!(identifier_at(text, &starts, 0, 3), Some("value"));
        assert_eq!(identifier_at(text, &starts, 0, 1), None);
        assert_eq!(identifier_at(text, &starts, 1, 0), Some("next"));
        assert_eq!(line_count(text), 2);
    }
}
