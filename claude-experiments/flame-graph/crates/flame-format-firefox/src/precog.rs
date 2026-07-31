//! samply's `.syms.json` presymbolication sidecar.
//!
//! `samply record --unstable-presymbolicate -o profile.json` writes a second
//! file next to the profile — `profile.syms.json` — holding every symbol that
//! the profile's frame addresses resolve to. The profile itself stays
//! unsymbolicated: its `funcTable` names are hex RVAs (`"0x1a23f"`) and
//! `meta.symbolicated` is `false`, because samply expects the front-end to ask
//! a symbol server for names. We have no symbol server, so without the sidecar
//! every frame of a saved samply profile reads as a bare address.
//!
//! Schema mirror of `samply/src/shared/symbol_precog.rs`.
//!
//! ```text
//! { "string_table": ["…", "inl::outer_hot", …],
//!   "data": [ { "debug_name": "inl",
//!               "debug_id": "77dccb24-c9ea-386d-a2ba-5d14c5ee7e62",
//!               "code_id": "77DCCB24C9EA386DA2BA5D14C5EE7E62",
//!               "symbol_table": [ { "rva": 2284, "size": 84, "symbol": 7,
//!                                   "frames": [ {"function": 7, "file": 3, "line": 19} ] } ],
//!               "known_addresses": [ [2332, 0], [2348, 0] ] } ] }
//! ```
//!
//! `known_addresses` is the authoritative lookup: it maps each address the
//! companion profile references to an index into that lib's `symbol_table`.
//! It is per-address rather than per-function because two addresses inside one
//! function can carry different inline stacks, so `symbol_table` may hold
//! several entries with the same `rva`.

use std::collections::HashMap;

use flame_core::LoadError;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct SidecarJson {
    #[serde(default)]
    string_table: Vec<String>,
    #[serde(default)]
    data: Vec<LibJson>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct LibJson {
    debug_name: String,
    debug_id: String,
    code_id: String,
    symbol_table: Vec<SymbolJson>,
    /// `(rva, index into symbol_table)`, sorted ascending by rva.
    known_addresses: Vec<(u32, usize)>,
}

#[derive(Deserialize, Debug)]
struct SymbolJson {
    /// Start of the containing function, relative to the lib's load base.
    rva: u32,
    /// Function length. Absent when the symbol source didn't know it.
    #[serde(default)]
    size: Option<u32>,
    /// Index into the sidecar's `string_table`.
    symbol: usize,
    /// The inline stack at this address: callee-most inlinee first, outer
    /// function last. Absent when the lib had no debug info (symbol table
    /// only), which is the common case for system dylibs.
    #[serde(default)]
    frames: Option<Vec<FrameJson>>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct FrameJson {
    function: Option<usize>,
    file: Option<usize>,
    line: Option<u32>,
}

/// One resolved frame. A single machine address yields several of these when
/// the compiler inlined calls at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymFrame {
    pub name: String,
    pub file: String,
    /// Line inside `file` for this address, or 0 when unknown.
    pub line: u32,
}

/// Symbols for one lib, in lookup-ready form.
#[derive(Debug, Default)]
struct LibSymbols {
    /// Human-readable name, for logging.
    debug_name: String,
    /// One entry per `symbol_table` row, outermost frame first — i.e. already
    /// in the order a flame graph stacks them.
    stacks: Vec<Vec<SymFrame>>,
    /// `known_addresses` as a map into `stacks`.
    addresses: HashMap<u32, usize>,
    /// `(start, end_exclusive, stacks index)` sorted by start, for addresses
    /// that aren't in `known_addresses`. Only rows with a known `size` land
    /// here, so a hit means the address really is inside that function.
    ranges: Vec<(u32, u32, usize)>,
}

/// A parsed sidecar, keyed so a profile's `libs[]` entry can find its symbols.
#[derive(Debug, Default)]
pub struct PrecogSymbols {
    libs: Vec<LibSymbols>,
    /// Normalized debug id (32 hex chars, uppercase) → index into `libs`.
    by_debug_id: HashMap<String, usize>,
    /// Uppercased code id → index into `libs`. Second key because a profile
    /// entry may carry only one of the two.
    by_code_id: HashMap<String, usize>,
}

impl PrecogSymbols {
    /// Parse a `.syms.json` sidecar.
    pub fn parse(input: &[u8]) -> Result<Self, LoadError> {
        let json: SidecarJson = serde_json::from_slice(input)
            .map_err(|e| LoadError::Parse(format!("samply syms.json: {e}")))?;
        let strings = &json.string_table;
        let string_at = |idx: Option<usize>| -> String {
            idx.and_then(|i| strings.get(i)).cloned().unwrap_or_default()
        };

        let mut out = Self::default();
        for lib in &json.data {
            let mut sym = LibSymbols {
                debug_name: lib.debug_name.clone(),
                ..Default::default()
            };
            for entry in &lib.symbol_table {
                let stack: Vec<SymFrame> = match &entry.frames {
                    // Innermost-first on disk; reverse so the outer function
                    // leads and inlinees hang off it, which is the order the
                    // flame graph pushes them in.
                    Some(frames) if !frames.is_empty() => frames
                        .iter()
                        .rev()
                        .map(|f| SymFrame {
                            name: match string_at(f.function) {
                                n if n.is_empty() => string_at(Some(entry.symbol)),
                                n => n,
                            },
                            file: string_at(f.file),
                            line: f.line.unwrap_or(0),
                        })
                        .collect(),
                    _ => vec![SymFrame {
                        name: string_at(Some(entry.symbol)),
                        file: String::new(),
                        line: 0,
                    }],
                };
                if let Some(size) = entry.size {
                    if size > 0 {
                        sym.ranges.push((
                            entry.rva,
                            entry.rva.saturating_add(size),
                            sym.stacks.len(),
                        ));
                    }
                }
                sym.stacks.push(stack);
            }
            for &(rva, idx) in &lib.known_addresses {
                if idx < sym.stacks.len() {
                    sym.addresses.insert(rva, idx);
                }
            }
            sym.ranges.sort_by_key(|&(start, _, _)| start);

            let idx = out.libs.len();
            if let Some(key) = normalize_debug_id(&lib.debug_id) {
                out.by_debug_id.entry(key).or_insert(idx);
            }
            if !lib.code_id.is_empty() {
                out.by_code_id
                    .entry(lib.code_id.to_ascii_uppercase())
                    .or_insert(idx);
            }
            out.libs.push(sym);
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.libs.is_empty()
    }

    /// Number of libs the sidecar carries symbols for.
    pub fn lib_count(&self) -> usize {
        self.libs.len()
    }

    /// Resolve `rva` inside the lib identified by `debug_id` (a Breakpad id or
    /// a dashed uuid) and/or `code_id`. Returns the frames outermost-first, or
    /// `None` when the lib or the address isn't covered.
    pub fn lookup(
        &self,
        debug_id: Option<&str>,
        code_id: Option<&str>,
        rva: u32,
    ) -> Option<&[SymFrame]> {
        let idx = self.lib_index(debug_id, code_id)?;
        let lib = &self.libs[idx];
        if let Some(&stack) = lib.addresses.get(&rva) {
            return Some(&lib.stacks[stack]);
        }
        // Not an address the sidecar was generated for. Fall back to the
        // function ranges: the sidecar and the profile are normally a matched
        // pair, so this only fires when they were mixed by hand.
        let pos = lib.ranges.partition_point(|&(start, _, _)| start <= rva);
        let &(_, end, stack) = lib.ranges.get(pos.checked_sub(1)?)?;
        if rva < end {
            Some(&lib.stacks[stack])
        } else {
            None
        }
    }

    /// The `debug_name` of the lib a lookup key resolves to, for logging.
    pub fn lib_name(&self, debug_id: Option<&str>, code_id: Option<&str>) -> Option<&str> {
        let idx = self.lib_index(debug_id, code_id)?;
        Some(self.libs[idx].debug_name.as_str())
    }

    fn lib_index(&self, debug_id: Option<&str>, code_id: Option<&str>) -> Option<usize> {
        if let Some(id) = debug_id.and_then(normalize_debug_id) {
            if let Some(&idx) = self.by_debug_id.get(&id) {
                return Some(idx);
            }
        }
        let code = code_id?;
        if code.is_empty() {
            return None;
        }
        self.by_code_id.get(&code.to_ascii_uppercase()).copied()
    }
}

/// Reduce a debug id to a comparable key. The profile stores a Breakpad id —
/// 32 uppercase hex digits plus an age suffix (`"77DCCB24…7E620"`); the
/// sidecar stores a dashed uuid (`"77dccb24-c9ea-…"`). Strip separators,
/// uppercase, keep the leading 32 hex digits.
fn normalize_debug_id(id: &str) -> Option<String> {
    let hex: String = id
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .map(|c| c.to_ascii_uppercase())
        .take(32)
        .collect();
    if hex.len() == 32 {
        Some(hex)
    } else {
        None
    }
}
