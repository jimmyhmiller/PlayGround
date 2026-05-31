//! Persistent, per-branch todo log for the structural-editing layer.
//!
//! Background: a type-changing `update` can break dependents. Those breakages
//! come back as [`typecheck::Todo`]s inline in the write response. That is fine
//! while the agent stays connected, but an agent that gets interrupted has no
//! way to later ask "what's still broken?" — the inline todos are gone. This
//! module closes that gap with an on-disk, queryable worklist.
//!
//! ## On-disk format & scoping
//!
//! Todos are scoped *per branch*: a fix on one branch must not silently clear a
//! breakage that still exists on another. Each branch gets its own file:
//!
//! ```text
//! <codebase_root>/todos/<branch>.json
//! ```
//!
//! The file is a small hand-rolled JSON array of entry objects (so it is
//! human-inspectable and round-trips exactly). An absent file means "no
//! outstanding todos" — the empty store. The store is treated as a cache that
//! is always consistent with what has been written: every mutating op records
//! fresh todos, prunes resolved ones, and saves.
//!
//! ## The clear-resolved invariant (the whole point)
//!
//! A todo is keyed by the *current hash of the broken def*. When a later
//! `update` fixes that def, the name now points at a NEW hash and the old
//! broken hash is orphaned — no current name resolves to it. So a recorded
//! todo is RESOLVED exactly when `cb.get_name(todo.name) != Some(todo.hash)`.
//! `clear_resolved` drops every such entry. A still-broken todo (whose name
//! still points at the recorded broken hash) is retained.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::codebase::Codebase;
use crate::hash::Hash;
use crate::typecheck::Todo;

// =============================================================================
// Errors (house style: Display + Error)
// =============================================================================

#[derive(Debug)]
pub enum TodoStoreError {
    /// An I/O failure reading or writing the on-disk todo file.
    Io(std::io::Error),
    /// The on-disk file was present but malformed (truncated / hand-corrupted).
    Parse(String),
}

impl fmt::Display for TodoStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TodoStoreError::Io(e) => write!(f, "todo store I/O error: {}", e),
            TodoStoreError::Parse(m) => write!(f, "todo store parse error: {}", m),
        }
    }
}

impl Error for TodoStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TodoStoreError::Io(e) => Some(e),
            TodoStoreError::Parse(_) => None,
        }
    }
}

impl From<std::io::Error> for TodoStoreError {
    fn from(e: std::io::Error) -> Self {
        TodoStoreError::Io(e)
    }
}

// =============================================================================
// Entry
// =============================================================================

/// One outstanding todo: a def that does not currently typecheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoEntry {
    /// The CURRENT hash of the broken def (the key — dedups + drives clearing).
    pub hash: Hash,
    /// The name the broken hash resolves through, if any.
    pub name: Option<String>,
    /// Human-readable description of the breakage.
    pub message: String,
    /// A short, stable kind string (e.g. "ArityMismatch", "TypeError") derived
    /// from the message when possible; "TypeError" otherwise.
    pub code: String,
}

impl TodoEntry {
    fn from_todo(t: &Todo) -> Self {
        TodoEntry {
            hash: t.hash,
            name: t.name.clone(),
            message: t.message.clone(),
            code: classify(&t.message),
        }
    }
}

/// Derive a short, stable code from a free-form type-error message. This is a
/// best-effort classifier over the messages `typecheck` actually emits; when no
/// known phrase matches we fall back to the generic "TypeError".
fn classify(message: &str) -> String {
    let m = message.to_ascii_lowercase();
    if m.contains("arity") || m.contains("expected") && m.contains("argument") {
        "ArityMismatch".to_string()
    } else if m.contains("unknown") || m.contains("not found") || m.contains("unresolved") {
        "UnresolvedRef".to_string()
    } else if m.contains("mismatch") || m.contains("expected") || m.contains("type") {
        "TypeError".to_string()
    } else {
        "TypeError".to_string()
    }
}

// =============================================================================
// Store
// =============================================================================

/// A per-branch, on-disk-backed set of outstanding todos, keyed by broken hash.
pub struct TodoStore {
    /// `<codebase_root>/todos/<branch>.json`.
    path: PathBuf,
    /// Outstanding entries, keyed by broken-def hash (so records dedup).
    entries: Vec<TodoEntry>,
}

impl TodoStore {
    /// Path to the per-branch todo file under the codebase root.
    fn path_for(root: &Path, branch: &str) -> PathBuf {
        // Branch names in this codebase are simple identifiers; still, guard
        // against a stray path separator so we never escape the todos dir.
        let safe: String = branch
            .chars()
            .map(|c| if c == '/' || c == '\\' { '_' } else { c })
            .collect();
        root.join("todos").join(format!("{}.json", safe))
    }

    /// Load the store for `branch`. An absent file yields an empty store; a
    /// present-but-corrupt file is a hard error (never silently treated empty).
    pub fn load(root: &Path, branch: &str) -> Result<Self, TodoStoreError> {
        let path = Self::path_for(root, branch);
        let entries = match std::fs::read_to_string(&path) {
            Ok(text) => parse_entries(&text)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(TodoStoreError::Io(e)),
        };
        Ok(TodoStore { path, entries })
    }

    /// Persist the current set, creating the `todos/` dir as needed. Writing an
    /// empty set leaves an empty `[]` file (so the on-disk state is explicit and
    /// a reader can distinguish "cleared" from "never touched"… both load empty,
    /// but the explicit file documents intent and keeps save/load symmetric).
    pub fn save(&self) -> Result<(), TodoStoreError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, render_entries(&self.entries))?;
        Ok(())
    }

    /// Add or replace entries, keyed by hash. New todos for a hash overwrite any
    /// prior entry for that same hash (a refreshed message/code wins).
    pub fn record(&mut self, todos: &[Todo]) {
        for t in todos {
            let entry = TodoEntry::from_todo(t);
            match self.entries.iter_mut().find(|e| e.hash == entry.hash) {
                Some(slot) => *slot = entry,
                None => self.entries.push(entry),
            }
        }
    }

    /// Drop every recorded todo that has been resolved. A todo is resolved when
    /// the name it was keyed to no longer points at the recorded (broken) hash —
    /// i.e. `cb.get_name(name) != Some(hash)`. A todo with no name is resolved
    /// when no current name in the codebase resolves to its hash at all.
    pub fn clear_resolved(&mut self, cb: &Codebase) {
        self.entries.retain(|e| match &e.name {
            Some(name) => cb.get_name(name) == Some(e.hash),
            None => cb.names().values().any(|h| *h == e.hash),
        });
    }

    /// The outstanding todos, sorted deterministically (by name, then hash).
    pub fn list(&self) -> Vec<TodoEntry> {
        let mut out = self.entries.clone();
        out.sort_by(|a, b| {
            let an = a.name.as_deref().unwrap_or("");
            let bn = b.name.as_deref().unwrap_or("");
            an.cmp(bn).then_with(|| a.hash.to_hex().cmp(&b.hash.to_hex()))
        });
        out
    }

    /// Number of outstanding todos (test/diagnostic convenience).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// Minimal, exact JSON (de)serialization for the entry array
// =============================================================================
//
// We deliberately hand-roll a tiny serializer/parser for the array of entries
// rather than pull in a generic JSON dependency, matching the codebase's
// dependency-light style. The format is a JSON array of objects with string
// fields `hash`, `name` (string or null), `message`, `code`.

fn render_entries(entries: &[TodoEntry]) -> String {
    let mut s = String::from("[\n");
    for (i, e) in entries.iter().enumerate() {
        s.push_str("  {");
        s.push_str(&format!("\"hash\": {}, ", json_string(&e.hash.to_hex())));
        match &e.name {
            Some(n) => s.push_str(&format!("\"name\": {}, ", json_string(n))),
            None => s.push_str("\"name\": null, "),
        }
        s.push_str(&format!("\"message\": {}, ", json_string(&e.message)));
        s.push_str(&format!("\"code\": {}", json_string(&e.code)));
        s.push('}');
        if i + 1 < entries.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("]\n");
    s
}

fn json_string(raw: &str) -> String {
    let mut s = String::with_capacity(raw.len() + 2);
    s.push('"');
    for c in raw.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
    s
}

/// Parse the array of entries. Reuses the project's JSONL parser so escapes,
/// nulls, and whitespace are handled correctly and identically to the rest of
/// the server.
fn parse_entries(text: &str) -> Result<Vec<TodoEntry>, TodoStoreError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let json = crate::jsonl::parse(trimmed)
        .map_err(|e| TodoStoreError::Parse(format!("invalid JSON: {}", e)))?;
    let arr = json
        .as_array()
        .ok_or_else(|| TodoStoreError::Parse("top-level value is not an array".to_string()))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let hash_hex = item
            .get("hash")
            .and_then(|j| j.as_str())
            .ok_or_else(|| TodoStoreError::Parse("entry missing string `hash`".to_string()))?;
        let hash = parse_hash_hex(hash_hex)
            .ok_or_else(|| TodoStoreError::Parse(format!("malformed hash hex: {}", hash_hex)))?;
        let name = match item.get("name") {
            Some(j) => j.as_str().map(|s| s.to_string()),
            None => None,
        };
        let message = item
            .get("message")
            .and_then(|j| j.as_str())
            .unwrap_or("")
            .to_string();
        let code = item
            .get("code")
            .and_then(|j| j.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| classify(&message));
        out.push(TodoEntry {
            hash,
            name,
            message,
            code,
        });
    }
    Ok(out)
}

/// Parse a 64-char lowercase-or-uppercase hex string into a [`Hash`]. There is
/// no `Hash::from_hex` in the public API, so we reconstruct the 32 bytes here.
fn parse_hash_hex(s: &str) -> Option<Hash> {
    if s.len() != Hash::SIZE * 2 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..Hash::SIZE {
        let hi = hex_val(bytes[i * 2])?;
        let lo = hex_val(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(Hash(out))
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tempdir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai_lang_todostore_{}_{}_{}_{}", tag, pid, nanos, n));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn h(byte: u8) -> Hash {
        Hash([byte; 32])
    }

    fn todo(hash: Hash, name: Option<&str>, msg: &str) -> Todo {
        Todo {
            hash,
            name: name.map(|s| s.to_string()),
            message: msg.to_string(),
        }
    }

    #[test]
    fn record_save_load_roundtrip() {
        let root = tempdir("roundtrip");
        let mut store = TodoStore::load(&root, "main").unwrap();
        assert!(store.is_empty());
        store.record(&[
            todo(h(1), Some("alpha"), "type mismatch: expected Int"),
            todo(h(2), None, "unknown reference foo"),
        ]);
        store.save().unwrap();

        let reloaded = TodoStore::load(&root, "main").unwrap();
        let list = reloaded.list();
        assert_eq!(list.len(), 2);
        // Sorted by name: "" (the None entry) sorts first.
        assert_eq!(list[0].name, None);
        assert_eq!(list[0].code, "UnresolvedRef");
        assert_eq!(list[1].name.as_deref(), Some("alpha"));
        assert_eq!(list[1].message, "type mismatch: expected Int");
    }

    #[test]
    fn record_dedups_by_hash() {
        let root = tempdir("dedup");
        let mut store = TodoStore::load(&root, "main").unwrap();
        store.record(&[todo(h(1), Some("alpha"), "first message")]);
        store.record(&[todo(h(1), Some("alpha"), "refreshed message")]);
        assert_eq!(store.len(), 1);
        assert_eq!(store.list()[0].message, "refreshed message");
    }

    #[test]
    fn absent_file_is_empty() {
        let root = tempdir("absent");
        let store = TodoStore::load(&root, "never-saved").unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn per_branch_files_are_independent() {
        let root = tempdir("perbranch");
        let mut main = TodoStore::load(&root, "main").unwrap();
        main.record(&[todo(h(1), Some("alpha"), "broken on main")]);
        main.save().unwrap();

        let mut scratch = TodoStore::load(&root, "scratch").unwrap();
        scratch.record(&[todo(h(2), Some("beta"), "broken on scratch")]);
        scratch.save().unwrap();

        // Each branch sees only its own.
        let main2 = TodoStore::load(&root, "main").unwrap();
        let scratch2 = TodoStore::load(&root, "scratch").unwrap();
        assert_eq!(main2.list().len(), 1);
        assert_eq!(main2.list()[0].name.as_deref(), Some("alpha"));
        assert_eq!(scratch2.list().len(), 1);
        assert_eq!(scratch2.list()[0].name.as_deref(), Some("beta"));
    }

    /// The headline resolution invariant, exercised with a real Codebase: a
    /// todo keyed to `X@hashA` is dropped once `X` moves to `hashB`, while a
    /// still-broken todo (name still at its recorded hash) is retained.
    #[test]
    fn clear_resolved_drops_fixed_keeps_broken() {
        let root = tempdir("resolve");
        let cb_dir = root.join("cb");
        let mut cb = Codebase::open(&cb_dir).unwrap();

        // Put two real defs in the codebase so we have real hashes/names. The
        // bodies differ so the two content hashes differ (names are not part of
        // the content hash in this content-addressed store).
        let added_x = crate::edit::add(&mut cb, "def x(a: Int) -> Int = a").unwrap();
        let added_y = crate::edit::add(&mut cb, "def y(a: Int) -> Int = a + 100").unwrap();
        let x_hash_a = added_x[0].hash;
        let _y_hash = added_y[0].hash;
        assert_ne!(x_hash_a, _y_hash, "distinct bodies must hash distinctly");

        // Record a todo for x@hashA (pretend x is broken) and one for y.
        let mut store = TodoStore::load(&root, "main").unwrap();
        store.record(&[
            todo(x_hash_a, Some("x"), "broken: type mismatch"),
            todo(_y_hash, Some("y"), "broken: type mismatch"),
        ]);
        assert_eq!(store.len(), 2);

        // "Fix" x: update it so its name moves to a NEW hash.
        crate::edit::update(&mut cb, "x", "def x(a: Int) -> Int = a + 1").unwrap();
        let x_hash_b = cb.get_name("x").unwrap();
        assert_ne!(x_hash_b, x_hash_a, "update must move x's hash");
        // y is untouched, still pointing at its recorded hash.
        assert_eq!(cb.get_name("y"), Some(_y_hash));

        store.clear_resolved(&cb);

        // x's todo is resolved (dropped); y's is retained.
        let names: Vec<Option<String>> = store.list().into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec![Some("y".to_string())], "x cleared, y kept");
    }

    #[test]
    fn unnamed_todo_cleared_when_hash_orphaned() {
        let root = tempdir("unnamed");
        let cb_dir = root.join("cb");
        let mut cb = Codebase::open(&cb_dir).unwrap();
        let added = crate::edit::add(&mut cb, "def z(a: Int) -> Int = a").unwrap();
        let z_hash = added[0].hash;

        let mut store = TodoStore::load(&root, "main").unwrap();
        // An unnamed todo whose hash IS currently reachable: retained.
        store.record(&[todo(z_hash, None, "broken")]);
        store.clear_resolved(&cb);
        assert_eq!(store.len(), 1, "reachable unnamed hash retained");

        // After moving z, the old hash is orphaned: cleared.
        crate::edit::update(&mut cb, "z", "def z(a: Int) -> Int = a + 2").unwrap();
        store.clear_resolved(&cb);
        assert_eq!(store.len(), 0, "orphaned unnamed hash cleared");
    }
}
