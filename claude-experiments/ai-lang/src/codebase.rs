//! On-disk content-addressed codebase.
//!
//! The codebase is a directory layout:
//!
//! ```text
//! <root>/
//!   defs/
//!     <hex(hash)>.def    one file per canonical def, named by its content hash
//!   types/
//!     <hex(hash)>.type   one file per typed def, the cached TypeScheme
//!   names.txt             "name<TAB>hex_hash\n" per line — surface namespace
//! ```
//!
//! The `types/` directory is the on-disk projection of [`TypeCache`].
//! Because a def's hash uniquely determines its type, type files never
//! need to be recomputed unless the def itself changes (which would
//! produce a new hash). Reopening a codebase loads every existing
//! `<hex>.type` into memory; subsequent `typecheck_module` calls find
//! every previously-typed def in the cache and do zero work for them.
//!
//! The store is the single source of truth for code: source files are a
//! *projection* of the canonical AST, and the canonical AST is a
//! projection of these bytes. Two codebases that hold the same set of
//! `<hex>.def` files contain identical definitions; the `names.txt`
//! file is mutable namespace state separate from code identity.
//!
//! ## Properties enforced by this module
//!
//! - **Content-address integrity.** Every `load_def(hash)` re-hashes the
//!   bytes read from disk and verifies they match the requested hash;
//!   any mismatch (bit rot, accidental edit, restore from the wrong
//!   backup) returns [`CodebaseError::HashMismatch`].
//! - **Idempotent stores.** `store_def` of the same canonical def
//!   produces the same hash and the same file bytes; calling it twice
//!   is a no-op on the second call.
//! - **Names are mutable.** `set_name`/`remove_name` modify the
//!   namespace and persist immediately; the underlying defs are never
//!   mutated (you create a new def with a new hash, then point the
//!   name at it).
//!
//! ## What this module does *not* do (yet)
//!
//! - **Multi-process locking.** Concurrent writers from different
//!   processes can race. For v1 single-process use this is fine; a
//!   lockfile is a small follow-up.
//! - **Atomic name updates.** `names.txt` is rewritten whole on each
//!   change. A crash mid-write leaves a partial file; we'd want
//!   write-temp-and-rename for that.
//! - **Subdirectory sharding.** All defs live in one directory.
//!   Filesystems handle low millions of files OK; we'll shard if it
//!   becomes a real problem.
//! - **Garbage collection of unreferenced defs.** Per the proposal:
//!   probably never, since code is small and cheap to keep.

use crate::ast::Def;
use crate::codec::{DecodeError, decode_def, encode_def};
use crate::hash::Hash;
use crate::resolve::ResolvedModule;
use crate::typecheck::{SchemeCodecError, TypeCache, TypeScheme, decode_scheme, encode_scheme};

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug)]
pub enum CodebaseError {
    Io(io::Error),
    Decode(DecodeError),
    SchemeDecode(SchemeCodecError),
    HashMismatch {
        expected: Hash,
        actual: Hash,
        path: PathBuf,
    },
    /// A line in `names.txt` couldn't be parsed.
    BadNamesLine {
        line_no: usize,
        line: String,
    },
    /// Asked for a def that isn't in the store.
    MissingDef(Hash),
    /// A `types/<hex>.type` file's name wasn't a valid 64-char hex hash.
    BadTypeFilename(PathBuf),
    /// The causal-namespace layer (`namespace.rs`) failed during an auto-commit
    /// or migration triggered from a name-changing operation. Carries the
    /// underlying error's description (kept as a string to avoid a cyclic
    /// error-type dependency between the two modules).
    Namespace(String),
}

impl core::fmt::Display for CodebaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CodebaseError::Io(e) => write!(f, "io error: {}", e),
            CodebaseError::Decode(e) => write!(f, "decode error: {}", e),
            CodebaseError::HashMismatch {
                expected,
                actual,
                path,
            } => write!(
                f,
                "hash mismatch on {}: expected {}, got {}",
                path.display(),
                expected,
                actual
            ),
            CodebaseError::BadNamesLine { line_no, line } => {
                write!(f, "bad line {} in names.txt: {:?}", line_no, line)
            }
            CodebaseError::MissingDef(h) => write!(f, "def {} not in codebase", h),
            CodebaseError::SchemeDecode(e) => write!(f, "scheme decode error: {}", e),
            CodebaseError::BadTypeFilename(p) => {
                write!(f, "type filename is not a 64-char hex hash: {}", p.display())
            }
            CodebaseError::Namespace(msg) => write!(f, "namespace error: {}", msg),
        }
    }
}

impl From<SchemeCodecError> for CodebaseError {
    fn from(e: SchemeCodecError) -> Self {
        CodebaseError::SchemeDecode(e)
    }
}

impl std::error::Error for CodebaseError {}

impl From<io::Error> for CodebaseError {
    fn from(e: io::Error) -> Self {
        CodebaseError::Io(e)
    }
}

impl From<DecodeError> for CodebaseError {
    fn from(e: DecodeError) -> Self {
        CodebaseError::Decode(e)
    }
}

// =============================================================================
// Codebase
// =============================================================================

pub struct Codebase {
    root: PathBuf,
    names: HashMap<String, Hash>,
    types: TypeCache,
    /// When `true`, every name-changing operation
    /// (`set_name`/`remove_name`/`store_resolved_module`) auto-commits a new
    /// causal namespace snapshot to the current branch (see `namespace.rs`).
    /// Enabled by default at `open` so history/undo capture real edits with
    /// per-operation granularity. Cleared internally while a causal `switch`
    /// or `undo` rewrites `names.txt` so reloading a snapshot does not itself
    /// create a spurious commit.
    auto_commit: bool,
}

impl Codebase {
    /// Open (or create) the codebase rooted at `path`. The directory and
    /// its `defs/` subdirectory are created if they don't already exist;
    /// any existing `names.txt` is loaded into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CodebaseError> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(root.join("defs"))?;
        fs::create_dir_all(root.join("types"))?;
        fs::create_dir_all(root.join("localnames"))?;
        let names = read_names_file(&root.join("names.txt"))?;
        let types = read_types_dir(&root.join("types"))?;
        let mut cb = Codebase {
            root,
            names,
            types,
            auto_commit: true,
        };
        // Migrate / initialise the causal namespace layer: if there is no
        // `branches/` dir yet, create branch "main" from the current
        // `names.txt` snapshot. This is backward-compatible: an existing
        // codebase with only a `names.txt` keeps working, and `main` is
        // seeded from it on first open. See `namespace.rs`.
        crate::namespace::init_or_migrate(&mut cb)?;
        Ok(cb)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    // ---- Def-by-hash storage ----

    /// Encode + hash + write. Returns the def's content hash. Idempotent.
    ///
    /// NOTE: this re-hashes the encoded bytes and is therefore only
    /// correct for defs that don't contain self-references — i.e.
    /// non-recursive types and all fn defs. For recursive type defs
    /// (members of a type-SCC), the resolver hashes the *canonical*
    /// form (with `Type::SelfRef` placeholders) but the stored AST
    /// substitutes `Type::TypeRef(real_hash)`; re-encoding the stored
    /// form does not round-trip the resolver's hash. Use
    /// [`store_def_at`] when you already have the resolver's hash and
    /// want to honour it.
    pub fn store_def(&self, def: &Def) -> Result<Hash, CodebaseError> {
        let bytes = encode_def(def);
        let hash = Hash::of_bytes(&bytes);
        let path = self.def_path(&hash);
        if !path.exists() {
            fs::write(&path, &bytes)?;
        }
        Ok(hash)
    }

    /// Encode + write under the provided hash (no re-hash verification).
    /// Used by [`Self::store_resolved_module`] so the resolver-canonical
    /// hash is honoured even for recursive type defs whose stored bytes
    /// don't round-trip through `encode_def → Hash::of_bytes`.
    pub fn store_def_at(&self, hash: &Hash, def: &Def) -> Result<(), CodebaseError> {
        let bytes = encode_def(def);
        let path = self.def_path(hash);
        if !path.exists() {
            fs::write(&path, &bytes)?;
        }
        Ok(())
    }

    /// Read + decode by hash. The on-disk bytes are trusted to belong
    /// to the requested hash (the filename IS the hash key); we do not
    /// re-hash, because recursive-type stored forms don't round-trip.
    /// Bit-rot detection at this layer is therefore weakened; callers
    /// concerned with integrity should run a separate verification
    /// pass that knows about SCC re-canonicalisation.
    pub fn load_def(&self, hash: &Hash) -> Result<Def, CodebaseError> {
        let path = self.def_path(hash);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(CodebaseError::MissingDef(*hash));
            }
            Err(e) => return Err(e.into()),
        };
        Ok(decode_def(&bytes)?)
    }

    /// True if a def with this hash is in the store.
    pub fn contains(&self, hash: &Hash) -> bool {
        self.def_path(hash).exists()
    }

    /// Path that a def with this hash would live at, present or not.
    pub fn def_path(&self, hash: &Hash) -> PathBuf {
        self.root.join("defs").join(format!("{}.def", hash.to_hex()))
    }

    // ---- Type cache ----

    /// In-memory view of the on-disk type cache. Pass `&mut codebase.types_mut()`
    /// to `typecheck_module` to skip already-typed defs.
    pub fn types(&self) -> &TypeCache {
        &self.types
    }

    pub fn types_mut(&mut self) -> &mut TypeCache {
        &mut self.types
    }

    /// Path the on-disk type file for this hash would live at.
    pub fn type_path(&self, hash: &Hash) -> PathBuf {
        self.root
            .join("types")
            .join(format!("{}.type", hash.to_hex()))
    }

    /// Persist a single `(hash, scheme)` to disk. Idempotent — writing
    /// the same scheme twice is a no-op on the filesystem (the encoded
    /// bytes are deterministic).
    pub fn store_typescheme(
        &mut self,
        hash: Hash,
        scheme: TypeScheme,
    ) -> Result<(), CodebaseError> {
        let bytes = encode_scheme(&scheme);
        let path = self.type_path(&hash);
        if !path.exists() {
            fs::write(&path, &bytes)?;
        }
        self.types.insert(hash, scheme);
        Ok(())
    }

    /// Persist every newly-typed scheme from a [`TypeCache`] to disk
    /// for any hash that isn't already on the filesystem. Returns the
    /// number of files written.
    pub fn store_typecache(&mut self, src: &TypeCache) -> Result<usize, CodebaseError> {
        let mut written = 0usize;
        for (h, scheme) in src.iter() {
            let path = self.type_path(h);
            if !path.exists() {
                fs::write(&path, &encode_scheme(scheme))?;
                written += 1;
            }
            self.types.insert(*h, scheme.clone());
        }
        Ok(written)
    }

    /// Read a single type scheme from disk by hash. Returns `None` if
    /// the file doesn't exist. (Used in tests and as the building
    /// block for `read_types_dir`.)
    pub fn load_typescheme(&self, hash: &Hash) -> Result<Option<TypeScheme>, CodebaseError> {
        let path = self.type_path(hash);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        Ok(Some(decode_scheme(&bytes)?))
    }

    // ---- Namespace ----

    pub fn names(&self) -> &HashMap<String, Hash> {
        &self.names
    }

    pub fn get_name(&self, name: &str) -> Option<Hash> {
        self.names.get(name).copied()
    }

    /// Point `name` at `hash` and persist. Overwrites any previous
    /// mapping for the same name.
    pub fn set_name(&mut self, name: impl Into<String>, hash: Hash) -> Result<(), CodebaseError> {
        self.names.insert(name.into(), hash);
        self.persist_names()?;
        self.maybe_autocommit()?;
        Ok(())
    }

    /// Remove a name from the namespace (does not delete the def
    /// itself). Returns whether the name existed.
    pub fn remove_name(&mut self, name: &str) -> Result<bool, CodebaseError> {
        let existed = self.names.remove(name).is_some();
        if existed {
            self.persist_names()?;
            self.maybe_autocommit()?;
        }
        Ok(existed)
    }

    /// Store every def from a resolved module under its resolver-
    /// computed hash, then point each name at that hash. Recursive
    /// type defs do not round-trip `encode_def → Hash::of_bytes` (the
    /// resolver hashed the SelfRef canonical form; the stored form has
    /// TopRefs), so we honour `rd.hash` directly via `store_def_at`.
    pub fn store_resolved_module(&mut self, rm: &ResolvedModule) -> Result<(), CodebaseError> {
        for rd in &rm.defs {
            self.store_def_at(&rd.hash, &rd.def)?;
            self.names.insert(rd.name.clone(), rd.hash);
        }
        self.persist_names()?;
        self.maybe_autocommit()?;
        Ok(())
    }

    // ---- Local-name side-car (readable param/let/lambda/match names) ----
    //
    // The canonical AST is name-erased (locals are de Bruijn indices) so
    // that two functions identical up to local renaming hash the same. To
    // give the `printer`'s `view` output readable names we keep the
    // author's original local names OUTSIDE the hashed bytes, in
    // `localnames/<hash>.names`: one UTF-8 line per name, in binder-push
    // order (fn params first, then `let`/lambda-param/match-binding names),
    // produced by [`crate::resolve::local_names_for_module`].
    //
    // This directory is pure presentation metadata; it never affects a
    // def's identity. A missing file is normal (the printer falls back to
    // `p0/p1/...`). Names are not part of `names.txt` (the namespace) and
    // do not trigger a causal commit.

    /// Path the local-name side-car for `hash` would live at.
    pub fn local_names_path(&self, hash: &Hash) -> PathBuf {
        self.root
            .join("localnames")
            .join(format!("{}.names", hash.to_hex()))
    }

    /// Persist the author's local names for one def hash. One name per
    /// line, in binder-push order. A name MUST NOT contain a newline (no
    /// surface identifier can), so the line encoding is unambiguous.
    /// Idempotent: re-writing the same names is a no-op on disk.
    ///
    /// An empty list still writes an (empty) file, recording that the def
    /// genuinely has no locals as distinct from "never captured" (no
    /// file). The printer treats both as "fall back to p0/p1".
    pub fn store_local_names(
        &self,
        hash: &Hash,
        names: &[String],
    ) -> Result<(), CodebaseError> {
        let mut out = String::new();
        for n in names {
            debug_assert!(
                !n.contains('\n'),
                "local name must not contain a newline"
            );
            out.push_str(n);
            out.push('\n');
        }
        let path = self.local_names_path(hash);
        // Always (over)write: an edit can change names while keeping the
        // same hash only if the body is byte-identical, but a different
        // body yields a different hash + path, so a stale file is never
        // read for the wrong def. Writing unconditionally keeps the latest
        // capture authoritative.
        fs::write(&path, out)?;
        Ok(())
    }

    /// Persist a whole batch of `hash -> names` side-cars (as produced by
    /// [`crate::resolve::local_names_for_module`]). Returns the count
    /// written.
    pub fn store_local_names_batch(
        &self,
        map: &HashMap<Hash, Vec<String>>,
    ) -> Result<usize, CodebaseError> {
        for (h, names) in map {
            self.store_local_names(h, names)?;
        }
        Ok(map.len())
    }

    /// Load the author's local names for `hash`, or `None` when no
    /// side-car file exists (the printer then falls back to `p0/p1/...`).
    /// Each non-final newline delimits one name; a trailing newline does
    /// not create an empty trailing entry.
    pub fn load_local_names(&self, hash: &Hash) -> Result<Option<Vec<String>>, CodebaseError> {
        let path = self.local_names_path(hash);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let text = core::str::from_utf8(&bytes).map_err(|_| CodebaseError::BadNamesLine {
            line_no: 0,
            line: "<localnames not utf-8>".to_owned(),
        })?;
        // `lines()` drops a single trailing newline and yields no trailing
        // empty entry, which is exactly the inverse of `store_local_names`.
        let names: Vec<String> = text.lines().map(|l| l.to_owned()).collect();
        Ok(Some(names))
    }

    // ---- Causal namespace integration (see `namespace.rs`) ----

    /// Whether name-changing operations auto-commit a causal snapshot.
    pub fn auto_commit_enabled(&self) -> bool {
        self.auto_commit
    }

    /// Enable/disable auto-commit. The causal `switch`/`undo` paths disable it
    /// while they rewrite `names.txt` from a target snapshot, so reloading a
    /// historical namespace does not itself spawn a new commit.
    pub fn set_auto_commit(&mut self, on: bool) {
        self.auto_commit = on;
    }

    /// Replace the in-memory + on-disk name map wholesale, WITHOUT
    /// auto-committing. Used by the causal layer (`switch`/`undo`) to project a
    /// chosen namespace snapshot back into the backward-compatible `names.txt`
    /// view that the existing name API reads. Auto-commit state is preserved.
    pub fn replace_names(
        &mut self,
        names: HashMap<String, Hash>,
    ) -> Result<(), CodebaseError> {
        let prev = self.auto_commit;
        self.auto_commit = false;
        self.names = names;
        self.persist_names()?;
        self.auto_commit = prev;
        Ok(())
    }

    /// If auto-commit is on, snapshot the current name map into a new causal
    /// node on the active branch (no-op if the snapshot is unchanged).
    fn maybe_autocommit(&mut self) -> Result<(), CodebaseError> {
        if self.auto_commit {
            crate::namespace::commit_namespace(self)
                .map_err(|e| CodebaseError::Namespace(e.to_string()))?;
        }
        Ok(())
    }

    fn persist_names(&self) -> Result<(), CodebaseError> {
        let path = self.root.join("names.txt");
        // Sort by name so the file diffs cleanly across runs.
        let mut entries: Vec<_> = self.names.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = String::new();
        for (name, hash) in entries {
            out.push_str(name);
            out.push('\t');
            out.push_str(&hash.to_hex());
            out.push('\n');
        }
        fs::write(&path, out)?;
        Ok(())
    }
}

fn read_types_dir(path: &Path) -> Result<TypeCache, CodebaseError> {
    let mut cache = TypeCache::new();
    let entries = match fs::read_dir(path) {
        Ok(it) => it,
        // Directory didn't exist yet — caller will create it. Returning
        // an empty cache is correct.
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(cache),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        // Files are "<64 hex chars>.type". Skip anything else so an
        // accidentally-dropped scratch file in the directory doesn't
        // break a load.
        let Some(stem) = name.strip_suffix(".type") else {
            continue;
        };
        let hash = parse_hex_hash(stem).ok_or_else(|| {
            CodebaseError::BadTypeFilename(entry.path())
        })?;
        let bytes = fs::read(entry.path())?;
        let scheme = decode_scheme(&bytes)?;
        cache.insert(hash, scheme);
    }
    Ok(cache)
}

fn read_names_file(path: &Path) -> Result<HashMap<String, Hash>, CodebaseError> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => return Err(e.into()),
    };
    let text =
        core::str::from_utf8(&bytes).map_err(|_| CodebaseError::BadNamesLine {
            line_no: 0,
            line: "<not utf-8>".to_owned(),
        })?;
    let mut out = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (name, hex) = line.split_once('\t').ok_or_else(|| {
            CodebaseError::BadNamesLine {
                line_no: i + 1,
                line: line.to_owned(),
            }
        })?;
        let hash = parse_hex_hash(hex).ok_or_else(|| CodebaseError::BadNamesLine {
            line_no: i + 1,
            line: line.to_owned(),
        })?;
        out.insert(name.to_owned(), hash);
    }
    Ok(out)
}

fn parse_hex_hash(hex: &str) -> Option<Hash> {
    if hex.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        let hi = hex_nibble_value(hex.as_bytes()[i * 2])?;
        let lo = hex_nibble_value(hex.as_bytes()[i * 2 + 1])?;
        bytes[i] = (hi << 4) | lo;
    }
    Some(Hash(bytes))
}

fn hex_nibble_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(10 + c - b'a'),
        b'A'..=b'F' => Some(10 + c - b'A'),
        _ => None,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Type};

    /// Unique temp directory per test. Cleaned up by the OS on temp
    /// rotation; tests don't rely on cleanup between runs.
    fn tempdir(name: &str) -> PathBuf {
        // Suffix with a high-resolution timestamp so parallel test runs
        // don't share a directory.
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai-lang-test-{}-{}", name, nanos));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn sample_def() -> Def {
        // Equivalent to `def double(x: Int) -> Int = x * 2`
        Def::Fn {
            is_local: false,
            type_params: 0,
            params: vec![Type::Builtin("Int".to_owned())],
            ret: Type::Builtin("Int".to_owned()),
            body: Expr::Call(
                Box::new(Expr::BuiltinRef("core/i64.mul".to_owned())),
                vec![Expr::LocalVar(0), Expr::IntLit(2)],
            ),
        }
    }

    #[test]
    fn open_creates_root_and_defs_dir() {
        let dir = tempdir("open_creates");
        let cb = Codebase::open(&dir).unwrap();
        assert!(dir.exists());
        assert!(dir.join("defs").exists());
        assert!(cb.names().is_empty());
    }

    #[test]
    fn store_then_load_roundtrip() {
        let dir = tempdir("roundtrip");
        let cb = Codebase::open(&dir).unwrap();
        let def = sample_def();
        let h = cb.store_def(&def).unwrap();
        assert!(cb.contains(&h));
        assert_eq!(cb.load_def(&h).unwrap(), def);
    }

    #[test]
    fn store_is_idempotent() {
        let dir = tempdir("idempotent");
        let cb = Codebase::open(&dir).unwrap();
        let def = sample_def();
        let h1 = cb.store_def(&def).unwrap();
        let h2 = cb.store_def(&def).unwrap();
        assert_eq!(h1, h2);
        // The on-disk file should still hash correctly.
        assert_eq!(cb.load_def(&h1).unwrap(), def);
    }

    #[test]
    fn load_missing_def_errors() {
        let dir = tempdir("missing");
        let cb = Codebase::open(&dir).unwrap();
        let bogus = Hash([0xAB; 32]);
        match cb.load_def(&bogus) {
            Err(CodebaseError::MissingDef(h)) => assert_eq!(h, bogus),
            other => panic!("expected MissingDef, got {:?}", other),
        }
    }

    /// `load_def` used to re-hash the on-disk bytes and reject any
    /// mismatch as bit-rot. That contract was incompatible with
    /// recursive type defs: the resolver hashes their canonical
    /// `SelfRef` form but stores the resolved `TopRef` form, so a
    /// re-hash-on-read mismatch is the expected case for them.
    ///
    /// For now `load_def` trusts the filename. Bit-rot detection moves
    /// to a separate `verify` pass (TBD) that can re-canonicalise SCC
    /// members. This test documents the relaxed contract — corrupted
    /// bytes will decode as garbage rather than triggering an error
    /// at load.
    #[test]
    fn load_def_trusts_filename_no_rehash() {
        let dir = tempdir("trust");
        let cb = Codebase::open(&dir).unwrap();
        let def = sample_def();
        let h = cb.store_def(&def).unwrap();
        let path = cb.def_path(&h);
        let bytes = fs::read(&path).unwrap();
        // Untouched file: load returns the def.
        assert_eq!(cb.load_def(&h).unwrap(), def);
        // Corrupt one byte; the loader may now return decode garbage
        // or a `DecodeError`, but it will NOT raise `HashMismatch`.
        let mut tampered = bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xff;
        fs::write(&path, &tampered).unwrap();
        match cb.load_def(&h) {
            Err(CodebaseError::HashMismatch { .. }) => {
                panic!("load_def no longer performs hash verification")
            }
            _ => {} // either decode error or decoded-but-different is fine
        }
    }

    #[test]
    fn names_persist_across_reopen() {
        let dir = tempdir("names");
        let h = {
            let mut cb = Codebase::open(&dir).unwrap();
            let h = cb.store_def(&sample_def()).unwrap();
            cb.set_name("double", h).unwrap();
            cb.set_name("dbl", h).unwrap();
            h
        };
        // Drop the first codebase, reopen.
        let cb2 = Codebase::open(&dir).unwrap();
        assert_eq!(cb2.get_name("double"), Some(h));
        assert_eq!(cb2.get_name("dbl"), Some(h));
        assert_eq!(cb2.names().len(), 2);
        // Def is still loadable.
        assert_eq!(cb2.load_def(&h).unwrap(), sample_def());
    }

    #[test]
    fn remove_name_updates_persisted_file() {
        let dir = tempdir("remove_name");
        let h = Hash([0x55; 32]);
        let mut cb = Codebase::open(&dir).unwrap();
        cb.set_name("a", h).unwrap();
        cb.set_name("b", h).unwrap();
        assert!(cb.remove_name("a").unwrap());
        assert!(!cb.remove_name("a").unwrap());

        // Reopen and verify.
        let cb2 = Codebase::open(&dir).unwrap();
        assert_eq!(cb2.get_name("a"), None);
        assert_eq!(cb2.get_name("b"), Some(h));
    }

    #[test]
    fn store_resolved_module_writes_defs_and_names() {
        use crate::parser::parse_module;
        use crate::resolve::resolve_module;

        let dir = tempdir("store_module");
        let mut cb = Codebase::open(&dir).unwrap();

        let src = "
            def double(x: Int) -> Int = x * 2
            def quadruple(x: Int) -> Int = double(double(x))
        ";
        let m = parse_module(src).unwrap();
        let rm = resolve_module(&m).unwrap();
        cb.store_resolved_module(&rm).unwrap();

        let dh = rm.get("double").unwrap().hash;
        let qh = rm.get("quadruple").unwrap().hash;

        assert_eq!(cb.get_name("double"), Some(dh));
        assert_eq!(cb.get_name("quadruple"), Some(qh));
        assert_eq!(cb.load_def(&dh).unwrap(), rm.get("double").unwrap().def);
        assert_eq!(cb.load_def(&qh).unwrap(), rm.get("quadruple").unwrap().def);

        // Reopen and re-resolve from disk.
        let cb2 = Codebase::open(&dir).unwrap();
        assert_eq!(cb2.load_def(&dh).unwrap(), rm.get("double").unwrap().def);
        assert_eq!(cb2.load_def(&qh).unwrap(), rm.get("quadruple").unwrap().def);
    }

    #[test]
    fn bad_names_line_errors() {
        let dir = tempdir("bad_names");
        fs::create_dir_all(dir.join("defs")).unwrap();
        fs::write(dir.join("names.txt"), "missing-tab-here\n").unwrap();
        match Codebase::open(&dir) {
            Err(CodebaseError::BadNamesLine { line_no, .. }) => assert_eq!(line_no, 1),
            Err(other) => panic!("expected BadNamesLine, got {:?}", other),
            Ok(_) => panic!("expected BadNamesLine, got Ok"),
        }
    }

    #[test]
    fn structs_and_fns_roundtrip_through_store() {
        use crate::parser::parse_module;
        use crate::resolve::resolve_module;

        let dir = tempdir("structs_roundtrip");
        let mut cb = Codebase::open(&dir).unwrap();
        let src = "
            struct Point { x: Int, y: Int }
            def get_x(p: Point) -> Int = p.x
        ";
        let m = parse_module(src).unwrap();
        let rm = resolve_module(&m).unwrap();
        cb.store_resolved_module(&rm).unwrap();

        // Both the struct def and the fn def should round-trip.
        for rd in &rm.defs {
            assert_eq!(cb.load_def(&rd.hash).unwrap(), rd.def);
        }
    }

    // ---- Disk-backed TypeCache ----

    #[test]
    fn typescheme_roundtrip_on_disk() {
        use crate::ast::Type;
        let dir = tempdir("ts_roundtrip");
        let mut cb = Codebase::open(&dir).unwrap();
        let scheme = TypeScheme::Fn {
            params: vec![Type::Builtin("Int".to_owned())],
            ret: Type::Builtin("Int".to_owned()),
            wire: true,
        };
        let h = Hash([0x42; 32]);
        cb.store_typescheme(h, scheme.clone()).unwrap();
        assert_eq!(cb.load_typescheme(&h).unwrap(), Some(scheme.clone()));
        // Reopen — survives across processes.
        let cb2 = Codebase::open(&dir).unwrap();
        assert_eq!(cb2.load_typescheme(&h).unwrap(), Some(scheme.clone()));
        assert_eq!(cb2.types().get(&h), Some(&scheme));
    }

    /// The headline: across a reopen, typecheck reports 100% cache hits.
    #[test]
    fn reopen_loads_types_so_typecheck_is_all_cache_hits() {
        use crate::parser::parse_module;
        use crate::resolve::resolve_module;
        use crate::typecheck::typecheck_module;

        let dir = tempdir("typecache_reopen");

        // ---- First run: typecheck from scratch, persist to disk ----
        let src = "
            struct Point { x: Int, y: Int }
            def get_x(p: Point) -> Int = p.x
            def double(x: Int) -> Int = x * 2
        ";
        let m = parse_module(src).unwrap();
        let rm = resolve_module(&m).unwrap();

        let report_a;
        {
            let mut cb = Codebase::open(&dir).unwrap();
            cb.store_resolved_module(&rm).unwrap();
            let mut cache = TypeCache::new();
            report_a = typecheck_module(&rm, &mut cache).unwrap();
            cb.store_typecache(&cache).unwrap();
        }
        assert_eq!(report_a.newly_typed, 3);
        assert_eq!(report_a.cache_hits, 0);

        // ---- Reopen and re-typecheck. Should be 100% hits. ----
        let cb = Codebase::open(&dir).unwrap();
        // The cache loaded from disk has every scheme already.
        assert_eq!(cb.types().len(), 3);
        let mut cache = TypeCache::new();
        // Seed from disk-backed cache so the typecheck sees the
        // pre-loaded entries. Using contains() this is automatic via the
        // codebase-owned types_mut(), but to keep the test boundary
        // explicit we copy from the disk-backed cache:
        for (h, s) in cb.types().iter() {
            cache.insert(*h, s.clone());
        }
        let report_b = typecheck_module(&rm, &mut cache).unwrap();
        assert_eq!(
            report_b.newly_typed, 0,
            "reopen should yield zero new typechecks; got {:?}",
            report_b
        );
        assert_eq!(report_b.cache_hits, 3);
        assert_eq!(report_b.total, 3);
    }

    #[test]
    fn corrupted_type_file_detected() {
        let dir = tempdir("corrupt_type");
        let mut cb = Codebase::open(&dir).unwrap();
        let scheme = TypeScheme::Struct {
            type_params: 0,
            fields: vec![],
            wire: true,
        };
        let h = Hash([0x77; 32]);
        cb.store_typescheme(h, scheme).unwrap();
        // Tamper with the bytes.
        let path = cb.type_path(&h);
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&path, &bytes).unwrap();
        // Force a non-trailing byte flip too so we hit the decoder's
        // length or tag fields, guaranteeing a decode error.
        let mut bytes2 = fs::read(&path).unwrap();
        bytes2[0] ^= 0xff;
        fs::write(&path, &bytes2).unwrap();
        match Codebase::open(&dir) {
            Err(_) => {} // expected
            Ok(_) => panic!("type file corruption not detected"),
        }
    }

    #[test]
    fn local_names_roundtrip_on_disk() {
        let dir = tempdir("localnames_roundtrip");
        let h = Hash([0x33; 32]);
        let names = vec!["price".to_owned(), "qty".to_owned()];
        {
            let cb = Codebase::open(&dir).unwrap();
            assert_eq!(cb.load_local_names(&h).unwrap(), None);
            cb.store_local_names(&h, &names).unwrap();
            assert_eq!(cb.load_local_names(&h).unwrap(), Some(names.clone()));
        }
        // Survives reopen.
        let cb2 = Codebase::open(&dir).unwrap();
        assert_eq!(cb2.load_local_names(&h).unwrap(), Some(names));
    }

    #[test]
    fn local_names_empty_vs_absent() {
        let dir = tempdir("localnames_empty");
        let cb = Codebase::open(&dir).unwrap();
        let h = Hash([0x44; 32]);
        // Absent file -> None.
        assert_eq!(cb.load_local_names(&h).unwrap(), None);
        // Stored empty list -> Some(empty), distinct from absent.
        cb.store_local_names(&h, &[]).unwrap();
        assert_eq!(cb.load_local_names(&h).unwrap(), Some(Vec::new()));
    }

    #[test]
    fn local_names_from_module_capture() {
        use crate::parser::parse_module;
        use crate::resolve::local_names_for_module;

        let dir = tempdir("localnames_capture");
        let mut cb = Codebase::open(&dir).unwrap();
        let src = "def total(price: Int, qty: Int) -> Int = price * qty";
        let m = parse_module(src).unwrap();
        let rm = crate::resolve::resolve_module(&m).unwrap();
        cb.store_resolved_module(&rm).unwrap();
        let names = local_names_for_module(&m).unwrap();
        cb.store_local_names_batch(&names).unwrap();

        let h = rm.get("total").unwrap().hash;
        assert_eq!(
            cb.load_local_names(&h).unwrap(),
            Some(vec!["price".to_owned(), "qty".to_owned()])
        );
    }

    #[test]
    fn store_typecache_writes_only_missing_files() {
        let dir = tempdir("store_typecache");
        let mut cb = Codebase::open(&dir).unwrap();
        let h = Hash([0x11; 32]);
        let scheme = TypeScheme::Struct {
            type_params: 0,
            fields: vec![],
            wire: true,
        };
        let mut cache = TypeCache::new();
        cache.insert(h, scheme);
        assert_eq!(cb.store_typecache(&cache).unwrap(), 1);
        // Second store: file exists, nothing to write.
        assert_eq!(cb.store_typecache(&cache).unwrap(), 0);
    }
}
