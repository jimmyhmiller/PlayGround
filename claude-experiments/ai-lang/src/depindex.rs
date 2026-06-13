//! Forward + reverse dependency index over content-addressed definitions.
//!
//! The kernel walks dependencies *forward* on demand
//! (`knowledge::walk_def_deps`). The editing layer needs the *inverse*: given a
//! hash, who references it. This module builds and persists both directions.
//!
//! It lives strictly above the hashing line: it never touches `hash.rs`,
//! `codec.rs`, or the `ast.rs` identity model. The persisted file under
//! `<root>/index/` is a pure cache and can always be regenerated from `defs/`
//! via [`DependencyIndex::rebuild_from_codebase`].
//!
//! ## Forward edge set of a Def
//!
//! [`def_dependencies`] is the single authority for "what does this Def
//! reference". It collects:
//!   - every `TopRef` in the body, plus every `TypeRef` in the signature
//!     (params / ret / struct fields / enum payloads, recursing through `Apply`
//!     and `FnType`), plus every `struct_ref`/`enum_ref` carried on
//!     constructor / `Field` / `Match` / `Try` nodes, plus every nested lambda's
//!     content hash. This part is delegated to [`knowledge::walk_def_deps`] so
//!     the walker is never duplicated.
//!   - every hash embedded in a `core/net.at#<hex>` (optionally `#<hex>#<hex>`)
//!     `BuiltinRef` name. `walk_def_deps` drops `BuiltinRef` entirely, so this
//!     is collected here via [`resolve::parse_at_builtin_name`]. A
//!     prefix-matching builtin name with malformed hex is a HARD ERROR (panic),
//!     never a silent skip.

use crate::ast::{Def, Expr, MatchArm};
use crate::hash::Hash;
use crate::knowledge::walk_def_deps;
use crate::resolve::parse_at_builtin_name;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;

/// Errors from persisting / loading the index.
#[derive(Debug)]
pub enum DepIndexError {
    Io(io::Error),
    /// A persisted line could not be parsed. Carries the offending text.
    Parse(String),
    /// Failed to load or decode a stored definition during a rebuild.
    Codebase(crate::codebase::CodebaseError),
}

impl From<io::Error> for DepIndexError {
    fn from(e: io::Error) -> Self {
        DepIndexError::Io(e)
    }
}

impl From<crate::codebase::CodebaseError> for DepIndexError {
    fn from(e: crate::codebase::CodebaseError) -> Self {
        DepIndexError::Codebase(e)
    }
}

impl std::fmt::Display for DepIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepIndexError::Io(e) => write!(f, "io error: {}", e),
            DepIndexError::Parse(s) => write!(f, "index parse error: {}", s),
            DepIndexError::Codebase(e) => write!(f, "codebase error: {}", e),
        }
    }
}

impl std::error::Error for DepIndexError {}

/// The complete set of definition hashes a `Def` references.
///
/// See the module docs for exactly what is collected. Panics if a
/// `core/net.at#...` builtin name contains malformed hex.
pub fn def_dependencies(def: &Def) -> BTreeSet<Hash> {
    // (a)+(b)+(c): TopRef, TypeRef, struct_ref/enum_ref, nested-lambda hashes —
    // the kernel's own forward walker is the single source of truth for these.
    let mut out: Vec<Hash> = Vec::new();
    let mut seen: HashSet<Hash> = HashSet::new();
    walk_def_deps(def, &mut out, &mut seen);
    let mut acc: BTreeSet<Hash> = out.into_iter().collect();

    // (d): hashes embedded in `core/net.at#<hex>` builtin names. The kernel
    // walker does not look at BuiltinRef, so collect those here.
    if let Def::Fn { body, .. } = def {
        collect_at_builtin_deps(body, &mut acc);
    }

    acc
}

/// Walk an expression collecting hashes embedded in `core/net.at#<hex>`
/// builtin names. This is intentionally narrow: it does NOT re-collect TopRef /
/// struct_ref / enum_ref (those come from `knowledge::walk_def_deps`); it only
/// inspects `BuiltinRef` strings, which the kernel walker discards.
fn collect_at_builtin_deps(expr: &Expr, acc: &mut BTreeSet<Hash>) {
    match expr {
        Expr::BuiltinRef(name) => {
            // Only `#`-bearing builtin names carry embedded hashes:
            // `core/net.at#<hex>` and its async twin
            // `core/net.at_async#<hex>`. We gate on the family prefix and
            // treat any `#` after it as load-bearing.
            if crate::resolve::is_at_family_builtin(name) {
                match parse_at_builtin_name(name) {
                    Some((primary, secondary)) => {
                        acc.insert(primary);
                        if let Some(s) = secondary {
                            acc.insert(s);
                        }
                    }
                    None => panic!(
                        "depindex: malformed hash-bearing builtin name: {:?}",
                        name
                    ),
                }
            }
        }
        Expr::Call(callee, args) => {
            collect_at_builtin_deps(callee, acc);
            for a in args {
                collect_at_builtin_deps(a, acc);
            }
        }
        Expr::Lambda { body, .. } => collect_at_builtin_deps(body, acc),
        Expr::Let { value, body } => {
            collect_at_builtin_deps(value, acc);
            collect_at_builtin_deps(body, acc);
        }
        Expr::Defer { cleanup, body } => {
            collect_at_builtin_deps(cleanup, acc);
            collect_at_builtin_deps(body, acc);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_at_builtin_deps(cond, acc);
            collect_at_builtin_deps(then_branch, acc);
            collect_at_builtin_deps(else_branch, acc);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                collect_at_builtin_deps(f, acc);
            }
        }
        Expr::Field { base, .. } => collect_at_builtin_deps(base, acc),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_at_builtin_deps(p, acc);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_at_builtin_deps(scrutinee, acc);
            for MatchArm { body, .. } in arms {
                collect_at_builtin_deps(body, acc);
            }
        }
        Expr::Try { expr, .. } => collect_at_builtin_deps(expr, acc),
        Expr::IntLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::FloatLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_) => {}
    }
}

/// Forward + reverse dependency index.
pub struct DependencyIndex {
    /// H -> hashes H references.
    forward: HashMap<Hash, BTreeSet<Hash>>,
    /// H -> hashes that reference H.
    reverse: HashMap<Hash, BTreeSet<Hash>>,
}

impl Default for DependencyIndex {
    fn default() -> Self {
        DependencyIndex::new()
    }
}

impl DependencyIndex {
    pub fn new() -> Self {
        DependencyIndex {
            forward: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Record `hash`'s forward edges and the matching reverse edges.
    ///
    /// Idempotent: re-adding the same hash recomputes the same forward set and
    /// re-inserts the same reverse entries (a `BTreeSet` ignores duplicates), so
    /// the net effect is unchanged.
    pub fn add_def(&mut self, hash: Hash, def: &Def) {
        let deps = def_dependencies(def);
        for dep in &deps {
            self.reverse.entry(*dep).or_default().insert(hash);
        }
        self.forward.insert(hash, deps);
    }

    /// Direct forward dependencies of `hash` (empty if unknown).
    pub fn dependencies(&self, hash: &Hash) -> BTreeSet<Hash> {
        self.forward.get(hash).cloned().unwrap_or_default()
    }

    /// Direct dependents of `hash` (empty if unknown).
    pub fn dependents(&self, hash: &Hash) -> BTreeSet<Hash> {
        self.reverse.get(hash).cloned().unwrap_or_default()
    }

    /// Forward transitive closure, excluding the start hash.
    pub fn transitive_dependencies(&self, hash: &Hash) -> BTreeSet<Hash> {
        self.closure(hash, Direction::Forward)
    }

    /// Reverse transitive closure (the update-propagation cone), excluding the
    /// start hash.
    pub fn transitive_dependents(&self, hash: &Hash) -> BTreeSet<Hash> {
        self.closure(hash, Direction::Reverse)
    }

    fn closure(&self, start: &Hash, dir: Direction) -> BTreeSet<Hash> {
        let map = match dir {
            Direction::Forward => &self.forward,
            Direction::Reverse => &self.reverse,
        };
        let mut seen = BTreeSet::new();
        let mut stack: Vec<Hash> = map.get(start).into_iter().flatten().copied().collect();
        while let Some(h) = stack.pop() {
            if h == *start {
                continue;
            }
            if seen.insert(h) {
                if let Some(next) = map.get(&h) {
                    for n in next {
                        if !seen.contains(n) {
                            stack.push(*n);
                        }
                    }
                }
            }
        }
        seen.remove(start);
        seen
    }

    /// Rebuild the whole index from the on-disk store: scan every `defs/*.def`,
    /// decode it, and add its edges. This is the source-of-truth rebuild; the
    /// persisted index is only a cache of this.
    pub fn rebuild_from_codebase(cb: &crate::codebase::Codebase) -> Result<Self, DepIndexError> {
        let mut index = DependencyIndex::new();
        let defs_dir = cb.root().join("defs");
        if !defs_dir.exists() {
            return Ok(index);
        }
        for entry in fs::read_dir(&defs_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("def") {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            let hash = match hash_from_hex(stem) {
                Some(h) => h,
                None => {
                    return Err(DepIndexError::Parse(format!(
                        "def filename is not a 64-char hex hash: {}",
                        stem
                    )))
                }
            };
            let def = cb.load_def(&hash)?;
            index.add_def(hash, &def);
        }
        Ok(index)
    }

    /// Persist the forward map under `<root>/index/deps`. The reverse map is
    /// derived from forward on load, so it is not persisted.
    ///
    /// Line format: `<forward_hex>\t<dep_hex>,<dep_hex>,...` (no trailing comma;
    /// a node with zero deps still gets a line with an empty dep list, so the
    /// forward key set round-trips exactly).
    pub fn save(&self, root: &Path) -> io::Result<()> {
        let dir = root.join("index");
        fs::create_dir_all(&dir)?;
        let mut lines: Vec<String> = self
            .forward
            .iter()
            .map(|(h, deps)| {
                let dep_str = deps
                    .iter()
                    .map(|d| d.to_hex())
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{}\t{}", h.to_hex(), dep_str)
            })
            .collect();
        lines.sort();
        fs::write(dir.join("deps"), lines.join("\n"))?;
        Ok(())
    }

    /// Load the index from `<root>/index/deps`, rebuilding the reverse map from
    /// forward edges. Treat the file as a pure cache: if it is absent the caller
    /// should fall back to [`DependencyIndex::rebuild_from_codebase`].
    pub fn load(root: &Path) -> io::Result<Self> {
        let path = root.join("index").join("deps");
        let text = fs::read_to_string(&path)?;
        let mut index = DependencyIndex::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let (head, rest) = line.split_once('\t').ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("index line missing tab separator: {:?}", line),
                )
            })?;
            let hash = hash_from_hex(head).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("index line has malformed forward hash: {:?}", head),
                )
            })?;
            let mut deps = BTreeSet::new();
            for dep_hex in rest.split(',') {
                if dep_hex.is_empty() {
                    continue;
                }
                let dep = hash_from_hex(dep_hex).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("index line has malformed dep hash: {:?}", dep_hex),
                    )
                })?;
                deps.insert(dep);
            }
            for dep in &deps {
                index.reverse.entry(*dep).or_default().insert(hash);
            }
            index.forward.insert(hash, deps);
        }
        Ok(index)
    }
}

#[derive(Copy, Clone)]
enum Direction {
    Forward,
    Reverse,
}

/// Parse a 64-char lowercase hex string into a `Hash`. Returns `None` on any
/// length or character error.
fn hash_from_hex(hex: &str) -> Option<Hash> {
    if hex.len() != 64 {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_val(bytes[2 * i])?;
        let lo = hex_val(bytes[2 * i + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(Hash(out))
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebase::Codebase;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unique temp dir under the OS temp dir (no `tempfile` dev-dependency in
    /// this crate). Cleaned up on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("ai_lang_depindex_{}_{}_{}", pid, nanos, n));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Resolve `source` and store it into a fresh codebase, returning the
    /// codebase plus a name->hash map.
    fn build_codebase(tmp: &TempDir, source: &str) -> Codebase {
        let module = parse_module(source).expect("parse");
        let rm = resolve_module(&module).expect("resolve");
        let mut cb = Codebase::open(tmp.path()).expect("open");
        cb.store_resolved_module(&rm).expect("store");
        cb
    }

    fn hash_of(cb: &Codebase, name: &str) -> Hash {
        cb.get_name(name).unwrap_or_else(|| {
            panic!(
                "no name {:?} in codebase: {:?}",
                name,
                cb.names().keys().collect::<Vec<_>>()
            )
        })
    }

    #[test]
    fn caller_depends_on_callee_and_callee_has_dependent() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let cb = build_codebase(&tmp, src);
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");

        let leaf = hash_of(&cb, "leaf");
        let caller = hash_of(&cb, "caller");

        assert!(
            index.dependencies(&caller).contains(&leaf),
            "caller should depend on leaf"
        );
        assert!(
            index.dependents(&leaf).contains(&caller),
            "leaf should have caller as a dependent"
        );
        // leaf references nothing user-defined.
        assert!(
            index.dependencies(&leaf).is_empty(),
            "leaf should have no user-def dependencies, got {:?}",
            index.dependencies(&leaf)
        );
    }

    #[test]
    fn transitive_dependents_form_the_cone() {
        let tmp = TempDir::new();
        // c <- b <- a   (a calls b, b calls c)
        let src = "
            def c(x: Int) -> Int = x
            def b(x: Int) -> Int = c(x)
            def a(x: Int) -> Int = b(x)
        ";
        let cb = build_codebase(&tmp, src);
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");

        let a = hash_of(&cb, "a");
        let b = hash_of(&cb, "b");
        let c = hash_of(&cb, "c");

        let cone = index.transitive_dependents(&c);
        assert!(cone.contains(&b), "cone of c should contain b");
        assert!(cone.contains(&a), "cone of c should contain a");
        assert!(!cone.contains(&c), "cone must exclude the start hash");

        let fwd = index.transitive_dependencies(&a);
        assert!(fwd.contains(&b), "fwd closure of a should contain b");
        assert!(fwd.contains(&c), "fwd closure of a should contain c");
        assert!(!fwd.contains(&a), "fwd closure must exclude the start hash");
    }

    #[test]
    fn save_load_round_trip_preserves_forward_edges() {
        let tmp = TempDir::new();
        let src = "
            def c(x: Int) -> Int = x
            def b(x: Int) -> Int = c(x)
            def a(x: Int) -> Int = b(x)
        ";
        let cb = build_codebase(&tmp, src);
        let index = DependencyIndex::rebuild_from_codebase(&cb).expect("rebuild");

        index.save(tmp.path()).expect("save");
        let loaded = DependencyIndex::load(tmp.path()).expect("load");

        // Forward maps must be identical.
        assert_eq!(
            index.forward, loaded.forward,
            "forward edges should round-trip exactly"
        );

        // Reverse map must be correctly derived: check the cone still works.
        let c = hash_of(&cb, "c");
        let a = hash_of(&cb, "a");
        let b = hash_of(&cb, "b");
        let cone = loaded.transitive_dependents(&c);
        assert!(cone.contains(&a) && cone.contains(&b));
    }

    #[test]
    fn add_def_is_idempotent() {
        let tmp = TempDir::new();
        let src = "
            def leaf(x: Int) -> Int = x
            def caller(y: Int) -> Int = leaf(y)
        ";
        let cb = build_codebase(&tmp, src);
        let leaf = hash_of(&cb, "leaf");
        let caller = hash_of(&cb, "caller");
        let caller_def = cb.load_def(&caller).expect("load caller");

        let mut index = DependencyIndex::new();
        index.add_def(caller, &caller_def);
        let deps_once = index.dependencies(&caller);
        let dependents_once = index.dependents(&leaf);

        // Re-add the same def.
        index.add_def(caller, &caller_def);
        assert_eq!(deps_once, index.dependencies(&caller));
        assert_eq!(dependents_once, index.dependents(&leaf));
        assert!(index.dependents(&leaf).contains(&caller));
    }
}
