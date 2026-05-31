//! Causal namespaces, branches, history and diff/merge (Phase 4).
//!
//! This module turns the flat single-namespace codebase into a branchable,
//! causally-versioned one, mirroring modern Unison's projects/branches model:
//! a branch is a copy with full history, `merge` brings updated defs across, and
//! O(1) `branch`/`undo` fall straight out of content addressing.
//!
//! It lives strictly above the hashing line: it never touches `hash.rs`,
//! `codec.rs`, or the `ast.rs` identity model. It hashes only the NAME layer
//! (name → def-hash pairs), never the defs themselves, so it cannot affect any
//! definition's identity.
//!
//! ## The three structures
//!
//! 1. **Namespace snapshot** — a `name → hash` map. (Dotted names like
//!    `math.float.sqrt` are just string keys, so a "hierarchical" namespace is
//!    representable today without a tree rewrite; nothing here precludes a real
//!    tree later.) Its Merkle identity is the [`namespace_hash`]: blake3 over the
//!    sorted `(name, hash)` pairs. Two snapshots with the same set of bindings
//!    have the same `namespace_hash`.
//!
//! 2. **Causal node** — `{ namespace_hash, parents, snapshot }`. Its
//!    `causal_hash` = blake3 over `namespace_hash` + the parent `causal_hash`es.
//!    A linear edit chains one parent; a merge has two. The causal hash is the
//!    Merkle identity of a *namespace state in history*: same bindings reached
//!    by a different path have different causal hashes.
//!
//! 3. **Branch** — a named pointer at a single `causal_hash` (its head).
//!    Branching is O(1): the new branch head is the source head, no defs copied
//!    (defs are shared, content-addressed). History is the parent walk; undo
//!    moves the head back to its parent.
//!
//! ## On-disk layout
//!
//! ```text
//! <root>/
//!   causal/<causal_hash_hex>      one causal node per state (see CODEC below)
//!   branches/<branchname>         a file holding the branch head's causal_hex
//!   HEAD                          the active branch name (default "main")
//!   names.txt                     the active branch's snapshot, kept for the
//!                                 existing name API + backward compatibility
//! ```
//!
//! ## Backward compatibility (load-bearing)
//!
//! The existing `Codebase` name methods (`get_name`/`set_name`/`remove_name`/
//! `names`/`store_resolved_module`) are unchanged in signature and keep
//! operating on the CURRENT branch's snapshot, serialized to `names.txt` exactly
//! as before. The causal layer is *additive*: opening a codebase that has a
//! `names.txt` but no `branches/` dir creates `main` from that snapshot
//! ([`init_or_migrate`]), so old codebases and old tests just keep working.
//!
//! ## Commit granularity (documented decision)
//!
//! **Per-operation auto-commit.** Every name-changing call
//! (`set_name`/`remove_name`/`store_resolved_module`) auto-commits a new causal
//! snapshot to the active branch (a no-op when the snapshot is unchanged). This
//! is the simplest model and makes `undo` intuitive: one undo reverses one
//! name-changing operation. (A type-changing `update`/refactor that re-points a
//! whole cone moves several names via repeated `set_name` calls; each is its own
//! commit, so undo peels them back one at a time — `update` callers that want a
//! single atomic commit can wrap the batch with `set_auto_commit(false)` and a
//! final explicit [`commit_namespace`]. The default favours fine-grained undo.)

use crate::codebase::{Codebase, CodebaseError};
use crate::hash::Hash;

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// =============================================================================
// Errors
// =============================================================================

/// Errors from the causal-namespace layer. House style of the other edit-layer
/// modules: a small typed enum with `Display` + `std::error::Error` and `From`
/// for the wrapped codebase / io errors.
#[derive(Debug)]
pub enum NamespaceError {
    Io(io::Error),
    Codebase(CodebaseError),
    /// A named branch does not exist.
    NoSuchBranch(String),
    /// `branch <new>` refused because `<new>` already exists.
    BranchExists(String),
    /// A causal node referenced by a branch head / parent link is missing or
    /// unreadable on disk. The causal store is corrupt.
    MissingCausal(Hash),
    /// A persisted causal node / head / HEAD file could not be parsed.
    Corrupt(String),
    /// `undo` was asked to move past the root of a branch's history (the initial
    /// commit has no parent).
    NoParent(String),
}

impl From<io::Error> for NamespaceError {
    fn from(e: io::Error) -> Self {
        NamespaceError::Io(e)
    }
}

impl From<CodebaseError> for NamespaceError {
    fn from(e: CodebaseError) -> Self {
        NamespaceError::Codebase(e)
    }
}

impl std::fmt::Display for NamespaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NamespaceError::Io(e) => write!(f, "io error: {}", e),
            NamespaceError::Codebase(e) => write!(f, "codebase error: {}", e),
            NamespaceError::NoSuchBranch(b) => write!(f, "no such branch: {}", b),
            NamespaceError::BranchExists(b) => {
                write!(f, "branch already exists (refusing to overwrite): {}", b)
            }
            NamespaceError::MissingCausal(h) => {
                write!(f, "causal node {} is missing from the store", h)
            }
            NamespaceError::Corrupt(s) => write!(f, "corrupt namespace metadata: {}", s),
            NamespaceError::NoParent(b) => write!(
                f,
                "nothing to undo on branch {}: at the root of history",
                b
            ),
        }
    }
}

impl std::error::Error for NamespaceError {}

// =============================================================================
// Namespace snapshot hashing (the Merkle identity of a name map)
// =============================================================================

/// Blake3 over the sorted `(name, hash)` pairs of a snapshot. Deterministic and
/// order-independent: the same set of bindings always yields the same hash.
///
/// Encoding per pair: `u32` big-endian name length, the name's UTF-8 bytes, then
/// the 32 raw hash bytes. Pairs are emitted in ascending name order.
pub fn namespace_hash(names: &HashMap<String, Hash>) -> Hash {
    let mut pairs: Vec<(&String, &Hash)> = names.iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    let mut buf: Vec<u8> = Vec::with_capacity(pairs.len() * 48);
    buf.extend_from_slice(&(pairs.len() as u32).to_be_bytes());
    for (name, hash) in pairs {
        buf.extend_from_slice(&(name.len() as u32).to_be_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(hash.as_bytes());
    }
    Hash::of_bytes(&buf)
}

// =============================================================================
// Causal node
// =============================================================================

/// A causal node: a namespace snapshot plus links to the state(s) it descends
/// from. Linear history has one parent; a merge has two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalNode {
    /// The Merkle identity of `snapshot` (see [`namespace_hash`]).
    pub namespace_hash: Hash,
    /// The causal_hash(es) of the parent state(s). Empty for the root commit.
    pub parents: Vec<Hash>,
    /// The actual `name → hash` bindings at this state.
    pub snapshot: HashMap<String, Hash>,
}

impl CausalNode {
    /// The causal hash: blake3 over `namespace_hash` followed by each parent
    /// `causal_hash` (in the order they were recorded). This binds a state both
    /// to *what* it names (`namespace_hash`) and *how it was reached* (parents),
    /// so two identical snapshots reached differently get distinct identities.
    pub fn causal_hash(&self) -> Hash {
        let mut buf: Vec<u8> = Vec::with_capacity(32 + self.parents.len() * 32 + 4);
        buf.extend_from_slice(self.namespace_hash.as_bytes());
        buf.extend_from_slice(&(self.parents.len() as u32).to_be_bytes());
        for p in &self.parents {
            buf.extend_from_slice(p.as_bytes());
        }
        Hash::of_bytes(&buf)
    }
}

// ---- Causal node codec (a small self-describing text format) ----
//
//   line 1: "ns\t<namespace_hash_hex>"
//   line 2: "parents\t<hex>,<hex>,..."     (empty list => "parents\t")
//   then one line per binding: "<hex_hash>\t<name>"
//     (hash first so a name containing a tab is still unambiguous: the name is
//      everything after the first tab.)
//
// We persist the full snapshot inside each causal node so history is
// self-contained and a snapshot can be restored without consulting names.txt.

fn encode_causal(node: &CausalNode) -> String {
    let mut out = String::new();
    out.push_str("ns\t");
    out.push_str(&node.namespace_hash.to_hex());
    out.push('\n');
    out.push_str("parents\t");
    out.push_str(
        &node
            .parents
            .iter()
            .map(|p| p.to_hex())
            .collect::<Vec<_>>()
            .join(","),
    );
    out.push('\n');
    let mut pairs: Vec<(&String, &Hash)> = node.snapshot.iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    for (name, hash) in pairs {
        out.push_str(&hash.to_hex());
        out.push('\t');
        out.push_str(name);
        out.push('\n');
    }
    out
}

fn decode_causal(text: &str) -> Result<CausalNode, NamespaceError> {
    let mut lines = text.lines();
    let ns_line = lines
        .next()
        .ok_or_else(|| NamespaceError::Corrupt("empty causal node".into()))?;
    let ns_hex = ns_line
        .strip_prefix("ns\t")
        .ok_or_else(|| NamespaceError::Corrupt(format!("bad ns line: {:?}", ns_line)))?;
    let namespace_hash = hash_from_hex(ns_hex)
        .ok_or_else(|| NamespaceError::Corrupt(format!("bad namespace hash: {:?}", ns_hex)))?;

    let parents_line = lines
        .next()
        .ok_or_else(|| NamespaceError::Corrupt("missing parents line".into()))?;
    let parents_str = parents_line
        .strip_prefix("parents\t")
        .ok_or_else(|| NamespaceError::Corrupt(format!("bad parents line: {:?}", parents_line)))?;
    let mut parents = Vec::new();
    for p in parents_str.split(',') {
        if p.is_empty() {
            continue;
        }
        parents.push(
            hash_from_hex(p)
                .ok_or_else(|| NamespaceError::Corrupt(format!("bad parent hash: {:?}", p)))?,
        );
    }

    let mut snapshot = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (hex, name) = line
            .split_once('\t')
            .ok_or_else(|| NamespaceError::Corrupt(format!("bad binding line: {:?}", line)))?;
        let hash = hash_from_hex(hex)
            .ok_or_else(|| NamespaceError::Corrupt(format!("bad binding hash: {:?}", hex)))?;
        snapshot.insert(name.to_owned(), hash);
    }

    Ok(CausalNode {
        namespace_hash,
        parents,
        snapshot,
    })
}

// =============================================================================
// On-disk paths
// =============================================================================

fn causal_dir(root: &Path) -> PathBuf {
    root.join("causal")
}
fn causal_path(root: &Path, h: &Hash) -> PathBuf {
    causal_dir(root).join(h.to_hex())
}
fn branches_dir(root: &Path) -> PathBuf {
    root.join("branches")
}
fn branch_path(root: &Path, name: &str) -> PathBuf {
    branches_dir(root).join(name)
}
fn head_path(root: &Path) -> PathBuf {
    root.join("HEAD")
}

/// Persist a causal node under its causal hash (idempotent: content-addressed,
/// so re-writing the same node is a harmless no-op). Returns the causal hash.
fn store_causal(root: &Path, node: &CausalNode) -> Result<Hash, NamespaceError> {
    fs::create_dir_all(causal_dir(root))?;
    let h = node.causal_hash();
    let path = causal_path(root, &h);
    if !path.exists() {
        fs::write(&path, encode_causal(node))?;
    }
    Ok(h)
}

fn load_causal(root: &Path, h: &Hash) -> Result<CausalNode, NamespaceError> {
    let path = causal_path(root, h);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(NamespaceError::MissingCausal(*h));
        }
        Err(e) => return Err(e.into()),
    };
    decode_causal(&text)
}

fn write_branch_head(root: &Path, name: &str, head: &Hash) -> Result<(), NamespaceError> {
    fs::create_dir_all(branches_dir(root))?;
    fs::write(branch_path(root, name), head.to_hex())?;
    Ok(())
}

fn read_branch_head(root: &Path, name: &str) -> Result<Hash, NamespaceError> {
    let text = match fs::read_to_string(branch_path(root, name)) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(NamespaceError::NoSuchBranch(name.to_owned()));
        }
        Err(e) => return Err(e.into()),
    };
    hash_from_hex(text.trim())
        .ok_or_else(|| NamespaceError::Corrupt(format!("bad branch head for {}: {:?}", name, text)))
}

fn write_head_file(root: &Path, branch: &str) -> Result<(), NamespaceError> {
    fs::write(head_path(root), branch)?;
    Ok(())
}

fn read_head_file(root: &Path) -> Result<Option<String>, NamespaceError> {
    match fs::read_to_string(head_path(root)) {
        Ok(t) => Ok(Some(t.trim().to_owned())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// =============================================================================
// Migration / initialisation (called from Codebase::open)
// =============================================================================

/// Initialise the causal layer for a freshly-opened codebase, or migrate an
/// existing flat one. Idempotent and backward-compatible:
///
///   - If `HEAD` + `branches/main` already exist, do nothing (already migrated).
///   - Otherwise create branch `main` whose head is a root causal node built
///     from the CURRENT `names.txt` snapshot (no parent), and point `HEAD` at
///     `main`. This is the migration path for a codebase that has a `names.txt`
///     but no `branches/` dir; the existing name API keeps working and now
///     operates on `main`.
///
/// Returns a `CodebaseError` (not `NamespaceError`) because it is called from
/// `Codebase::open`, whose signature is fixed.
pub fn init_or_migrate(cb: &mut Codebase) -> Result<(), CodebaseError> {
    do_init_or_migrate(cb).map_err(|e| CodebaseError::Namespace(e.to_string()))
}

fn do_init_or_migrate(cb: &mut Codebase) -> Result<(), NamespaceError> {
    let root = cb.root().to_path_buf();
    let head = read_head_file(&root)?;
    let main_exists = branch_path(&root, "main").exists();
    if head.is_some() && main_exists {
        // Already initialised. Nothing to do; the existing names.txt is the
        // active branch's working snapshot.
        return Ok(());
    }
    // Build the root causal node from the current names.txt snapshot.
    let snapshot = cb.names().clone();
    let node = CausalNode {
        namespace_hash: namespace_hash(&snapshot),
        parents: Vec::new(),
        snapshot,
    };
    let head_hash = store_causal(&root, &node)?;
    write_branch_head(&root, "main", &head_hash)?;
    if read_head_file(&root)?.is_none() {
        write_head_file(&root, "main")?;
    }
    Ok(())
}

// =============================================================================
// Branch / HEAD reads
// =============================================================================

/// The active branch name (the contents of `HEAD`, defaulting to "main").
pub fn current_branch(cb: &Codebase) -> Result<String, NamespaceError> {
    Ok(read_head_file(cb.root())?.unwrap_or_else(|| "main".to_owned()))
}

/// Every branch name, sorted.
pub fn branches(cb: &Codebase) -> Result<Vec<String>, NamespaceError> {
    let dir = branches_dir(cb.root());
    let mut out = Vec::new();
    match fs::read_dir(&dir) {
        Ok(entries) => {
            for e in entries {
                let e = e?;
                if let Some(name) = e.file_name().to_str() {
                    out.push(name.to_owned());
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    out.sort();
    Ok(out)
}

/// The head causal hash of `branch`.
pub fn branch_head(cb: &Codebase, branch: &str) -> Result<Hash, NamespaceError> {
    read_branch_head(cb.root(), branch)
}

// =============================================================================
// commit_namespace — snapshot the current name map into a new causal node
// =============================================================================

/// Snapshot the current (active-branch) name map into a new causal node whose
/// parent is the branch's current head, and advance the head.
///
/// A no-op (no new node, head unchanged) when the current snapshot is already
/// the head's snapshot — so repeated commits with no name change don't pile up
/// duplicate history, and undo never has to step over empty commits.
pub fn commit_namespace(cb: &mut Codebase) -> Result<(), NamespaceError> {
    let root = cb.root().to_path_buf();
    let branch = current_branch(cb)?;

    // Ensure the branch head exists (it does after init_or_migrate, but be
    // defensive if a caller commits before any migration ran).
    let parent = match read_branch_head(&root, &branch) {
        Ok(h) => Some(h),
        Err(NamespaceError::NoSuchBranch(_)) => None,
        Err(e) => return Err(e),
    };

    let snapshot = cb.names().clone();
    let ns_hash = namespace_hash(&snapshot);

    // If the head already has this exact namespace_hash, nothing changed.
    if let Some(parent_hash) = parent {
        let parent_node = load_causal(&root, &parent_hash)?;
        if parent_node.namespace_hash == ns_hash {
            return Ok(());
        }
    }

    let node = CausalNode {
        namespace_hash: ns_hash,
        parents: parent.into_iter().collect(),
        snapshot,
    };
    let new_head = store_causal(&root, &node)?;
    write_branch_head(&root, &branch, &new_head)?;
    Ok(())
}

// =============================================================================
// branch — O(1) fork
// =============================================================================

/// Create branch `new_name` pointing at the same causal node as `from_branch`
/// (default: the current branch). O(1): no def copying — defs are shared,
/// content-addressed. The new branch shares the source's full history.
///
/// Refuses ([`NamespaceError::BranchExists`]) if `new_name` already exists.
pub fn branch(
    cb: &Codebase,
    new_name: &str,
    from_branch: Option<&str>,
) -> Result<Hash, NamespaceError> {
    let root = cb.root();
    if branch_path(root, new_name).exists() {
        return Err(NamespaceError::BranchExists(new_name.to_owned()));
    }
    let src = match from_branch {
        Some(b) => b.to_owned(),
        None => current_branch(cb)?,
    };
    let head = read_branch_head(root, &src)?;
    write_branch_head(root, new_name, &head)?;
    Ok(head)
}

// =============================================================================
// switch — change active branch and reload names.txt from its head snapshot
// =============================================================================

/// Switch the active branch to `branch_name`: update `HEAD` and rewrite
/// `names.txt` (and the in-memory name map) from that branch's head snapshot, so
/// the existing name API sees the right branch. Reloading the snapshot does NOT
/// create a new commit (auto-commit is suppressed during the rewrite).
pub fn switch(cb: &mut Codebase, branch_name: &str) -> Result<(), NamespaceError> {
    let root = cb.root().to_path_buf();
    let head = read_branch_head(&root, branch_name)?;
    let node = load_causal(&root, &head)?;
    write_head_file(&root, branch_name)?;
    cb.replace_names(node.snapshot)?;
    Ok(())
}

// =============================================================================
// history / undo
// =============================================================================

/// The causal-hash chain of `branch` (default current), head first, walking
/// `parents[0]` back to the root. (For merge nodes only the first parent is
/// followed, which gives the branch's own mainline; the full DAG is still on
/// disk and reachable via [`load_node`].)
pub fn history(cb: &Codebase, branch: Option<&str>) -> Result<Vec<Hash>, NamespaceError> {
    let root = cb.root();
    let b = match branch {
        Some(b) => b.to_owned(),
        None => current_branch(cb)?,
    };
    let mut out = Vec::new();
    let mut cur = Some(read_branch_head(root, &b)?);
    while let Some(h) = cur {
        out.push(h);
        let node = load_causal(root, &h)?;
        cur = node.parents.first().copied();
    }
    Ok(out)
}

/// Load a causal node by its causal hash (for callers that want the snapshot or
/// parents of a specific history entry).
pub fn load_node(cb: &Codebase, causal: &Hash) -> Result<CausalNode, NamespaceError> {
    load_causal(cb.root(), causal)
}

/// Move the current branch head back to its (first) parent and reload names from
/// the parent snapshot. Returns the new head causal hash.
///
/// The undone causal node and every def remain on disk (immutability), so the
/// move is itself reversible by re-committing. Errors with
/// [`NamespaceError::NoParent`] at the root of history.
pub fn undo(cb: &mut Codebase) -> Result<Hash, NamespaceError> {
    let root = cb.root().to_path_buf();
    let branch = current_branch(cb)?;
    let head = read_branch_head(&root, &branch)?;
    let node = load_causal(&root, &head)?;
    let parent = node
        .parents
        .first()
        .copied()
        .ok_or_else(|| NamespaceError::NoParent(branch.clone()))?;
    let parent_node = load_causal(&root, &parent)?;
    write_branch_head(&root, &branch, &parent)?;
    cb.replace_names(parent_node.snapshot)?;
    Ok(parent)
}

// =============================================================================
// diff
// =============================================================================

/// A structural diff between two namespace snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NamespaceDiff {
    /// Names present in `b` but not `a`, with their `b` hash. Sorted by name.
    pub added: Vec<(String, Hash)>,
    /// Names present in `a` but not `b`, with their `a` hash. Sorted by name.
    pub removed: Vec<(String, Hash)>,
    /// Names in both whose hash differs: `(name, a_hash, b_hash)`. Sorted by name.
    pub changed: Vec<(String, Hash, Hash)>,
}

impl NamespaceDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Diff the head snapshots of `branch_a` (the base) against `branch_b`: what
/// `b` added, removed, or changed relative to `a`. Cheap: a comparison of the
/// two name maps.
pub fn diff(
    cb: &Codebase,
    branch_a: &str,
    branch_b: &str,
) -> Result<NamespaceDiff, NamespaceError> {
    let root = cb.root();
    let a = load_causal(root, &read_branch_head(root, branch_a)?)?.snapshot;
    let b = load_causal(root, &read_branch_head(root, branch_b)?)?.snapshot;
    Ok(diff_snapshots(&a, &b))
}

/// Pure snapshot diff (exposed for callers that already hold two maps).
pub fn diff_snapshots(a: &HashMap<String, Hash>, b: &HashMap<String, Hash>) -> NamespaceDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    for (name, bh) in b {
        match a.get(name) {
            None => added.push((name.clone(), *bh)),
            Some(ah) if ah != bh => changed.push((name.clone(), *ah, *bh)),
            Some(_) => {}
        }
    }
    for (name, ah) in a {
        if !b.contains_key(name) {
            removed.push((name.clone(), *ah));
        }
    }
    added.sort_by(|x, y| x.0.cmp(&y.0));
    removed.sort_by(|x, y| x.0.cmp(&y.0));
    changed.sort_by(|x, y| x.0.cmp(&y.0));
    NamespaceDiff {
        added,
        removed,
        changed,
    }
}

// =============================================================================
// merge — 3-way over the name layer using the common causal ancestor
// =============================================================================

/// A naming conflict surfaced by [`merge`]: the same name was changed to two
/// different hashes on both sides since their common ancestor. We never resolve
/// this silently — it is returned for the caller (agent or human) to settle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeConflict {
    pub name: String,
    /// The hash at the common ancestor (`None` if the name was added on both
    /// sides independently — an add/add conflict).
    pub base: Option<Hash>,
    /// `into`'s hash for this name.
    pub into_hash: Hash,
    /// `from`'s hash for this name.
    pub from_hash: Hash,
}

/// The outcome of a [`merge`]: either a clean merge (a new merge causal node was
/// created on `into`) or a list of conflicts (nothing was committed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeResult {
    /// Merge succeeded; carries the new merge head and the names brought across
    /// from `from` (added or changed into `into`).
    Merged {
        new_head: Hash,
        brought: Vec<(String, Hash)>,
    },
    /// Merge could not complete: these names conflict. Nothing was committed;
    /// `into` is untouched.
    Conflicts(Vec<MergeConflict>),
}

/// 3-way merge of `from_branch` into `into_branch`, over the name layer only.
///
/// Using the common causal ancestor of the two heads as the base:
///   - names added in `from` (and absent or identical in `into`) appear in
///     `into`,
///   - names changed in `from` (and not conflictingly changed in `into`) take
///     `from`'s hash,
///   - names where both sides changed the same name to DIFFERENT hashes since
///     the base (or both added the same name with different hashes) are
///     CONFLICTS, returned structurally — never silently resolved.
///
/// Defs never conflict (different hashes coexist in `defs/`); only the name
/// layer can. On a clean merge a merge causal node with BOTH parents is created
/// on `into` and its head advanced. On conflict, nothing is committed.
pub fn merge(
    cb: &mut Codebase,
    from_branch: &str,
    into_branch: &str,
) -> Result<MergeResult, NamespaceError> {
    let root = cb.root().to_path_buf();
    let from_head = read_branch_head(&root, from_branch)?;
    let into_head = read_branch_head(&root, into_branch)?;

    let from = load_causal(&root, &from_head)?.snapshot;
    let into = load_causal(&root, &into_head)?.snapshot;

    // Common ancestor snapshot (empty map if none — treat everything as added).
    let base = match common_ancestor(&root, &from_head, &into_head)? {
        Some(anc) => load_causal(&root, &anc)?.snapshot,
        None => HashMap::new(),
    };

    let mut merged = into.clone();
    let mut conflicts: Vec<MergeConflict> = Vec::new();
    let mut brought: Vec<(String, Hash)> = Vec::new();

    // Walk every name `from` knows about and decide its fate in `into`.
    for (name, from_hash) in &from {
        let base_hash = base.get(name).copied();
        let into_hash = into.get(name).copied();
        let from_changed = base_hash != Some(*from_hash);
        if !from_changed {
            // `from` didn't touch this name relative to base — nothing to bring.
            continue;
        }
        match into_hash {
            // `into` doesn't have it (or removed it). If `into` also changed it
            // relative to base by removing it AND it existed at base, that's a
            // delete/modify situation — surface as a conflict so it isn't
            // silently re-added.
            None => {
                let into_changed = base_hash.is_some(); // removed in into
                if into_changed {
                    conflicts.push(MergeConflict {
                        name: name.clone(),
                        base: base_hash,
                        // No into hash; represent the removal by reusing from's
                        // for display purposes is misleading, so we only reach
                        // here when into removed a name from changed. Use base
                        // as into side marker via a distinct conflict isn't in
                        // the struct; keep into_hash = from_hash sentinel? No:
                        // surface with into_hash = base (the value into deleted).
                        into_hash: base_hash.unwrap(),
                        from_hash: *from_hash,
                    });
                } else {
                    merged.insert(name.clone(), *from_hash);
                    brought.push((name.clone(), *from_hash));
                }
            }
            Some(ih) => {
                if ih == *from_hash {
                    // Both ended at the same hash; no conflict, nothing to do.
                    continue;
                }
                let into_changed = base_hash != Some(ih);
                if into_changed {
                    // Both sides changed the same name to different hashes.
                    conflicts.push(MergeConflict {
                        name: name.clone(),
                        base: base_hash,
                        into_hash: ih,
                        from_hash: *from_hash,
                    });
                } else {
                    // Only `from` changed it; take from's hash.
                    merged.insert(name.clone(), *from_hash);
                    brought.push((name.clone(), *from_hash));
                }
            }
        }
    }

    if !conflicts.is_empty() {
        conflicts.sort_by(|a, b| a.name.cmp(&b.name));
        return Ok(MergeResult::Conflicts(conflicts));
    }

    brought.sort_by(|a, b| a.0.cmp(&b.0));

    // Create a merge causal node with both parents. If the merge changed
    // nothing in `into` (everything from was already present/identical), still
    // record a merge node so the DAG reflects the merge — but only if `from`
    // actually had distinct history; if into already contained from's head we
    // simply leave into untouched.
    if merged == into {
        // No content change. Still advance with a merge node ONLY if the heads
        // differ, to record provenance; otherwise nothing to do.
        if from_head == into_head {
            return Ok(MergeResult::Merged {
                new_head: into_head,
                brought,
            });
        }
    }

    let node = CausalNode {
        namespace_hash: namespace_hash(&merged),
        parents: vec![into_head, from_head],
        snapshot: merged.clone(),
    };
    let new_head = store_causal(&root, &node)?;
    write_branch_head(&root, into_branch, &new_head)?;

    // If we merged into the active branch, reflect it in names.txt.
    if current_branch(cb)? == into_branch {
        cb.replace_names(merged)?;
    }

    Ok(MergeResult::Merged { new_head, brought })
}

/// Lowest common ancestor of two causal heads, walking first-parent chains.
/// Returns `None` if the two histories share no ancestor (independent roots).
fn common_ancestor(
    root: &Path,
    a: &Hash,
    b: &Hash,
) -> Result<Option<Hash>, NamespaceError> {
    use std::collections::HashSet;
    // Collect all ancestors of `a` (including itself), following ALL parents so
    // a prior merge is correctly seen as a shared ancestor.
    let mut a_anc: HashSet<Hash> = HashSet::new();
    let mut stack = vec![*a];
    while let Some(h) = stack.pop() {
        if !a_anc.insert(h) {
            continue;
        }
        let node = load_causal(root, &h)?;
        for p in node.parents {
            stack.push(p);
        }
    }
    // BFS from `b` (all parents) to find the first hash that is in a_anc.
    let mut seen: HashSet<Hash> = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(*b);
    while let Some(h) = queue.pop_front() {
        if !seen.insert(h) {
            continue;
        }
        if a_anc.contains(&h) {
            return Ok(Some(h));
        }
        let node = load_causal(root, &h)?;
        for p in node.parents {
            queue.push_back(p);
        }
    }
    Ok(None)
}

// =============================================================================
// Hex helper (local; mirrors the other edit-layer modules)
// =============================================================================

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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("ai_lang_ns_{}_{}_{}_{}", tag, pid, nanos, n));
            let _ = std::fs::remove_dir_all(&dir);
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

    /// Store a small program into a fresh codebase and return it.
    fn build(dir: &Path, src: &str) -> Codebase {
        let m = parse_module(src).expect("parse");
        let rm = resolve_module(&m).expect("resolve");
        let mut cb = Codebase::open(dir).expect("open");
        cb.store_resolved_module(&rm).expect("store");
        cb
    }

    fn count_files(dir: &Path) -> usize {
        match std::fs::read_dir(dir) {
            Ok(it) => it.count(),
            Err(_) => 0,
        }
    }

    // ---- backward compat: old flow still works, operates on main ----

    #[test]
    fn existing_flow_still_works_on_main() {
        let dir = TempDir::new("compat");
        let cb = build(
            dir.path(),
            "def double(x: Int) -> Int = x * 2",
        );
        // get_name still works.
        assert!(cb.get_name("double").is_some());
        // We are on main.
        assert_eq!(current_branch(&cb).unwrap(), "main");
        // main exists and history has at least the commit.
        assert!(branches(&cb).unwrap().contains(&"main".to_owned()));
        assert!(!history(&cb, None).unwrap().is_empty());
    }

    #[test]
    fn migration_creates_main_from_existing_names_txt() {
        let dir = TempDir::new("migrate");
        // Hand-build a flat codebase: names.txt + a def, NO branches/ dir.
        {
            let cb = build(dir.path(), "def f(x: Int) -> Int = x");
            let _ = cb;
        }
        // Simulate a pre-Phase-4 codebase by deleting the causal metadata.
        let _ = std::fs::remove_dir_all(branches_dir(dir.path()));
        let _ = std::fs::remove_dir_all(causal_dir(dir.path()));
        let _ = std::fs::remove_file(head_path(dir.path()));
        assert!(!branches_dir(dir.path()).exists());

        // Reopen: migration must create main from names.txt.
        let cb = Codebase::open(dir.path()).expect("reopen");
        assert!(cb.get_name("f").is_some(), "name survives migration");
        assert_eq!(current_branch(&cb).unwrap(), "main");
        let main_head = branch_head(&cb, "main").unwrap();
        let node = load_node(&cb, &main_head).unwrap();
        assert!(node.snapshot.contains_key("f"));
    }

    // ---- branch is O(1) and shares defs ----

    #[test]
    fn branch_is_o1_and_shares_defs() {
        let dir = TempDir::new("branch");
        let cb = build(
            dir.path(),
            "def helper(x: Int) -> Int = x * 2
             def usesHelper(y: Int) -> Int = helper(y) + 1",
        );
        let defs_before = count_files(&dir.path().join("defs"));

        // Fork b from main.
        branch(&cb, "b", None).expect("branch");
        assert!(branches(&cb).unwrap().contains(&"b".to_owned()));

        // No def files were copied.
        let defs_after = count_files(&dir.path().join("defs"));
        assert_eq!(defs_before, defs_after, "branch must not copy defs");

        // b and main share the same head causal node -> same snapshot.
        assert_eq!(
            branch_head(&cb, "b").unwrap(),
            branch_head(&cb, "main").unwrap(),
            "fork points at the same causal node"
        );
        // Both resolve the same names to the same hashes.
        let main_snap = load_node(&cb, &branch_head(&cb, "main").unwrap())
            .unwrap()
            .snapshot;
        let b_snap = load_node(&cb, &branch_head(&cb, "b").unwrap())
            .unwrap()
            .snapshot;
        assert_eq!(main_snap, b_snap);
        assert_eq!(main_snap.get("helper"), cb.get_name("helper").as_ref());
    }

    #[test]
    fn branch_refuses_overwrite() {
        let dir = TempDir::new("branchdup");
        let cb = build(dir.path(), "def f(x: Int) -> Int = x");
        branch(&cb, "b", None).unwrap();
        match branch(&cb, "b", None) {
            Err(NamespaceError::BranchExists(b)) => assert_eq!(b, "b"),
            other => panic!("expected BranchExists, got {:?}", other),
        }
    }

    // ---- switch changes visible names; switching back restores ----

    #[test]
    fn switch_changes_and_restores_visible_names() {
        let dir = TempDir::new("switch");
        let mut cb = build(dir.path(), "def f(x: Int) -> Int = x");
        let f_hash = cb.get_name("f").unwrap();

        // Fork b, switch to it, add a new name only on b.
        branch(&cb, "b", None).unwrap();
        switch(&mut cb, "b").unwrap();
        assert_eq!(current_branch(&cb).unwrap(), "b");
        cb.set_name("only_on_b", f_hash).unwrap();
        assert!(cb.get_name("only_on_b").is_some());

        // Switch back to main: only_on_b is gone.
        switch(&mut cb, "main").unwrap();
        assert_eq!(current_branch(&cb).unwrap(), "main");
        assert!(cb.get_name("only_on_b").is_none(), "b-only name hidden on main");
        assert!(cb.get_name("f").is_some());

        // Switch to b again: it returns.
        switch(&mut cb, "b").unwrap();
        assert!(cb.get_name("only_on_b").is_some(), "b-only name restored on b");
    }

    // ---- diff ----

    #[test]
    fn diff_reports_added_and_changed() {
        let dir = TempDir::new("diff");
        let mut cb = build(dir.path(), "def f(x: Int) -> Int = x");
        let f_hash = cb.get_name("f").unwrap();

        branch(&cb, "b", None).unwrap();
        switch(&mut cb, "b").unwrap();
        // Add a brand-new name on b.
        cb.set_name("g", f_hash).unwrap();
        // Change f to point at a different hash (simulate an update).
        let other = Hash([0xAB; 32]);
        cb.set_name("f", other).unwrap();

        let d = diff(&cb, "main", "b").expect("diff");
        assert!(d.added.iter().any(|(n, _)| n == "g"), "g added on b");
        assert!(
            d.changed.iter().any(|(n, a, bnew)| n == "f" && *a == f_hash && *bnew == other),
            "f changed on b"
        );
        assert!(d.removed.is_empty());
    }

    #[test]
    fn diff_reports_removed() {
        let dir = TempDir::new("diffrm");
        let mut cb = build(dir.path(), "def f(x: Int) -> Int = x");
        branch(&cb, "b", None).unwrap();
        switch(&mut cb, "b").unwrap();
        cb.remove_name("f").unwrap();
        let d = diff(&cb, "main", "b").unwrap();
        assert!(d.removed.iter().any(|(n, _)| n == "f"));
    }

    // ---- history + undo ----

    #[test]
    fn undo_restores_prior_state_and_keeps_defs() {
        let dir = TempDir::new("undo");
        let mut cb = build(dir.path(), "def f(x: Int) -> Int = x");
        let f_orig = cb.get_name("f").unwrap();
        let defs_before = count_files(&dir.path().join("defs"));

        // Change f to a new hash (a new commit).
        let new_hash = Hash([0x11; 32]);
        cb.set_name("f", new_hash).unwrap();
        assert_eq!(cb.get_name("f"), Some(new_hash));

        let hist_before = history(&cb, None).unwrap();
        assert!(hist_before.len() >= 2, "two commits in history");

        // Undo: f reverts to its old hash.
        undo(&mut cb).unwrap();
        assert_eq!(cb.get_name("f"), Some(f_orig), "name reverted to old hash");

        // Def files all still exist (immutability).
        let defs_after = count_files(&dir.path().join("defs"));
        assert_eq!(defs_before, defs_after, "undo destroys no defs");
    }

    #[test]
    fn undo_at_root_errors() {
        let dir = TempDir::new("undoroot");
        // Empty codebase -> root commit only (the migration snapshot).
        let mut cb = Codebase::open(dir.path()).unwrap();
        match undo(&mut cb) {
            Err(NamespaceError::NoParent(_)) => {}
            other => panic!("expected NoParent at root, got {:?}", other),
        }
    }

    // ---- merge ----

    #[test]
    fn merge_brings_nonconflicting_changes() {
        let dir = TempDir::new("merge_ok");
        let mut cb = build(dir.path(), "def f(x: Int) -> Int = x");
        let f_hash = cb.get_name("f").unwrap();

        // Fork b, add g on b only.
        branch(&cb, "b", None).unwrap();
        switch(&mut cb, "b").unwrap();
        let g_hash = Hash([0x22; 32]);
        cb.set_name("g", g_hash).unwrap();

        // Merge b into main.
        let res = merge(&mut cb, "b", "main").expect("merge");
        match res {
            MergeResult::Merged { brought, .. } => {
                assert!(brought.iter().any(|(n, h)| n == "g" && *h == g_hash));
            }
            other => panic!("expected clean merge, got {:?}", other),
        }
        // main now has g.
        let main_snap = load_node(&cb, &branch_head(&cb, "main").unwrap())
            .unwrap()
            .snapshot;
        assert_eq!(main_snap.get("g"), Some(&g_hash));
        assert_eq!(main_snap.get("f"), Some(&f_hash));
        // merge node has two parents.
        let merged_node = load_node(&cb, &branch_head(&cb, "main").unwrap()).unwrap();
        assert_eq!(merged_node.parents.len(), 2, "merge node has two parents");
    }

    #[test]
    fn merge_reports_conflict_not_silently_resolved() {
        let dir = TempDir::new("merge_conflict");
        let mut cb = build(dir.path(), "def f(x: Int) -> Int = x");

        // Fork b. Change f to hash A on main, to hash B on b.
        branch(&cb, "b", None).unwrap();

        // On main: f -> A.
        let a = Hash([0xAA; 32]);
        cb.set_name("f", a).unwrap();

        // Switch to b: f -> B.
        switch(&mut cb, "b").unwrap();
        let bn = Hash([0xBB; 32]);
        cb.set_name("f", bn).unwrap();

        // Merge b into main: same name, two different hashes -> conflict.
        let before_head = branch_head(&cb, "main").unwrap();
        let res = merge(&mut cb, "b", "main").expect("merge runs");
        match res {
            MergeResult::Conflicts(cs) => {
                assert_eq!(cs.len(), 1);
                let c = &cs[0];
                assert_eq!(c.name, "f");
                assert_eq!(c.into_hash, a);
                assert_eq!(c.from_hash, bn);
            }
            other => panic!("expected a conflict, got {:?}", other),
        }
        // main is untouched on conflict (nothing silently committed).
        assert_eq!(branch_head(&cb, "main").unwrap(), before_head);
    }
}
