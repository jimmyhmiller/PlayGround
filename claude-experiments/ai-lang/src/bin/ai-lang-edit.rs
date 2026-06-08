//! `ai-lang-edit` — the JSON/stdio structural-edit server (Phase 6).
//!
//! A long-lived process speaking newline-delimited JSON (JSONL): one request
//! object per line on stdin, one response object per line on stdout. It keeps
//! the codebase + dependency index warm in memory across an agent's whole
//! editing session and exposes the entire edit algebra (`edit.rs`, `slice.rs`,
//! `namespace.rs`, `printer.rs`) as structured operations.
//!
//! This is the SAME engine as the `ai-lang` CLI; this binary is the persistent,
//! machine-facing adapter described in `docs/structural-editing.md` section 9.
//!
//! ## Protocol (one request / one response per line)
//!
//! Request:  `{ "id": <int>, "op": "<name>", "branch": "<optional>",
//!             "params": { ... } }`
//! Response: `{ "id": <int>, "ok": true, "result": { ... } }`
//!        or `{ "id": <int>, "ok": false,
//!             "error": { "kind": "...", "message": "..." } }`
//!
//! Responses echo the request `id`. Ops are applied serially over one in-memory
//! `Codebase`, so ordering is well-defined.
//!
//! ## Staleness preconditions (borrowed from zerolang, doc 13.3)
//!
//! A write request MAY include `"expect_hash"` (the target's expected current
//! hash) and/or `"expect_root"` (the expected current branch causal/namespace
//! root hash). Before applying, the server checks them; if the world moved it
//! responds `ok:false` with `error.kind == "Conflict"` and includes the actual
//! current `hash`/`root` so the agent can refetch. Nothing is applied on a
//! conflict. This is optimistic concurrency for parallel agents.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use ai_lang::ast::Type;
use ai_lang::codebase::Codebase;
use ai_lang::depindex::DependencyIndex;
use ai_lang::edit::{self, Direction, EditError, EditResult, ExtractSelector};
use ai_lang::evalrun;
use ai_lang::hash::Hash;
use ai_lang::jsonl::{self, Json};
use ai_lang::namespace::{self, MergeResult};
use ai_lang::printer::print_def;
use ai_lang::slice;
use ai_lang::todostore::TodoStore;
use ai_lang::typecheck::{Todo, TypeScheme};

// =============================================================================
// Server state — the warm, in-memory codebase
// =============================================================================

/// Everything the server keeps loaded between requests. The `Codebase` already
/// holds the in-memory name map + type cache; we additionally keep the
/// dependency index warm so reads (`deps`/`usages`) and the update cone are
/// fast across the whole session.
pub struct ServerState {
    cb: Codebase,
    index: DependencyIndex,
}

impl ServerState {
    /// Open the codebase rooted at `path` and load (or rebuild) its dependency
    /// index. A failure here is fatal at startup, not a per-request error.
    pub fn open(path: &PathBuf) -> Result<Self, String> {
        let cb = Codebase::open(path).map_err(|e| format!("open codebase: {}", e))?;
        let index = edit::load_or_rebuild_index(&cb)
            .map_err(|e| format!("load dependency index: {}", e))?;
        Ok(ServerState { cb, index })
    }

    /// Rebuild + persist the dependency index from the store after a write that
    /// added new defs. The persisted index is a pure cache; a rebuild is always
    /// correct. Called after mutating ops so subsequent reads see new defs.
    fn refresh_index(&mut self) {
        if let Ok(idx) = DependencyIndex::rebuild_from_codebase(&self.cb) {
            let _ = idx.save(self.cb.root());
            self.index = idx;
        }
    }
}

/// After any successful mutating op, reconcile the persisted todo log for the
/// active branch with the now-current codebase: record any fresh todos the op
/// produced, prune the ones that are now resolved (the fixed-def hash is
/// orphaned), and save. Called uniformly from every write site so the on-disk
/// worklist is always consistent with what was written.
///
/// `branch` is the branch the op was applied on (the active branch, after any
/// `branch`-field routing). A failure here is logged to stderr but never fails
/// the op itself — the edit already committed; the todo log is a cache.
fn refresh_todos(state: &ServerState, branch: &str, new_todos: &[Todo]) {
    let root = state.cb.root();
    let mut store = match TodoStore::load(root, branch) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[ai-lang-edit] todo store load failed for branch {}: {}", branch, e);
            return;
        }
    };
    store.record(new_todos);
    store.clear_resolved(&state.cb);
    if let Err(e) = store.save() {
        eprintln!("[ai-lang-edit] todo store save failed for branch {}: {}", branch, e);
    }
}

/// The active branch name, or "main" if the namespace can't be read (the todo
/// store still needs *some* per-branch key; "main" is the default branch).
fn active_branch(state: &ServerState) -> String {
    namespace::current_branch(&state.cb).unwrap_or_else(|_| "main".to_string())
}

// =============================================================================
// Structured op errors (every failure is one of these — never a silent no-op)
// =============================================================================

/// A structured server error. `kind` is a stable, machine-branchable code; the
/// optional `extra` fields carry conflict details (actual current hash / root).
struct OpError {
    kind: &'static str,
    message: String,
    /// Extra fields merged into the error object (used by `Conflict`).
    extra: Vec<(String, Json)>,
}

impl OpError {
    fn new(kind: &'static str, message: impl Into<String>) -> Self {
        OpError {
            kind,
            message: message.into(),
            extra: Vec::new(),
        }
    }

    fn bad_params(message: impl Into<String>) -> Self {
        OpError::new("BadParams", message)
    }

    fn unknown_op(op: &str) -> Self {
        OpError::new("UnknownOp", format!("unknown op: {}", op))
    }

    fn with(mut self, key: &str, value: Json) -> Self {
        self.extra.push((key.to_owned(), value));
        self
    }

    /// Map an [`EditError`] to a structured op error, preserving a stable kind
    /// per variant so an agent can branch on `kind` without parsing prose.
    fn from_edit(e: EditError) -> Self {
        let kind = match &e {
            EditError::NameNotFound(_) => "NameNotFound",
            EditError::NameExists(_) => "NameExists",
            EditError::HashNotFound(_) => "HashNotFound",
            EditError::AmbiguousHashPrefix { .. } => "AmbiguousHashPrefix",
            EditError::BadTarget(_) => "BadTarget",
            EditError::ParseError(_) => "ParseError",
            EditError::NotSingleDef { .. } => "NotSingleDef",
            EditError::NameMismatch { .. } => "NameMismatch",
            EditError::ResolveError(_) => "ResolveError",
            EditError::TypeError(_) => "TypeError",
            EditError::Codebase(_) => "Codebase",
            EditError::Unsupported(_) => "Unsupported",
        };
        OpError::new(kind, e.to_string())
    }

    fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("kind".to_string(), Json::Str(self.kind.to_owned())),
            ("message".to_string(), Json::Str(self.message.clone())),
        ];
        pairs.extend(self.extra.iter().cloned());
        Json::obj(pairs)
    }
}

impl From<EditError> for OpError {
    fn from(e: EditError) -> Self {
        OpError::from_edit(e)
    }
}

type OpResult = Result<Json, OpError>;

// =============================================================================
// JSON helpers for reading params / building results
// =============================================================================

fn require_str(params: &Json, key: &str) -> Result<String, OpError> {
    params
        .get(key)
        .and_then(|j| j.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| OpError::bad_params(format!("missing or non-string param `{}`", key)))
}

fn opt_str(params: &Json, key: &str) -> Option<String> {
    params.get(key).and_then(|j| j.as_str()).map(|s| s.to_owned())
}

fn opt_bool(params: &Json, key: &str) -> bool {
    params.get(key).and_then(|j| j.as_bool()).unwrap_or(false)
}

fn hash_json(h: &Hash) -> Json {
    Json::Str(h.to_hex())
}

fn opt_name_json(name: &Option<String>) -> Json {
    match name {
        Some(n) => Json::Str(n.clone()),
        None => Json::Null,
    }
}

// =============================================================================
// Rendering engine values to JSON
// =============================================================================

/// Render an `ast::Type` to a compact surface-like string for `type`/`type_of`.
fn render_type(t: &Type) -> String {
    match t {
        Type::Builtin(name) => name.clone(),
        Type::TypeRef(h) => format!("#{}", &h.to_hex()[..8]),
        Type::TypeVar(i) => format!("T{}", i),
        Type::SelfRef(i) => format!("Self{}", i),
        Type::Apply(head, args) => {
            let inner: Vec<String> = args.iter().map(render_type).collect();
            format!("{}<{}>", render_type(head), inner.join(", "))
        }
        Type::FnType { params, ret } => {
            let ps: Vec<String> = params.iter().map(render_type).collect();
            format!("({}) -> {}", ps.join(", "), render_type(ret))
        }
    }
}

/// Render a cached [`TypeScheme`] to a compact string.
fn render_scheme(s: &TypeScheme) -> String {
    match s {
        TypeScheme::Fn { params, ret, .. } => {
            let ps: Vec<String> = params.iter().map(render_type).collect();
            format!("({}) -> {}", ps.join(", "), render_type(ret))
        }
        TypeScheme::Struct { fields, .. } => {
            let fs: Vec<String> = fields
                .iter()
                .map(|(n, t)| format!("{}: {}", n, render_type(t)))
                .collect();
            format!("struct {{ {} }}", fs.join(", "))
        }
        TypeScheme::Enum { variants, .. } => {
            let vs: Vec<String> = variants
                .iter()
                .map(|(n, p)| match p {
                    Some(t) => format!("{}({})", n, render_type(t)),
                    None => n.clone(),
                })
                .collect();
            format!("enum {{ {} }}", vs.join(" | "))
        }
        TypeScheme::State { ty } => format!("state {}", render_type(ty)),
    }
}

fn defref_json(d: &edit::DefRef) -> Json {
    Json::obj([
        ("name".to_string(), opt_name_json(&d.name)),
        ("hash".to_string(), hash_json(&d.hash)),
    ])
}

fn defrefs_json(items: &[edit::DefRef]) -> Json {
    Json::Array(items.iter().map(defref_json).collect())
}

fn change_json(c: &edit::Change) -> Json {
    Json::obj([
        ("name".to_string(), opt_name_json(&c.name)),
        ("old".to_string(), hash_json(&c.old)),
        ("new".to_string(), hash_json(&c.new)),
    ])
}

/// Serialize one typecheck `Todo`. The element type is named through the public
/// `EditResult.todos` field type (`ai_lang::typecheck::Todo`) rather than a
/// direct import, so we depend only on the field's public fields.
fn todo_json(t: &ai_lang::typecheck::Todo) -> Json {
    Json::obj([
        ("hash".to_string(), hash_json(&t.hash)),
        ("name".to_string(), opt_name_json(&t.name)),
        ("message".to_string(), Json::Str(t.message.clone())),
    ])
}

/// Serialize the full [`EditResult`] changelog to JSON (the write-op result).
fn edit_result_json(r: &EditResult) -> Json {
    let renamed: Vec<Json> = r
        .renamed
        .iter()
        .map(|(from, to, h)| {
            Json::obj([
                ("from".to_string(), Json::Str(from.clone())),
                ("to".to_string(), Json::Str(to.clone())),
                ("hash".to_string(), hash_json(h)),
            ])
        })
        .collect();
    Json::obj([
        ("renamed".to_string(), Json::Array(renamed)),
        (
            "updated".to_string(),
            Json::Array(r.updated.iter().map(change_json).collect()),
        ),
        (
            "propagated".to_string(),
            Json::Array(r.propagated.iter().map(change_json).collect()),
        ),
        (
            "todos".to_string(),
            Json::Array(r.todos.iter().map(todo_json).collect()),
        ),
        ("no_op".to_string(), Json::Bool(r.no_op)),
    ])
}

// =============================================================================
// Branch selection + staleness preconditions
// =============================================================================

/// If the request carries a `branch` field different from the active branch,
/// switch to it (or error if it doesn't exist). Returns the active branch name
/// after any switch. A `branch` naming a non-existent branch is a hard error,
/// never silently ignored.
fn apply_branch_field(state: &mut ServerState, req: &Json) -> Result<String, OpError> {
    let current = namespace::current_branch(&state.cb)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    let requested = match req.get("branch").and_then(|j| j.as_str()) {
        Some(b) => b.to_owned(),
        None => return Ok(current),
    };
    if requested == current {
        return Ok(current);
    }
    // Verify the branch exists, then switch.
    let all = namespace::branches(&state.cb)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    if !all.contains(&requested) {
        return Err(OpError::new(
            "NoSuchBranch",
            format!("requested branch does not exist: {}", requested),
        ));
    }
    namespace::switch(&mut state.cb, &requested)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    state.refresh_index();
    Ok(requested)
}

/// The current causal/namespace-root hash of the active branch (used for
/// `expect_root`). Reachable directly from `namespace::branch_head`.
fn current_root(state: &ServerState) -> Result<Hash, OpError> {
    let branch = namespace::current_branch(&state.cb)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    namespace::branch_head(&state.cb, &branch)
        .map_err(|e| OpError::new("Namespace", e.to_string()))
}

/// Check the optional staleness preconditions on a write request, BEFORE any
/// mutation is attempted. `target` (if given) is the name whose `expect_hash`
/// is checked against its current `get_name` hash. On a mismatch we return a
/// `Conflict` carrying the ACTUAL current hash/root so the agent can refetch,
/// and the caller must NOT apply the write.
fn check_preconditions(
    state: &ServerState,
    req: &Json,
    target: Option<&str>,
) -> Result<(), OpError> {
    let params = req.get("params").cloned().unwrap_or(Json::Null);

    // expect_hash: compare against the current hash of `target`.
    if let Some(expect) = opt_str(&params, "expect_hash") {
        let name = target.ok_or_else(|| {
            OpError::bad_params("expect_hash given but this op has no single hash target")
        })?;
        let actual = state.cb.get_name(name);
        let actual_hex = actual.map(|h| h.to_hex());
        let matches = actual_hex.as_deref() == Some(expect.as_str());
        if !matches {
            let actual_json = match &actual_hex {
                Some(h) => Json::Str(h.clone()),
                None => Json::Null,
            };
            return Err(OpError::new(
                "Conflict",
                format!(
                    "expect_hash {} does not match current hash {} for `{}`",
                    expect,
                    actual_hex.as_deref().unwrap_or("<none>"),
                    name
                ),
            )
            .with("expected_hash", Json::Str(expect))
            .with("actual_hash", actual_json));
        }
    }

    // expect_root: compare against the active branch's causal/namespace root.
    if let Some(expect_root) = opt_str(&params, "expect_root") {
        let actual = current_root(state)?;
        if actual.to_hex() != expect_root {
            return Err(OpError::new(
                "Conflict",
                format!(
                    "expect_root {} does not match current branch root {}",
                    expect_root,
                    actual.to_hex()
                ),
            )
            .with("expected_root", Json::Str(expect_root))
            .with("actual_root", hash_json(&actual)));
        }
    }

    Ok(())
}

// =============================================================================
// The dispatcher — factored as a testable pure-ish function over ServerState
// =============================================================================

/// Handle one request and produce one response. This is the single, testable
/// core: it takes the warm `ServerState` and a parsed request `Json`, applies
/// the op against the real engine, and returns the response `Json` (echoing the
/// request id). No stdio here, so tests drive it directly.
pub fn handle_request(state: &mut ServerState, req: &Json) -> Json {
    // The id is echoed verbatim on every response. A missing/garbage id is
    // itself reported (id = null) rather than silently dropped.
    let id = req.get("id").cloned().unwrap_or(Json::Null);

    let result = dispatch(state, req);
    match result {
        Ok(value) => Json::obj([
            ("id".to_string(), id),
            ("ok".to_string(), Json::Bool(true)),
            ("result".to_string(), value),
        ]),
        Err(err) => Json::obj([
            ("id".to_string(), id),
            ("ok".to_string(), Json::Bool(false)),
            ("error".to_string(), err.to_json()),
        ]),
    }
}

/// The complete op catalogue — the single source of truth for both the `ops`
/// discovery response and the `canonical_op` resolver. Each entry is
/// `(canonical_name, kind, params, summary)`. `params` lists each parameter as
/// `name` (required) or `name?` (optional).
const OP_TABLE: &[(&str, &str, &[&str], &str)] = &[
    // ---- discovery ----
    ("ops", "read", &[], "List every operation with its params and a one-line summary."),
    // ---- reads ----
    ("view", "read", &["target"], "Render a definition as surface source (canonical AST projected back to readable code, with the author's local names)."),
    ("type_of", "read", &["target"], "The inferred type of a definition."),
    ("deps", "read", &["target", "reverse?", "transitive?"], "Dependencies (or, with reverse, dependents) of a definition."),
    ("find_usages", "read", &["target"], "Every definition that references the target (exact, via the reverse index)."),
    ("find_by_type", "read", &["signature"], "Find definitions whose type matches a signature, modulo type-variable renaming."),
    ("ls", "read", &[], "List every name on the current branch with its hash."),
    ("diff", "read", &["a", "b"], "Structural namespace diff between two branches (added/removed/changed)."),
    ("todo", "read", &["branch?"], "Outstanding todos (broken dependents) for the active or given branch; persisted, auto-clears when fixed."),
    ("history", "read", &["branch?"], "The causal commit history of a branch."),
    ("branches", "read", &[], "List branches and the current one."),
    ("eval", "read", &["target", "args?"], "JIT-run a definition and return its real value as JSON. Returns any type; args may be any type (Int/Float/Bool/String/Array<T>/struct/enum, nested freely) except closures."),
    // ---- writes ----
    ("add", "write", &["source"], "Add new definition(s) from surface source."),
    ("update", "write", &["name", "source", "dry_run?", "propagate?", "expect_hash?", "expect_root?"], "Replace a definition's body. Dependents are NOT touched by default. With `propagate: true`, rewrites the cone (same-type only)."),
    ("rename", "write", &["from", "to", "expect_hash?"], "Rename a definition (namespace-only, O(1), never breaks callers)."),
    ("move", "write", &["from", "to"], "Move a definition to a new (possibly dotted) name."),
    ("inline", "write", &["name", "dry_run?"], "Inline a definition into its callers (capture-correct beta reduction)."),
    ("reorder_params", "write", &["name", "perm", "dry_run?"], "Permute a function's parameters and rewrite every call site to match."),
    ("extract", "write", &["name", "selector", "new_name"], "Lift a sub-expression into a new named definition and call it."),
    ("delete", "write", &["name"], "Remove a name (the definition stays on disk; history is immutable)."),
    ("import", "write", &["path"], "Import a definition bundle (idempotent; shared hashes dedupe)."),
    ("export", "write", &["names", "out"], "Export names plus their transitive hash-closure to a bundle file."),
    ("propagate", "write", &["name", "dry_run?"], "Verify all transitive dependents typecheck against the current hash (same-type safety check)."),
    ("todos", "read", &[], "List the pending worklist of dependents that still reference an old hash after type changes."),
    // ---- branch / VCS ----
    ("branch", "vcs", &["name", "from?"], "Create a branch (O(1), shares all definitions)."),
    ("switch", "vcs", &["name"], "Switch the active branch."),
    ("merge", "vcs", &["from", "into"], "Merge one branch into another; conflicts are reported, never silently resolved."),
    ("undo", "vcs", &[], "Move the current branch head back to its parent."),
];

/// Normalize an op name to its canonical form. Case-insensitive; `-` and `_`
/// are interchangeable; a small alias table covers the CLI spellings and the
/// most natural agent guesses. Returns the canonical `&'static str` (so the
/// dispatch match can stay on string literals) or `None` if unrecognized.
fn canonical_op(raw: &str) -> Option<&'static str> {
    let norm = raw.trim().to_ascii_lowercase().replace('-', "_");
    // Friendly aliases -> canonical.
    let norm = match norm.as_str() {
        "usages" | "uses" | "callers" | "references" | "refs" | "who_calls" => "find_usages",
        "search" | "by_type" | "search_by_type" => "find_by_type",
        "reorder" | "permute_params" => "reorder_params",
        "type" | "typeof" => "type_of",
        "propagate_cone" | "rewire" => "propagate",
        "worklist" | "pending" => "todos",
        "list" | "names" => "ls",
        "remove" | "rm" => "delete",
        "run" => "eval",
        "dependents" => "deps",
        "help" | "commands" | "operations" => "ops",
        other => other,
    };
    OP_TABLE
        .iter()
        .find(|(name, ..)| *name == norm)
        .map(|(name, ..)| *name)
}

/// Up to a few canonical op names whose text is closest to `raw`, to enrich an
/// unknown-op error. Cheap substring/prefix heuristic — enough to be helpful.
fn suggestions(raw: &str) -> Json {
    let norm = raw.trim().to_ascii_lowercase().replace('-', "_");
    let mut hits: Vec<&str> = OP_TABLE
        .iter()
        .map(|(n, ..)| *n)
        .filter(|n| n.contains(&norm) || norm.contains(*n) || shares_prefix(n, &norm))
        .collect();
    if hits.is_empty() {
        // Fall back to listing the read ops as a gentle nudge toward `ops`.
        hits.push("ops");
    }
    hits.truncate(5);
    Json::Array(hits.into_iter().map(|s| Json::Str(s.to_string())).collect())
}

fn shares_prefix(a: &str, b: &str) -> bool {
    let n = a.len().min(b.len()).min(3);
    n >= 3 && a.as_bytes()[..n] == b.as_bytes()[..n]
}

/// The `ops` discovery response: the full catalogue so an agent can learn the
/// vocabulary instead of guessing.
fn op_ops() -> Json {
    let list: Vec<Json> = OP_TABLE
        .iter()
        .map(|(name, kind, params, summary)| {
            Json::obj([
                ("op".to_string(), Json::Str(name.to_string())),
                ("kind".to_string(), Json::Str(kind.to_string())),
                (
                    "params".to_string(),
                    Json::Array(params.iter().map(|p| Json::Str(p.to_string())).collect()),
                ),
                ("summary".to_string(), Json::Str(summary.to_string())),
            ])
        })
        .collect();
    Json::obj([
        ("count".to_string(), Json::Int(OP_TABLE.len() as i64)),
        ("ops".to_string(), Json::Array(list)),
        (
            "note".to_string(),
            Json::Str(
                "Param names ending in `?` are optional. Op names are case-insensitive and \
                 `-`/`_` interchangeable; common aliases (usages, run, search, ...) are accepted."
                    .to_string(),
            ),
        ),
    ])
}

fn dispatch(state: &mut ServerState, req: &Json) -> OpResult {
    let raw_op = req
        .get("op")
        .and_then(|j| j.as_str())
        .ok_or_else(|| OpError::bad_params("missing or non-string `op`"))?
        .to_owned();
    let params = req.get("params").cloned().unwrap_or(Json::Null);

    // Resolve the op name to its canonical form: case-insensitive, with `-`
    // and `_` treated identically, plus a table of friendly aliases (so the
    // CLI's `usages`/`find-by-type`/`reorder-params` spellings and other
    // natural guesses all work against the server). An agent should never fail
    // just because it guessed `usages` instead of `find_usages`.
    let op = canonical_op(&raw_op)
        .ok_or_else(|| OpError::unknown_op(&raw_op).with("did_you_mean", suggestions(&raw_op)))?;

    // `ops` is the self-describing discovery op — it needs no branch context.
    if op == "ops" {
        return Ok(op_ops());
    }

    // Apply the optional `branch` selector for every op (reads + writes).
    apply_branch_field(state, req)?;

    match op {
        // ---- Reads ----
        "view" => op_view(state, &params),
        "type_of" => op_type_of(state, &params),
        "deps" => op_deps(state, &params),
        "find_usages" => op_find_usages(state, &params),
        "find_by_type" => op_find_by_type(state, &params),
        "ls" => op_ls(state),
        "diff" => op_diff(state, &params),
        "todo" => op_todo(state, &params),
        "history" => op_history(state, &params),
        "branches" => op_branches(state),
        "eval" => op_eval(state, &params),

        // ---- Writes ----
        "add" => op_add(state, &params),
        "update" => op_update(state, req, &params),
        "propagate" => op_propagate(state, &params),
        "todos" => op_todos(state, &params),
        "rename" => op_rename(state, req, &params),
        "move" => op_move(state, req, &params),
        "inline" => op_inline(state, req, &params),
        "reorder_params" => op_reorder_params(state, req, &params),
        "extract" => op_extract(state, req, &params),
        "delete" => op_delete(state, req, &params),
        "import" => op_import(state, &params),
        "export" => op_export(state, &params),

        // ---- Branch / VCS ----
        "branch" => op_branch(state, &params),
        "switch" => op_switch(state, &params),
        "merge" => op_merge(state, &params),
        "undo" => op_undo(state),

        // `canonical_op` only returns names in OP_TABLE, so this is unreachable
        // in practice; keep it as a typed error rather than a panic.
        other => Err(OpError::unknown_op(other)),
    }
}

// -----------------------------------------------------------------------------
// Reads
// -----------------------------------------------------------------------------

fn op_view(state: &ServerState, params: &Json) -> OpResult {
    let target = require_str(params, "target")?;
    let hash = edit::resolve_target(&state.cb, &target)?;
    let source = print_def(&state.cb, hash)
        .map_err(|e| OpError::new("Printer", e.to_string()))?;
    let ty = state.cb.types().get(&hash).map(render_scheme);
    Ok(Json::obj([
        ("hash".to_string(), hash_json(&hash)),
        ("source".to_string(), Json::Str(source)),
        (
            "type".to_string(),
            match ty {
                Some(t) => Json::Str(t),
                None => Json::Null,
            },
        ),
    ]))
}

fn op_type_of(state: &ServerState, params: &Json) -> OpResult {
    let target = require_str(params, "target")?;
    let hash = edit::resolve_target(&state.cb, &target)?;
    match state.cb.types().get(&hash) {
        Some(scheme) => Ok(Json::obj([
            ("hash".to_string(), hash_json(&hash)),
            ("type".to_string(), Json::Str(render_scheme(scheme))),
        ])),
        None => Err(OpError::new(
            "NoType",
            format!("no cached type scheme for `{}` ({})", target, hash.to_hex()),
        )),
    }
}

/// `eval` — JIT-run a def and return its real value of any type as structured
/// JSON. The return side is fully general (Int / Float / Bool / String /
/// struct / enum / Array / list-shaped enum). The arg side is now general too:
/// `{ "args": [ ... ] }` accepts Int / Float / Bool, struct objects
/// (`{"field": value, ...}`), enum values (`{"Variant": payload}` or bare
/// `"Variant"`), and cons-cell lists (they are enums). String and Array args
/// remain `Unsupported` (no wire value kind to build them from). Arities 0..=12.
///
/// A fresh `inkwell::Context` + `Jit` is built per call (LLVM contexts aren't
/// reusable), so this is a deliberate "run it" action, not a hot loop. The
/// runtime / knowledge-base / at-binding are installed and torn down inside
/// `evalrun::eval`, so repeated evals in this long-lived server don't corrupt
/// each other's state.
fn op_eval(state: &ServerState, params: &Json) -> OpResult {
    let target = require_str(params, "target")?;
    let hash = edit::resolve_target(&state.cb, &target)?;

    // The signature drives both the arg-side gate and the return rendering.
    let (sig_params, sig_ret) = match state.cb.types().get(&hash) {
        Some(TypeScheme::Fn { params, ret, .. }) => (params.clone(), ret.clone()),
        Some(_) => {
            return Err(OpError::new(
                "NotCallable",
                format!("`{}` is a struct/enum, not a function", target),
            ));
        }
        None => {
            return Err(OpError::new(
                "NoType",
                format!("no cached type scheme for `{}` ({})", target, hash.to_hex()),
            ));
        }
    };

    // Parse `args` (default empty). Args are now general JSON values; each is
    // built into a real heap object of its declared parameter type inside
    // `evalrun::eval` (Int/Float/Bool/struct/enum/list). String and Array args
    // are refused there with a structured `Unsupported` error.
    let args: Vec<Json> = match params.get("args") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::Array(items)) => items.clone(),
        Some(_) => return Err(OpError::bad_params("`args` must be an array")),
    };

    match evalrun::eval(&state.cb, hash, &sig_params, &sig_ret, &args) {
        Ok(ev) => Ok(Json::obj([
            ("target".to_string(), Json::Str(target)),
            ("hash".to_string(), hash_json(&hash)),
            ("type".to_string(), Json::Str(ev.type_str)),
            ("value".to_string(), ev.value),
        ])),
        Err(e) => Err(OpError::new(e.kind(), e.message().to_owned())),
    }
}

fn op_deps(state: &ServerState, params: &Json) -> OpResult {
    let target = require_str(params, "target")?;
    let reverse = opt_bool(params, "reverse");
    let transitive = opt_bool(params, "transitive");
    let direction = if reverse {
        Direction::Reverse
    } else {
        Direction::Forward
    };
    let result = edit::deps(&state.cb, &state.index, &target, direction, transitive)?;
    Ok(Json::obj([
        ("target".to_string(), Json::Str(target)),
        (
            "direction".to_string(),
            Json::Str(if reverse { "reverse" } else { "forward" }.to_owned()),
        ),
        ("transitive".to_string(), Json::Bool(transitive)),
        ("deps".to_string(), defrefs_json(&result)),
    ]))
}

fn op_find_usages(state: &ServerState, params: &Json) -> OpResult {
    let target = require_str(params, "target")?;
    let users = edit::find_usages(&state.cb, &state.index, &target)?;
    Ok(Json::obj([
        ("target".to_string(), Json::Str(target)),
        ("usages".to_string(), defrefs_json(&users)),
    ]))
}

fn op_find_by_type(state: &ServerState, params: &Json) -> OpResult {
    let sig = require_str(params, "signature")?;
    let query = parse_signature(&sig).map_err(|m| OpError::new("BadSignature", m))?;
    let hits = slice::find_by_type(&state.cb, &query);
    let arr: Vec<Json> = hits
        .iter()
        .map(|(name, hash)| {
            Json::obj([
                ("name".to_string(), opt_name_json(name)),
                ("hash".to_string(), hash_json(hash)),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("signature".to_string(), Json::Str(sig)),
        ("matches".to_string(), Json::Array(arr)),
    ]))
}

fn op_ls(state: &ServerState) -> OpResult {
    let mut entries: Vec<(&String, &Hash)> = state.cb.names().iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let arr: Vec<Json> = entries
        .iter()
        .map(|(name, hash)| {
            Json::obj([
                ("name".to_string(), Json::Str((*name).clone())),
                ("hash".to_string(), hash_json(hash)),
            ])
        })
        .collect();
    Ok(Json::obj([("names".to_string(), Json::Array(arr))]))
}

fn op_diff(state: &ServerState, params: &Json) -> OpResult {
    let a = require_str(params, "a")?;
    let b = require_str(params, "b")?;
    let d = namespace::diff(&state.cb, &a, &b)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    let added: Vec<Json> = d
        .added
        .iter()
        .map(|(n, h)| {
            Json::obj([
                ("name".to_string(), Json::Str(n.clone())),
                ("hash".to_string(), hash_json(h)),
            ])
        })
        .collect();
    let removed: Vec<Json> = d
        .removed
        .iter()
        .map(|(n, h)| {
            Json::obj([
                ("name".to_string(), Json::Str(n.clone())),
                ("hash".to_string(), hash_json(h)),
            ])
        })
        .collect();
    let changed: Vec<Json> = d
        .changed
        .iter()
        .map(|(n, ah, bh)| {
            Json::obj([
                ("name".to_string(), Json::Str(n.clone())),
                ("a".to_string(), hash_json(ah)),
                ("b".to_string(), hash_json(bh)),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("a".to_string(), Json::Str(a)),
        ("b".to_string(), Json::Str(b)),
        ("added".to_string(), Json::Array(added)),
        ("removed".to_string(), Json::Array(removed)),
        ("changed".to_string(), Json::Array(changed)),
    ]))
}

/// `todo` — the persisted, outstanding worklist for a branch. Every type-
/// changing write records its breakages here and prunes the ones it resolves,
/// so an agent that reconnects can ask "what's still broken?" and get a real
/// answer (the whole point of the persisted log). The top-level `branch` field
/// is already applied by the dispatcher; an optional `params.branch` lets a
/// caller peek at another branch's worklist without switching to it.
fn op_todo(state: &ServerState, params: &Json) -> OpResult {
    let branch = match opt_str(params, "branch") {
        Some(b) => b,
        None => active_branch(state),
    };
    let store = TodoStore::load(state.cb.root(), &branch)
        .map_err(|e| OpError::new("TodoStore", e.to_string()))?;
    let todos: Vec<Json> = store
        .list()
        .iter()
        .map(|e| {
            Json::obj([
                ("hash".to_string(), hash_json(&e.hash)),
                ("name".to_string(), opt_name_json(&e.name)),
                ("message".to_string(), Json::Str(e.message.clone())),
                ("code".to_string(), Json::Str(e.code.clone())),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("branch".to_string(), Json::Str(branch)),
        ("todos".to_string(), Json::Array(todos)),
    ]))
}

fn op_history(state: &ServerState, params: &Json) -> OpResult {
    let branch = opt_str(params, "branch");
    let chain = namespace::history(&state.cb, branch.as_deref())
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    let arr: Vec<Json> = chain.iter().map(hash_json).collect();
    Ok(Json::obj([
        (
            "branch".to_string(),
            match branch {
                Some(b) => Json::Str(b),
                None => Json::Str(
                    namespace::current_branch(&state.cb)
                        .map_err(|e| OpError::new("Namespace", e.to_string()))?,
                ),
            },
        ),
        ("history".to_string(), Json::Array(arr)),
    ]))
}

fn op_branches(state: &ServerState) -> OpResult {
    let all = namespace::branches(&state.cb)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    let current = namespace::current_branch(&state.cb)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    Ok(Json::obj([
        (
            "branches".to_string(),
            Json::Array(all.into_iter().map(Json::Str).collect()),
        ),
        ("current".to_string(), Json::Str(current)),
    ]))
}

// -----------------------------------------------------------------------------
// Writes
// -----------------------------------------------------------------------------

/// `add {source}` — parse + resolve + typecheck the source and store every def,
/// pointing each name at its hash. Mirrors the CLI `add` engine path but uses
/// the warm codebase (no stdlib concat — the agent adds against the existing
/// namespace; reference resolution is by the codebase's `ExternalEnv`).
fn op_add(state: &mut ServerState, params: &Json) -> OpResult {
    let source = require_str(params, "source")?;
    // Resolve against the codebase namespace so the new defs may reference
    // already-stored top-level defs by name (same env `update` uses).
    let added = edit::add(&mut state.cb, &source)?;
    state.refresh_index();
    // `add` produces no todos, but names changed — reconcile the worklist so a
    // re-added (now-fixed) def clears any stale todo keyed to its old name.
    let branch = active_branch(state);
    refresh_todos(state, &branch, &[]);
    let arr: Vec<Json> = added
        .iter()
        .map(|d| {
            let ty = state.cb.types().get(&d.hash).map(render_scheme);
            Json::obj([
                ("name".to_string(), opt_name_json(&d.name)),
                ("hash".to_string(), hash_json(&d.hash)),
                (
                    "type".to_string(),
                    match ty {
                        Some(t) => Json::Str(t),
                        None => Json::Null,
                    },
                ),
            ])
        })
        .collect();
    Ok(Json::obj([("added".to_string(), Json::Array(arr))]))
}

fn op_update(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    let source = require_str(params, "source")?;
    let dry_run = opt_bool(params, "dry_run");
    let propagate = opt_bool(params, "propagate");
    // Staleness: expect_hash is checked against `name`'s current hash.
    check_preconditions(state, req, Some(&name))?;
    let result = match (dry_run, propagate) {
        (false, false) => {
            let r = edit::update(&mut state.cb, &name, &source)?;
            state.refresh_index();
            let branch = active_branch(state);
            refresh_todos(state, &branch, &r.todos);
            r
        }
        (true, false) => edit::update_dry_run(&mut state.cb, &name, &source)?,
        (false, true) => {
            let r = edit::update_propagate(&mut state.cb, &name, &source)?;
            state.refresh_index();
            r
        }
        (true, true) => edit::update_dry_run_propagate(&mut state.cb, &name, &source)?,
    };
    Ok(edit_result_json(&result))
}

fn op_propagate(state: &mut ServerState, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    let dry_run = opt_bool(params, "dry_run");
    let result = if dry_run {
        edit::propagate_dry_run(&mut state.cb, &name)?
    } else {
        edit::propagate(&mut state.cb, &name)?
    };
    Ok(edit_result_json(&result))
}

fn op_todos(state: &mut ServerState, params: &Json) -> OpResult {
    let branch = active_branch(state);
    let _ = params;
    let store = ai_lang::todostore::TodoStore::load(state.cb.root(), &branch)
        .map_err(|e| OpError::new("Codebase", format!("todo store: {}", e)))?;
    let todos = store.list();
    let items: Vec<Json> = todos
        .iter()
        .map(|t| {
            Json::obj([
                ("hash".to_string(), hash_json(&t.hash)),
                ("name".to_string(), opt_name_json(&t.name)),
                ("message".to_string(), Json::Str(t.message.clone())),
                ("code".to_string(), Json::Str(t.code.clone())),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("branch".to_string(), Json::Str(branch)),
        ("todos".to_string(), Json::Array(items)),
    ]))
}

fn op_rename(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let from = require_str(params, "from")?;
    let to = require_str(params, "to")?;
    // Staleness: expect_hash is checked against `from`'s current hash.
    check_preconditions(state, req, Some(&from))?;
    let result = edit::rename(&mut state.cb, &from, &to)?;
    let branch = active_branch(state);
    refresh_todos(state, &branch, &result.todos);
    Ok(edit_result_json(&result))
}

fn op_move(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let from = require_str(params, "from")?;
    let to = require_str(params, "to")?;
    check_preconditions(state, req, Some(&from))?;
    let result = edit::move_def(&mut state.cb, &from, &to)?;
    let branch = active_branch(state);
    refresh_todos(state, &branch, &result.todos);
    Ok(edit_result_json(&result))
}

fn op_inline(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    let dry_run = opt_bool(params, "dry_run");
    check_preconditions(state, req, Some(&name))?;
    let result = if dry_run {
        edit::inline_dry_run(&mut state.cb, &name)?
    } else {
        let r = edit::inline(&mut state.cb, &name)?;
        state.refresh_index();
        let branch = active_branch(state);
        refresh_todos(state, &branch, &r.todos);
        r
    };
    Ok(edit_result_json(&result))
}

fn op_reorder_params(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    let dry_run = opt_bool(params, "dry_run");
    let perm_json = params
        .get("perm")
        .and_then(|j| j.as_array())
        .ok_or_else(|| OpError::bad_params("`perm` must be an array of integers"))?;
    let mut perm: Vec<usize> = Vec::with_capacity(perm_json.len());
    for item in perm_json {
        let v = item
            .as_i64()
            .ok_or_else(|| OpError::bad_params("`perm` entries must be integers"))?;
        if v < 0 {
            return Err(OpError::bad_params("`perm` entries must be non-negative"));
        }
        perm.push(v as usize);
    }
    check_preconditions(state, req, Some(&name))?;
    let result = if dry_run {
        edit::reorder_params_dry_run(&mut state.cb, &name, &perm)?
    } else {
        let r = edit::reorder_params(&mut state.cb, &name, &perm)?;
        state.refresh_index();
        let branch = active_branch(state);
        refresh_todos(state, &branch, &r.todos);
        r
    };
    Ok(edit_result_json(&result))
}

fn op_extract(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    let selector_str = require_str(params, "selector")?;
    let new_name = require_str(params, "new_name")?;
    let selector = ExtractSelector::parse(&selector_str)?;
    check_preconditions(state, req, Some(&name))?;
    let result = edit::extract(&mut state.cb, &name, selector, &new_name)?;
    state.refresh_index();
    let branch = active_branch(state);
    refresh_todos(state, &branch, &result.todos);
    Ok(edit_result_json(&result))
}

/// `delete {name}` — remove the name alias (the def stays in `defs/` forever;
/// names are the only mutable layer). Errors clearly if the name is unknown.
fn op_delete(state: &mut ServerState, req: &Json, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    check_preconditions(state, req, Some(&name))?;
    let existed = state
        .cb
        .remove_name(&name)
        .map_err(|e| OpError::new("Codebase", e.to_string()))?;
    if !existed {
        return Err(OpError::new(
            "NameNotFound",
            format!("no such name: {}", name),
        ));
    }
    // Deleting a name may orphan a hash a todo was keyed to — reconcile.
    let branch = active_branch(state);
    refresh_todos(state, &branch, &[]);
    Ok(Json::obj([
        ("deleted".to_string(), Json::Str(name)),
        ("removed".to_string(), Json::Bool(true)),
    ]))
}

fn op_import(state: &mut ServerState, params: &Json) -> OpResult {
    let path = require_str(params, "path")?;
    let report = slice::import(&mut state.cb, std::path::Path::new(&path))
        .map_err(|e| OpError::new("Slice", e.to_string()))?;
    state.refresh_index();
    let branch = active_branch(state);
    refresh_todos(state, &branch, &[]);
    let names: Vec<Json> = report
        .names
        .iter()
        .map(|(n, h)| {
            Json::obj([
                ("name".to_string(), Json::Str(n.clone())),
                ("hash".to_string(), hash_json(h)),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("new_defs".to_string(), Json::Int(report.new_defs as i64)),
        (
            "existing_defs".to_string(),
            Json::Int(report.existing_defs as i64),
        ),
        ("new_types".to_string(), Json::Int(report.new_types as i64)),
        (
            "existing_types".to_string(),
            Json::Int(report.existing_types as i64),
        ),
        ("names".to_string(), Json::Array(names)),
    ]))
}

fn op_export(state: &ServerState, params: &Json) -> OpResult {
    let out = require_str(params, "out")?;
    let names_json = params
        .get("names")
        .and_then(|j| j.as_array())
        .ok_or_else(|| OpError::bad_params("`names` must be an array of strings"))?;
    let mut names: Vec<String> = Vec::with_capacity(names_json.len());
    for item in names_json {
        names.push(
            item.as_str()
                .ok_or_else(|| OpError::bad_params("`names` entries must be strings"))?
                .to_owned(),
        );
    }
    let manifest = slice::export(
        &state.cb,
        &state.index,
        &names,
        std::path::Path::new(&out),
    )
    .map_err(|e| OpError::new("Slice", e.to_string()))?;
    let roots: Vec<Json> = manifest
        .names
        .iter()
        .map(|(n, h)| {
            Json::obj([
                ("name".to_string(), Json::Str(n.clone())),
                ("hash".to_string(), hash_json(h)),
            ])
        })
        .collect();
    Ok(Json::obj([
        ("out".to_string(), Json::Str(out)),
        (
            "defs".to_string(),
            Json::Array(manifest.defs.iter().map(hash_json).collect()),
        ),
        (
            "types".to_string(),
            Json::Array(manifest.types.iter().map(hash_json).collect()),
        ),
        ("roots".to_string(), Json::Array(roots)),
    ]))
}

// -----------------------------------------------------------------------------
// Branch / VCS
// -----------------------------------------------------------------------------

fn op_branch(state: &mut ServerState, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    let from = opt_str(params, "from");
    let head = namespace::branch(&state.cb, &name, from.as_deref())
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    Ok(Json::obj([
        ("branch".to_string(), Json::Str(name)),
        ("head".to_string(), hash_json(&head)),
    ]))
}

fn op_switch(state: &mut ServerState, params: &Json) -> OpResult {
    let name = require_str(params, "name")?;
    namespace::switch(&mut state.cb, &name)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    state.refresh_index();
    // Reconcile the branch we switched TO against its own per-branch worklist.
    refresh_todos(state, &name, &[]);
    let head = current_root(state)?;
    Ok(Json::obj([
        ("switched_to".to_string(), Json::Str(name)),
        ("head".to_string(), hash_json(&head)),
    ]))
}

fn op_merge(state: &mut ServerState, params: &Json) -> OpResult {
    let from = require_str(params, "from")?;
    let into = require_str(params, "into")?;
    let res = namespace::merge(&mut state.cb, &from, &into)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    state.refresh_index();
    // A merge can re-point names on the active branch — reconcile its worklist.
    let branch = active_branch(state);
    refresh_todos(state, &branch, &[]);
    match res {
        MergeResult::Merged { new_head, brought } => {
            let brought_json: Vec<Json> = brought
                .iter()
                .map(|(n, h)| {
                    Json::obj([
                        ("name".to_string(), Json::Str(n.clone())),
                        ("hash".to_string(), hash_json(h)),
                    ])
                })
                .collect();
            Ok(Json::obj([
                ("merged".to_string(), Json::Bool(true)),
                ("new_head".to_string(), hash_json(&new_head)),
                ("brought".to_string(), Json::Array(brought_json)),
            ]))
        }
        MergeResult::Conflicts(conflicts) => {
            let cj: Vec<Json> = conflicts
                .iter()
                .map(|c| {
                    Json::obj([
                        ("name".to_string(), Json::Str(c.name.clone())),
                        (
                            "base".to_string(),
                            match &c.base {
                                Some(h) => hash_json(h),
                                None => Json::Null,
                            },
                        ),
                        ("into".to_string(), hash_json(&c.into_hash)),
                        ("from".to_string(), hash_json(&c.from_hash)),
                    ])
                })
                .collect();
            Ok(Json::obj([
                ("merged".to_string(), Json::Bool(false)),
                ("conflicts".to_string(), Json::Array(cj)),
            ]))
        }
    }
}

fn op_undo(state: &mut ServerState) -> OpResult {
    let new_head = namespace::undo(&mut state.cb)
        .map_err(|e| OpError::new("Namespace", e.to_string()))?;
    state.refresh_index();
    // Undo re-points names to a prior root — reconcile the worklist.
    let branch = active_branch(state);
    refresh_todos(state, &branch, &[]);
    Ok(Json::obj([
        ("undone".to_string(), Json::Bool(true)),
        ("head".to_string(), hash_json(&new_head)),
    ]))
}

// =============================================================================
// Minimal type-signature parser for find_by_type (mirrors the CLI's parser)
// =============================================================================

/// Parse a function signature like `(Int, Int) -> Int` (or generic `(T) -> T`)
/// into a query `TypeScheme::Fn`. A single bare uppercase letter is a type
/// variable; any other identifier is a builtin type name.
fn parse_signature(sig: &str) -> Result<TypeScheme, String> {
    let s = sig.trim();
    let arrow = s
        .find("->")
        .ok_or_else(|| format!("signature must contain `->`: {:?}", sig))?;
    let (lhs, rhs) = s.split_at(arrow);
    let rhs = rhs[2..].trim();
    let lhs = lhs.trim();
    let inner = lhs
        .strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .ok_or_else(|| format!("params must be parenthesized, e.g. (Int): {:?}", lhs))?;

    let mut tvars: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut next_tv: u32 = 0;
    let mut params: Vec<Type> = Vec::new();
    if !inner.trim().is_empty() {
        for atom in inner.split(',') {
            let atom = atom.trim();
            if atom.is_empty() {
                return Err(format!("empty parameter in {:?}", lhs));
            }
            params.push(parse_type_atom(atom, &mut tvars, &mut next_tv));
        }
    }
    let ret = parse_type_atom(rhs, &mut tvars, &mut next_tv);
    Ok(TypeScheme::Fn {
        params,
        ret,
        wire: true,
    })
}

fn parse_type_atom(
    atom: &str,
    tvars: &mut std::collections::HashMap<String, u32>,
    next_tv: &mut u32,
) -> Type {
    let is_tvar = atom.len() == 1 && atom.chars().next().unwrap().is_ascii_uppercase();
    if is_tvar {
        let idx = *tvars.entry(atom.to_owned()).or_insert_with(|| {
            let i = *next_tv;
            *next_tv += 1;
            i
        });
        Type::TypeVar(idx)
    } else {
        Type::Builtin(atom.to_owned())
    }
}

// =============================================================================
// main — the stdio loop (thin wrapper over handle_request)
// =============================================================================

fn cb_root(opt: Option<&String>) -> PathBuf {
    if let Some(p) = opt {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("AI_LANG_CODEBASE") {
        return PathBuf::from(p);
    }
    PathBuf::from(".ai-lang")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Parse `--codebase <path>` (or `--codebase=<path>`); everything else is
    // ignored so the binary stays a clean JSONL pipe.
    let mut codebase: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--codebase" {
            if i + 1 >= args.len() {
                eprintln!("--codebase: missing value");
                std::process::exit(2);
            }
            codebase = Some(args[i + 1].clone());
            i += 2;
        } else if let Some(rest) = a.strip_prefix("--codebase=") {
            codebase = Some(rest.to_owned());
            i += 1;
        } else {
            i += 1;
        }
    }
    let cb_path = cb_root(codebase.as_ref());

    let mut state = match ServerState::open(&cb_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ai-lang-edit: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "[ai-lang-edit] ready (codebase {}); one JSON request per line on stdin",
        cb_path.display()
    );

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[ai-lang-edit] stdin read error: {}", e);
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        // A malformed request line still gets a structured response (id null)
        // rather than silently dropping the line.
        let response = match jsonl::parse(&line) {
            Ok(req) => handle_request(&mut state, &req),
            Err(pe) => Json::obj([
                ("id".to_string(), Json::Null),
                ("ok".to_string(), Json::Bool(false)),
                (
                    "error".to_string(),
                    Json::obj([
                        ("kind".to_string(), Json::Str("BadJson".to_owned())),
                        ("message".to_string(), Json::Str(pe.to_string())),
                    ]),
                ),
            ]),
        };
        if writeln!(out, "{}", response.to_string()).is_err() {
            break;
        }
        let _ = out.flush();
    }
}

// =============================================================================
// Tests — drive handle_request directly over a real ServerState
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
        let dir = std::env::temp_dir().join(format!("ai_lang_edit_{}_{}_{}_{}", tag, pid, nanos, n));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn req(id: i64, op: &str, params: Vec<(&str, Json)>) -> Json {
        Json::obj([
            ("id".to_string(), Json::Int(id)),
            ("op".to_string(), Json::Str(op.to_owned())),
            (
                "params".to_string(),
                Json::obj(params.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<Vec<_>>()),
            ),
        ])
    }

    fn s(v: &str) -> Json {
        Json::Str(v.to_owned())
    }

    /// Assert the response is ok and return its `result` object.
    fn ok_result(resp: &Json) -> Json {
        assert_eq!(
            resp.get("ok").and_then(|j| j.as_bool()),
            Some(true),
            "expected ok response, got {}",
            resp.to_string()
        );
        resp.get("result").cloned().expect("result field")
    }

    fn open_state(tag: &str) -> ServerState {
        let dir = tempdir(tag);
        ServerState::open(&dir).expect("open state")
    }

    #[test]
    fn add_view_and_id_echo() {
        let mut st = open_state("add_view");
        let r = handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def double(x: Int) -> Int = x * 2"))]),
        );
        // id echoed.
        assert_eq!(r.get("id").and_then(|j| j.as_i64()), Some(1));
        let result = ok_result(&r);
        let added = result.get("added").and_then(|j| j.as_array()).unwrap();
        assert!(added.iter().any(|d| d.get("name").and_then(|j| j.as_str()) == Some("double")));

        // view it.
        let r = handle_request(&mut st, &req(2, "view", vec![("target", s("double"))]));
        assert_eq!(r.get("id").and_then(|j| j.as_i64()), Some(2));
        let result = ok_result(&r);
        let src = result.get("source").and_then(|j| j.as_str()).unwrap();
        assert!(src.contains("double"), "view source: {}", src);
        assert!(result.get("hash").and_then(|j| j.as_str()).is_some());
    }

    #[test]
    fn rename_via_protocol() {
        let mut st = open_state("rename");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def double(x: Int) -> Int = x * 2"))]),
        );
        let r = handle_request(
            &mut st,
            &req(2, "rename", vec![("from", s("double")), ("to", s("twice"))]),
        );
        let result = ok_result(&r);
        let renamed = result.get("renamed").and_then(|j| j.as_array()).unwrap();
        assert_eq!(renamed.len(), 1);
        assert_eq!(renamed[0].get("from").and_then(|j| j.as_str()), Some("double"));
        assert_eq!(renamed[0].get("to").and_then(|j| j.as_str()), Some("twice"));
        // The new name resolves; the old does not.
        assert!(st.cb.get_name("twice").is_some());
        assert!(st.cb.get_name("double").is_none());
    }

    #[test]
    fn unknown_op_is_structured_error_not_silent_success() {
        let mut st = open_state("unknown");
        let r = handle_request(&mut st, &req(9, "frobnicate", vec![]));
        assert_eq!(r.get("id").and_then(|j| j.as_i64()), Some(9));
        assert_eq!(r.get("ok").and_then(|j| j.as_bool()), Some(false));
        assert_eq!(
            r.get("error").and_then(|e| e.get("kind")).and_then(|j| j.as_str()),
            Some("UnknownOp")
        );
        // The unknown-op error carries suggestions so an agent can recover.
        assert!(
            r.get("error").and_then(|e| e.get("did_you_mean")).is_some(),
            "unknown-op error should include did_you_mean suggestions"
        );
    }

    #[test]
    fn canonical_op_resolves_aliases_and_spellings() {
        // The CLI/server naming mismatch that bit a live session: `usages`
        // (CLI) must resolve to the server's `find_usages`. Plus hyphen/case
        // normalization and friendly synonyms.
        assert_eq!(canonical_op("usages"), Some("find_usages"));
        assert_eq!(canonical_op("find-usages"), Some("find_usages"));
        assert_eq!(canonical_op("FIND_USAGES"), Some("find_usages"));
        assert_eq!(canonical_op("find-by-type"), Some("find_by_type"));
        assert_eq!(canonical_op("reorder-params"), Some("reorder_params"));
        assert_eq!(canonical_op("run"), Some("eval"));
        assert_eq!(canonical_op("help"), Some("ops"));
        assert_eq!(canonical_op("view"), Some("view"));
        assert_eq!(canonical_op("frobnicate"), None);
    }

    #[test]
    fn usages_alias_works_end_to_end() {
        let mut st = open_state("usages_alias");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def leaf(x: Int) -> Int = x + 1"))]),
        );
        handle_request(
            &mut st,
            &req(2, "add", vec![("source", s("def caller(y: Int) -> Int = leaf(y)"))]),
        );
        // Drive the op by the WRONG-for-the-server (but natural) name.
        let r = handle_request(&mut st, &req(3, "usages", vec![("target", s("leaf"))]));
        let result = ok_result(&r);
        let usages = result.get("usages").and_then(|j| j.as_array()).unwrap();
        assert_eq!(usages.len(), 1);
        assert_eq!(usages[0].get("name").and_then(|j| j.as_str()), Some("caller"));
    }

    #[test]
    fn ops_discovery_lists_every_dispatchable_op() {
        let mut st = open_state("ops_discovery");
        let r = handle_request(&mut st, &req(1, "ops", vec![]));
        let result = ok_result(&r);
        let ops = result.get("ops").and_then(|j| j.as_array()).unwrap();
        // Every op the catalogue advertises must actually be dispatchable
        // (canonical_op recognizes it) — keeps the docs honest.
        for entry in ops {
            let name = entry.get("op").and_then(|j| j.as_str()).unwrap();
            assert_eq!(
                canonical_op(name),
                Some(name),
                "advertised op `{name}` must be dispatchable"
            );
            assert!(entry.get("summary").and_then(|j| j.as_str()).is_some());
            assert!(entry.get("kind").and_then(|j| j.as_str()).is_some());
        }
    }

    #[test]
    fn stale_expect_hash_rejects_with_conflict_and_no_change() {
        let mut st = open_state("stale_update");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def f(x: Int) -> Int = x * 2"))]),
        );
        let before = st.cb.get_name("f").unwrap();

        // Update with a stale expect_hash (all-zero hash that can't match).
        let stale = "0".repeat(64);
        let r = handle_request(
            &mut st,
            &Json::obj([
                ("id".to_string(), Json::Int(2)),
                ("op".to_string(), s("update")),
                (
                    "params".to_string(),
                    Json::obj([
                        ("name".to_string(), s("f")),
                        ("source".to_string(), s("def f(x: Int) -> Int = x * 3")),
                        ("expect_hash".to_string(), Json::Str(stale.clone())),
                    ]),
                ),
            ]),
        );
        assert_eq!(r.get("ok").and_then(|j| j.as_bool()), Some(false));
        let err = r.get("error").unwrap();
        assert_eq!(err.get("kind").and_then(|j| j.as_str()), Some("Conflict"));
        // The conflict carries the actual current hash so the agent can refetch.
        assert_eq!(
            err.get("actual_hash").and_then(|j| j.as_str()),
            Some(before.to_hex().as_str())
        );
        // NOTHING changed: f still points at the original hash.
        assert_eq!(st.cb.get_name("f"), Some(before));
    }

    #[test]
    fn correct_expect_hash_allows_update() {
        let mut st = open_state("good_update");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def f(x: Int) -> Int = x * 2"))]),
        );
        let before = st.cb.get_name("f").unwrap();

        let r = handle_request(
            &mut st,
            &Json::obj([
                ("id".to_string(), Json::Int(2)),
                ("op".to_string(), s("update")),
                (
                    "params".to_string(),
                    Json::obj([
                        ("name".to_string(), s("f")),
                        ("source".to_string(), s("def f(x: Int) -> Int = x * 3")),
                        ("expect_hash".to_string(), Json::Str(before.to_hex())),
                    ]),
                ),
            ]),
        );
        let result = ok_result(&r);
        // The update committed: f points at a NEW hash now.
        let after = st.cb.get_name("f").unwrap();
        assert_ne!(after, before, "update should change f's hash");
        let updated = result.get("updated").and_then(|j| j.as_array()).unwrap();
        assert_eq!(updated[0].get("old").and_then(|j| j.as_str()), Some(before.to_hex().as_str()));
        assert_eq!(updated[0].get("new").and_then(|j| j.as_str()), Some(after.to_hex().as_str()));
    }

    #[test]
    fn stale_expect_hash_rejects_rename_too() {
        let mut st = open_state("stale_rename");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def g(x: Int) -> Int = x"))]),
        );
        let before = st.cb.get_name("g").unwrap();
        let stale = "f".repeat(64);
        let r = handle_request(
            &mut st,
            &Json::obj([
                ("id".to_string(), Json::Int(2)),
                ("op".to_string(), s("rename")),
                (
                    "params".to_string(),
                    Json::obj([
                        ("from".to_string(), s("g")),
                        ("to".to_string(), s("gg")),
                        ("expect_hash".to_string(), Json::Str(stale)),
                    ]),
                ),
            ]),
        );
        assert_eq!(
            r.get("error").and_then(|e| e.get("kind")).and_then(|j| j.as_str()),
            Some("Conflict")
        );
        // Rename did NOT happen.
        assert!(st.cb.get_name("g").is_some());
        assert!(st.cb.get_name("gg").is_none());
        assert_eq!(st.cb.get_name("g"), Some(before));
    }

    #[test]
    fn branch_switch_diff_via_protocol() {
        let mut st = open_state("branch_diff");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def f(x: Int) -> Int = x"))]),
        );
        // Create branch b from current.
        let r = handle_request(&mut st, &req(2, "branch", vec![("name", s("b"))]));
        ok_result(&r);
        // Switch to b.
        let r = handle_request(&mut st, &req(3, "switch", vec![("name", s("b"))]));
        let result = ok_result(&r);
        assert_eq!(result.get("switched_to").and_then(|j| j.as_str()), Some("b"));

        // On b, add a new def g.
        handle_request(
            &mut st,
            &req(4, "add", vec![("source", s("def g(x: Int) -> Int = x"))]),
        );

        // diff main vs b: g is added on b.
        let r = handle_request(
            &mut st,
            &req(5, "diff", vec![("a", s("main")), ("b", s("b"))]),
        );
        let result = ok_result(&r);
        let added = result.get("added").and_then(|j| j.as_array()).unwrap();
        assert!(
            added.iter().any(|x| x.get("name").and_then(|j| j.as_str()) == Some("g")),
            "expected g in diff added, got {}",
            result.to_string()
        );

        // branches lists both.
        let r = handle_request(&mut st, &req(6, "branches", vec![]));
        let result = ok_result(&r);
        let branches: Vec<&str> = result
            .get("branches")
            .and_then(|j| j.as_array())
            .unwrap()
            .iter()
            .filter_map(|j| j.as_str())
            .collect();
        assert!(branches.contains(&"main"));
        assert!(branches.contains(&"b"));
        assert_eq!(result.get("current").and_then(|j| j.as_str()), Some("b"));
    }

    #[test]
    fn branch_field_routes_op_to_named_branch() {
        let mut st = open_state("branch_field");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def f(x: Int) -> Int = x"))]),
        );
        handle_request(&mut st, &req(2, "branch", vec![("name", s("scratch"))]));
        // An op carrying branch:"scratch" should switch us there.
        let r = Json::obj([
            ("id".to_string(), Json::Int(3)),
            ("op".to_string(), s("ls")),
            ("branch".to_string(), s("scratch")),
            ("params".to_string(), Json::obj(Vec::<(String, Json)>::new())),
        ]);
        let resp = handle_request(&mut st, &r);
        ok_result(&resp);
        assert_eq!(namespace::current_branch(&st.cb).unwrap(), "scratch");

        // A branch that doesn't exist is a hard error, not silent.
        let r = Json::obj([
            ("id".to_string(), Json::Int(4)),
            ("op".to_string(), s("ls")),
            ("branch".to_string(), s("nope")),
            ("params".to_string(), Json::obj(Vec::<(String, Json)>::new())),
        ]);
        let resp = handle_request(&mut st, &r);
        assert_eq!(
            resp.get("error").and_then(|e| e.get("kind")).and_then(|j| j.as_str()),
            Some("NoSuchBranch")
        );
    }

    /// HEADLINE: a type-changing update that breaks a dependent records a
    /// persisted todo that survives across `todo` queries, and a later fix
    /// clears it. This is the whole point of the persisted worklist.
    fn todos_of(resp: &Json) -> Vec<Json> {
        ok_result(resp)
            .get("todos")
            .and_then(|j| j.as_array())
            .map(|a| a.to_vec())
            .unwrap_or_default()
    }

    #[test]
    fn persisted_todo_survives_then_clears_on_fix() {
        let mut st = open_state("persist_todo");
        // f returns Int; g calls f and adds 1 (Int context).
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def f(x: Int) -> Int = x"))]),
        );
        handle_request(
            &mut st,
            &req(2, "add", vec![("source", s("def g(x: Int) -> Int = f(x) + 1"))]),
        );

        // todo starts empty.
        let r = handle_request(&mut st, &req(3, "todo", vec![]));
        assert_eq!(todos_of(&r).len(), 0, "no todos before any breakage");

        // Type-changing update: f now returns Bool. g's `f(x) + 1` no longer
        // typechecks → an inline todo is produced AND persisted.
        let r = handle_request(
            &mut st,
            &req(
                4,
                "update",
                vec![("name", s("f")), ("source", s("def f(x: Int) -> Bool = true"))],
            ),
        );
        let updated = ok_result(&r);
        let inline_todos = updated.get("todos").and_then(|j| j.as_array()).unwrap();
        assert!(
            inline_todos.iter().any(|t| t.get("name").and_then(|j| j.as_str()) == Some("g")),
            "update should report g as a todo inline, got {}",
            updated.to_string()
        );

        // The headline: a SEPARATE `todo` query now returns g (persisted, not
        // just inline) — even though this is a fresh request.
        let r = handle_request(&mut st, &req(5, "todo", vec![]));
        let persisted = todos_of(&r);
        assert!(
            persisted.iter().any(|t| t.get("name").and_then(|j| j.as_str()) == Some("g")),
            "todo op must return persisted g, got {}",
            r.to_string()
        );
        // And it carries a stable code, not just prose.
        let g_entry = persisted
            .iter()
            .find(|t| t.get("name").and_then(|j| j.as_str()) == Some("g"))
            .unwrap();
        assert!(g_entry.get("code").and_then(|j| j.as_str()).is_some());

        // Now FIX g so it typechecks against f: Bool. The fix clears the todo.
        handle_request(
            &mut st,
            &req(
                6,
                "update",
                vec![("name", s("g")), ("source", s("def g(x: Int) -> Bool = f(x)"))],
            ),
        );

        // todo is now empty: the persisted entry was cleared on the fix.
        let r = handle_request(&mut st, &req(7, "todo", vec![]));
        assert_eq!(
            todos_of(&r).len(),
            0,
            "todo must be empty after the fix, got {}",
            r.to_string()
        );
    }

    #[test]
    fn dry_run_update_does_not_commit() {
        let mut st = open_state("dry_run");
        handle_request(
            &mut st,
            &req(1, "add", vec![("source", s("def f(x: Int) -> Int = x * 2"))]),
        );
        let before = st.cb.get_name("f").unwrap();
        let r = handle_request(
            &mut st,
            &req(
                2,
                "update",
                vec![
                    ("name", s("f")),
                    ("source", s("def f(x: Int) -> Int = x * 5")),
                    ("dry_run", Json::Bool(true)),
                ],
            ),
        );
        ok_result(&r);
        // Name did NOT move (dry run).
        assert_eq!(st.cb.get_name("f"), Some(before));
    }
}
