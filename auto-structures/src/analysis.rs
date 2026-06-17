//! Usage analysis: the compiler's job is to look at how each abstract
//! collection is *used* and decide which concrete data structure fits best.
//!
//! A collection is any variable bound by `let v = collection()`. We then walk
//! the whole program collecting the set of operations applied to `v` (the
//! "usage profile"), and map that profile to a `Kind`. The runtime turns the
//! `Kind` into a concrete structure (see `value.rs`).

use crate::ast::*;
use crate::value::Kind;
use std::collections::{BTreeSet, HashMap};

/// What we learned about one collection variable.
#[derive(Clone)]
pub struct Selection {
    pub name: String,
    pub ops: BTreeSet<String>,
    pub specialized: &'static str,
    pub naive: &'static str,
    pub reason: String,
}

pub struct Analysis {
    pub selections: Vec<Selection>,
    by_name: HashMap<String, Kind>,
}

impl Analysis {
    /// Kind chosen for a collection variable, if it is an inferred collection.
    pub fn kind_of(&self, name: &str) -> Option<Kind> {
        self.by_name.get(name).copied()
    }
}

/// Operation names that, when applied to a collection, reveal what it must do.
const KNOWN_OPS: &[&str] = &[
    "add", "has", "del", "put", "get", "keys", "append", "at", "set_at", "push", "pop", "peek",
    "enqueue", "dequeue", "front", "min", "max", "sorted", "size", "len", "iter",
];

pub fn analyze(program: &Program) -> Analysis {
    // 1. Find every variable bound to `collection()`.
    let mut coll_vars: Vec<String> = Vec::new();
    collect_collection_vars(program, &mut coll_vars);

    // 2. Gather a usage profile for each one.
    let mut profiles: HashMap<String, BTreeSet<String>> = HashMap::new();
    for v in &coll_vars {
        profiles.insert(v.clone(), BTreeSet::new());
    }
    gather_uses(program, &mut profiles);

    // 3. Pick a representation per collection.
    let mut selections = Vec::new();
    let mut by_name = HashMap::new();
    for v in &coll_vars {
        let ops = profiles.remove(v).unwrap_or_default();
        let (kind, reason) = select(&ops);
        by_name.insert(v.clone(), kind);
        let (specialized, naive) = repr_names(kind);
        selections.push(Selection {
            name: v.clone(),
            ops,
            specialized,
            naive,
            reason,
        });
    }

    Analysis { selections, by_name }
}

/// Map a usage profile to a `Kind`, with an explanation for the report.
fn select(ops: &BTreeSet<String>) -> (Kind, String) {
    let has = |o: &str| ops.contains(o);
    let associative = has("put") || has("get") || has("keys");
    let sequence = has("append") || has("at") || has("set_at");
    let stack = has("push") || has("pop") || has("peek");
    let queue = has("enqueue") || has("dequeue") || has("front");
    let setish = has("add") || has("has") || has("del");
    let ordered = has("min") || has("max") || has("sorted");

    if associative {
        let kind = Kind::Map { ordered };
        let reason = if ordered {
            "keyed lookups (put/get) AND ordered traversal (sorted/min/max) → an ordered map"
        } else {
            "keyed lookups (put/get) with no ordering requirement → a hash map"
        };
        return (kind, reason.to_string());
    }

    if queue && !sequence && !stack {
        return (
            Kind::Queue,
            "FIFO access (enqueue/dequeue/front) → a queue (ring buffer)".to_string(),
        );
    }

    if sequence || stack {
        let reason = if stack && !sequence {
            "LIFO access (push/pop/peek) → a growable array used as a stack"
        } else {
            "positional access (append/at/set_at) → a growable array"
        };
        return (Kind::Sequence, reason.to_string());
    }

    if setish {
        let kind = Kind::Set { ordered };
        let reason = if ordered {
            "membership tests (add/has) AND ordered traversal (sorted/min/max) → an ordered set"
        } else {
            "membership tests (add/has/del) with no ordering requirement → a hash set"
        };
        return (kind, reason.to_string());
    }

    // Nothing distinguishing was observed (maybe only iterated or sized).
    let kind = Kind::Set { ordered };
    (
        kind,
        "no distinguishing operations observed; defaulting to a set".to_string(),
    )
}

fn repr_names(kind: Kind) -> (&'static str, &'static str) {
    match kind {
        Kind::Set { ordered: false } => ("HashSet — O(1) membership", "linear scan — O(n) membership"),
        Kind::Set { ordered: true } => ("BTreeSet — O(log n), ordered", "linear scan + sort"),
        Kind::Map { ordered: false } => ("HashMap — O(1) lookup", "linear assoc list — O(n) lookup"),
        Kind::Map { ordered: true } => ("BTreeMap — O(log n), ordered", "linear assoc list + sort"),
        Kind::Sequence => ("Vec — O(1) index/append", "Vec — O(1) index/append"),
        Kind::Queue => ("VecDeque — O(1) dequeue", "Vec shift — O(n) dequeue"),
    }
}

fn collect_collection_vars(stmts: &[Stmt], out: &mut Vec<String>) {
    for s in stmts {
        match s {
            Stmt::Let { name, value } => {
                if is_collection_ctor(value) && !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Stmt::If { then, els, .. } => {
                collect_collection_vars(then, out);
                collect_collection_vars(els, out);
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => {
                collect_collection_vars(body, out);
            }
            _ => {}
        }
    }
}

fn is_collection_ctor(e: &Expr) -> bool {
    matches!(e, Expr::Call(name, args) if name == "collection" && args.is_empty())
}

fn gather_uses(stmts: &[Stmt], profiles: &mut HashMap<String, BTreeSet<String>>) {
    for s in stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Expr(value) => {
                gather_expr(value, profiles);
            }
            Stmt::If { cond, then, els } => {
                gather_expr(cond, profiles);
                gather_uses(then, profiles);
                gather_uses(els, profiles);
            }
            Stmt::While { cond, body } => {
                gather_expr(cond, profiles);
                gather_uses(body, profiles);
            }
            Stmt::For { iter, body, .. } => {
                // Iterating a collection is itself a use.
                if let Expr::Var(v) = iter {
                    note(profiles, v, "iter");
                }
                gather_expr(iter, profiles);
                gather_uses(body, profiles);
            }
            Stmt::Break | Stmt::Continue => {}
        }
    }
}

fn gather_expr(e: &Expr, profiles: &mut HashMap<String, BTreeSet<String>>) {
    match e {
        Expr::Call(name, args) => {
            if KNOWN_OPS.contains(&name.as_str()) {
                if let Some(Expr::Var(v)) = args.first() {
                    note(profiles, v, name);
                }
            }
            for a in args {
                gather_expr(a, profiles);
            }
        }
        Expr::Unary(_, inner) => gather_expr(inner, profiles),
        Expr::Binary(_, l, r) => {
            gather_expr(l, profiles);
            gather_expr(r, profiles);
        }
        Expr::List(items) => {
            for it in items {
                gather_expr(it, profiles);
            }
        }
        _ => {}
    }
}

fn note(profiles: &mut HashMap<String, BTreeSet<String>>, var: &str, op: &str) {
    if let Some(set) = profiles.get_mut(var) {
        set.insert(op.to_string());
    }
}
