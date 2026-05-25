//! Free-variable analysis for `(fn [args] body)` expressions.
//!
//! Walks a Clojure form tree, tracking which symbol-ids are bound by
//! the surrounding lexical structure (`fn` params, `let` bindings,
//! `loop` bindings, …) and reports the symbols that are referenced
//! but not bound by any of those.
//!
//! Free vars are returned in *first-seen* order. This is the order
//! the closure-capture allocator stores them in, and the order the
//! inner fn's prologue reads them out — so deduplication uses an
//! order-preserving hash set.

use std::collections::HashSet;

use crate::collections::{is_list, seq_iter};
use crate::value as v;

/// Special-form heads that introduce bindings or alter walk shape.
/// The walker special-cases these — every other list head is treated
/// as a function call (head + args, all of which need to be walked).
const SPECIAL_HEADS: &[&str] = &[
    "let", "fn", "loop", "do", "if", "quote", "def", "defmacro",
    "recur", "quasiquote", "unquote", "unquote-splicing",
    "try", "throw", "catch",
];

/// Compute free symbol-ids in `body` given that `bound` is initially
/// bound by enclosing structure (e.g. the fn's params).
///
/// `name_of` resolves a symbol-id to its printed name, so the walker
/// can recognize special-form heads.
pub fn free_vars(
    body_forms: u64,
    initial_bound: &[u32],
    name_of: &impl Fn(u32) -> String,
) -> Vec<u32> {
    let mut bound: HashSet<u32> = initial_bound.iter().copied().collect();
    let mut free: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for form in v::list_iter(body_forms) {
        walk(form, &mut bound, &mut free, &mut seen, name_of);
    }
    free
}

/// Same as `free_vars` but takes a single expression rather than a
/// list of body forms. Useful when the caller already has a single
/// form in hand and doesn't want to wrap-and-unwrap into a list.
pub fn free_vars_in_form(
    form: u64,
    initial_bound: &[u32],
    name_of: &impl Fn(u32) -> String,
) -> Vec<u32> {
    let mut bound: HashSet<u32> = initial_bound.iter().copied().collect();
    let mut free: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    walk(form, &mut bound, &mut free, &mut seen, name_of);
    free
}

fn add_free(id: u32, bound: &HashSet<u32>, free: &mut Vec<u32>, seen: &mut HashSet<u32>) {
    if !bound.contains(&id) && !seen.contains(&id) {
        free.push(id);
        seen.insert(id);
    }
}

fn walk(
    form: u64,
    bound: &mut HashSet<u32>,
    free: &mut Vec<u32>,
    seen: &mut HashSet<u32>,
    name_of: &impl Fn(u32) -> String,
) {
    if v::is_sym_id(form) {
        add_free(v::as_sym_id(form), bound, free, seen);
        return;
    }
    if !v::is_ptr(form) {
        return;
    }
    if !is_list(form) {
        // Vectors / maps / strings / keywords are self-evaluating
        // literals and carry no symbol references that close over
        // anything. (Quoted lists are handled in `walk_list`.)
        return;
    }
    walk_list(form, bound, free, seen, name_of);
}

fn walk_list(
    form: u64,
    bound: &mut HashSet<u32>,
    free: &mut Vec<u32>,
    seen: &mut HashSet<u32>,
    name_of: &impl Fn(u32) -> String,
) {
    let head = v::first(form);
    if v::is_sym_id(head) {
        let name = name_of(v::as_sym_id(head));
        if SPECIAL_HEADS.contains(&name.as_str()) {
            walk_special(name.as_str(), v::rest(form), bound, free, seen, name_of);
            return;
        }
        // Generic call: head is a free var if not bound, then walk args.
        add_free(v::as_sym_id(head), bound, free, seen);
        for a in v::list_iter(v::rest(form)) {
            walk(a, bound, free, seen, name_of);
        }
        return;
    }
    // Non-symbol head: walk it as an expression, then args.
    walk(head, bound, free, seen, name_of);
    for a in v::list_iter(v::rest(form)) {
        walk(a, bound, free, seen, name_of);
    }
}

fn walk_special(
    head: &str,
    rest: u64,
    bound: &mut HashSet<u32>,
    free: &mut Vec<u32>,
    seen: &mut HashSet<u32>,
    name_of: &impl Fn(u32) -> String,
) {
    match head {
        "quote" | "quasiquote" | "unquote" | "unquote-splicing" => {
            // Quoted forms reference no runtime variables.
        }
        "let" => {
            // (let [n0 v0 n1 v1 ...] body...)
            let bindings = v::first(rest);
            let body = v::rest(rest);
            let pairs: Vec<u64> = seq_iter(bindings).collect();
            // Track which sym_ids we add so we can pop them later.
            let mut added: Vec<u32> = Vec::new();
            for chunk in pairs.chunks(2) {
                if chunk.len() < 2 {
                    break;
                }
                // Walk the value with the CURRENT bindings (sequential
                // let semantics: later bindings see earlier names).
                walk(chunk[1], bound, free, seen, name_of);
                if v::is_sym_id(chunk[0]) {
                    let id = v::as_sym_id(chunk[0]);
                    if bound.insert(id) {
                        added.push(id);
                    }
                }
            }
            for f in v::list_iter(body) {
                walk(f, bound, free, seen, name_of);
            }
            for id in added {
                bound.remove(&id);
            }
        }
        "loop" => {
            // Same shape as let.
            walk_special("let", rest, bound, free, seen, name_of);
        }
        "fn" => {
            // (fn [args...] body...) — args don't see outer locals
            // for the purpose of "is this captured?", but free vars
            // of the inner fn are themselves captured by the OUTER
            // closure (transitively). Add inner params as bound for
            // the duration of the body walk.
            let arg_vec = v::first(rest);
            let body = v::rest(rest);
            let mut added: Vec<u32> = Vec::new();
            for p in seq_iter(arg_vec) {
                if v::is_sym_id(p) {
                    let id = v::as_sym_id(p);
                    if bound.insert(id) {
                        added.push(id);
                    }
                }
            }
            for f in v::list_iter(body) {
                walk(f, bound, free, seen, name_of);
            }
            for id in added {
                bound.remove(&id);
            }
        }
        "def" | "defmacro" => {
            // The def'd name itself is not a reference; walk only the
            // value form (which is at position 1).
            let _name = v::first(rest);
            for f in v::list_iter(v::rest(rest)) {
                walk(f, bound, free, seen, name_of);
            }
        }
        "do" | "if" | "recur" | "try" | "throw" => {
            // try/throw: walk every subform normally. The catch arm
            // inside a try is itself a list whose head we recurse on
            // below, so its binding gets handled there.
            for f in v::list_iter(rest) {
                walk(f, bound, free, seen, name_of);
            }
        }
        "catch" => {
            // (catch T name body...) — `T` is a type-filter form
            // (currently ignored by the compiler; not a reference);
            // `name` binds the thrown value over `body...`.
            let _type_form = v::first(rest);
            let after_type = v::rest(rest);
            let name_v = v::first(after_type);
            let body = v::rest(after_type);
            let added = if v::is_sym_id(name_v) {
                let id = v::as_sym_id(name_v);
                if bound.insert(id) { Some(id) } else { None }
            } else {
                None
            };
            for f in v::list_iter(body) {
                walk(f, bound, free, seen, name_of);
            }
            if let Some(id) = added {
                bound.remove(&id);
            }
        }
        _ => {
            for f in v::list_iter(rest) {
                walk(f, bound, free, seen, name_of);
            }
        }
    }
}
