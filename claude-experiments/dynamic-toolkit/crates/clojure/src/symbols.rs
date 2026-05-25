//! Symbol intern table for the bootstrap Rust reader.
//!
//! Symbols are addressed by a `u32` id; identity is integer compare.
//! In Clojure, symbols can be qualified (`ns/name`) — we store the
//! qualified printed form as a single string and split on lookup if
//! needed. The compiler treats the head-of-list "symbol" as just an
//! id to dispatch on; the runtime Var system later resolves the
//! `ns/name` split into proper namespace lookups.

use std::collections::HashMap;
use std::sync::RwLock;

/// Thread-safe symbol intern table. Concurrency model mirrors
/// JVM Clojure's `clojure.lang.Symbol/intern`: every operation is
/// internally synchronized, so callers just hold `&SymbolTable` and
/// don't worry about locks. Reads (`name`) take a shared lock;
/// mutations (`intern`, `gensym`) take an exclusive lock briefly.
///
/// Critically, neither `&self` nor any returned value retains the
/// lock past the function call. That lets compile/expand JIT-call
/// macro bodies which themselves intern symbols (gensym, runtime
/// `(symbol "x")`, …) without deadlocking — which is what the
/// previous `Mutex<SymbolTable>`-held-across-expand model did.
///
/// `name(id)` returns an owned `String` rather than `&str` because
/// the lock is dropped before we return.
pub struct SymbolTable {
    inner: RwLock<Inner>,
}

struct Inner {
    name_to_id: HashMap<String, u32>,
    /// `Box<str>` keeps the heap-side string data at a stable
    /// address even when the Vec grows; we currently re-clone on
    /// every read but the layout leaves room for a future "interned
    /// strings handed out as `Arc<str>`" upgrade without changing
    /// the public API.
    id_to_name: Vec<Box<str>>,
    gensym_counter: u32,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            inner: RwLock::new(Inner {
                name_to_id: HashMap::new(),
                id_to_name: Vec::new(),
                gensym_counter: 0,
            }),
        }
    }

    pub fn intern(&self, name: &str) -> u32 {
        // Fast path: read-lock, see if it's already there.
        {
            let r = self.inner.read().unwrap();
            if let Some(&id) = r.name_to_id.get(name) {
                return id;
            }
        }
        // Slow path: write-lock, re-check (race), insert.
        let mut w = self.inner.write().unwrap();
        if let Some(&id) = w.name_to_id.get(name) {
            return id;
        }
        let id = w.id_to_name.len() as u32;
        let boxed: Box<str> = name.to_string().into_boxed_str();
        w.id_to_name.push(boxed);
        w.name_to_id.insert(name.to_string(), id);
        id
    }

    pub fn name(&self, id: u32) -> String {
        let r = self.inner.read().unwrap();
        r.id_to_name[id as usize].to_string()
    }

    pub fn gensym(&self, tag: &str) -> u32 {
        let mut w = self.inner.write().unwrap();
        w.gensym_counter += 1;
        let n = w.gensym_counter;
        let s: Box<str> = format!("#:{tag}{n}").into_boxed_str();
        let id = w.id_to_name.len() as u32;
        w.id_to_name.push(s);
        id
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
