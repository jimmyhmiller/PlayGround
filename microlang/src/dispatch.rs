//! The dispatch axis: how a polymorphic call site resolves `(method, receiver
//! type)` to an implementation. This is where dynamic languages spend their
//! performance, and it is a swappable strategy exactly like value layout and GC.
//!
//! Every strategy answers one question — given a call SITE, a method, and the
//! receiver's type, which impl? — and they differ only in what per-site state
//! they keep. All produce identical results; they differ in how much work each
//! call does. Three real strategies here, the classic ladder:
//!
//!   * `Megamorphic`    — no cache; hit the global registry every call.
//!   * `MonomorphicIc`  — cache one (type -> impl) per site; great when a site
//!                        always sees one type, thrashes when it sees many.
//!   * `PolymorphicIc`  — cache up to k per site; handles a few types per site,
//!                        falls back to a registry lookup beyond k.
//!
//! Coupling with GC (from the codegen-axes graph): cached impl pointers would
//! dangle after a moving collection, so `on_gc` clears the caches (they refill).
//! The method registry itself holds impl pointers that ARE roots — the collector
//! forwards them. Caches are a performance optimization; the registry is truth.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::value::Sym;

/// Maps `(method name, receiver type)` to an implementation closure ref. The
/// single source of truth; a GC root.
pub type MethodRegistry = HashMap<(Sym, Sym), u64>;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DispatchStats {
    pub hits: u64,
    pub misses: u64,
}

pub trait Dispatch {
    /// Resolve a call site to an impl, or `None` if no method is defined for
    /// this receiver type.
    fn resolve(&self, reg: &MethodRegistry, site: usize, method: Sym, ty: Sym) -> Option<u64>;
    /// Invalidate caches (called by the collector, since cached refs move).
    fn on_gc(&self) {}
    fn stats(&self) -> DispatchStats {
        DispatchStats::default()
    }
    fn name(&self) -> &'static str;
}

// ── Megamorphic: no cache ───────────────────────────────────
pub struct Megamorphic {
    lookups: Cell<u64>,
}
impl Megamorphic {
    pub fn new() -> Self {
        Megamorphic { lookups: Cell::new(0) }
    }
}
impl Dispatch for Megamorphic {
    fn resolve(&self, reg: &MethodRegistry, _site: usize, method: Sym, ty: Sym) -> Option<u64> {
        self.lookups.set(self.lookups.get() + 1);
        reg.get(&(method, ty)).copied()
    }
    fn stats(&self) -> DispatchStats {
        DispatchStats { hits: 0, misses: self.lookups.get() }
    }
    fn name(&self) -> &'static str {
        "Megamorphic"
    }
}

// ── Monomorphic inline cache: one entry per site ────────────
pub struct MonomorphicIc {
    sites: RefCell<Vec<Option<(Sym, u64)>>>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}
impl MonomorphicIc {
    pub fn new() -> Self {
        MonomorphicIc {
            sites: RefCell::new(Vec::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }
}
impl Dispatch for MonomorphicIc {
    fn resolve(&self, reg: &MethodRegistry, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        let mut s = self.sites.borrow_mut();
        if site >= s.len() {
            s.resize(site + 1, None);
        }
        if let Some((cty, cimpl)) = s[site] {
            if cty == ty {
                self.hits.set(self.hits.get() + 1);
                return Some(cimpl);
            }
        }
        self.misses.set(self.misses.get() + 1);
        let imp = reg.get(&(method, ty)).copied()?;
        s[site] = Some((ty, imp)); // (re)fill; a new type evicts the old
        Some(imp)
    }
    fn on_gc(&self) {
        self.sites.borrow_mut().iter_mut().for_each(|e| *e = None);
    }
    fn stats(&self) -> DispatchStats {
        DispatchStats { hits: self.hits.get(), misses: self.misses.get() }
    }
    fn name(&self) -> &'static str {
        "MonomorphicIc"
    }
}

// ── Polymorphic inline cache: up to k entries per site ──────
pub struct PolymorphicIc {
    k: usize,
    sites: RefCell<Vec<Vec<(Sym, u64)>>>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}
impl PolymorphicIc {
    pub fn new(k: usize) -> Self {
        PolymorphicIc {
            k,
            sites: RefCell::new(Vec::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }
}
impl Dispatch for PolymorphicIc {
    fn resolve(&self, reg: &MethodRegistry, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        let mut s = self.sites.borrow_mut();
        if site >= s.len() {
            s.resize_with(site + 1, Vec::new);
        }
        if let Some(&(_, imp)) = s[site].iter().find(|(t, _)| *t == ty) {
            self.hits.set(self.hits.get() + 1);
            return Some(imp);
        }
        self.misses.set(self.misses.get() + 1);
        let imp = reg.get(&(method, ty)).copied()?;
        if s[site].len() < self.k {
            s[site].push((ty, imp)); // cache; beyond k we just don't (megamorphic tail)
        }
        Some(imp)
    }
    fn on_gc(&self) {
        self.sites.borrow_mut().iter_mut().for_each(|v| v.clear());
    }
    fn stats(&self) -> DispatchStats {
        DispatchStats { hits: self.hits.get(), misses: self.misses.get() }
    }
    fn name(&self) -> &'static str {
        "PolymorphicIc"
    }
}
