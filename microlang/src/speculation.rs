//! The speculation + deoptimization axis.
//!
//! A speculative tier assumes something (from feedback) and runs a fast path
//! guarded by a cheap check; when the guard fails it *deoptimizes* — abandons
//! the assumption and completes on a general fallback, correctly. Two
//! replaceable interface boundaries:
//!
//!   * `Speculative` is a `Dispatch` strategy that WRAPS an inner (fallback)
//!     `Dispatch`. Because dispatch is a runtime hook both execution tiers call
//!     (`rt.resolve_method`), speculation composes with the interpreter AND the
//!     closure-compiler for free, and it is swapped in with `set_dispatch` like
//!     any other dispatch strategy.
//!
//!   * `SpeculationPolicy` is the strategy inside it: what to speculate and when
//!     to give up. Swap the policy, the mechanism is untouched.
//!
//! WHY a `Dispatch` strategy and not a `CodeSpace` wrapper: a node-level wrapper
//! can only intercept operations that backends route through `top`, and a
//! COMPILING backend inlines the dispatch node, so it never routes it. The
//! runtime hook survives compilation; the node does not. This is the same
//! erased-node boundary as `Traced` + `ClosureComp`, discovered again.
//!
//! HONESTY about scope: in an interpreter, speculation-of-dispatch reduces to an
//! adaptive dispatch strategy — a guarded cached target plus a policy for when
//! to deopt/blacklist. The guard and deopt are real and counted, and the
//! result is provably unchanged (each deopt reconciles with the actual type).
//! The DISTINCTIVE native payoff — inlining the impl body THROUGH the guard,
//! unboxing across it, and mid-frame deopt with SSA→frame state reconstruction —
//! needs the emit tier; that is the `emit_guard` + deopt-metadata interface the
//! codegen-axes doc describes.
//!
//! Coupling with GC: the cached target is an impl pointer a moving collection
//! would dangle, so `on_gc` clears the caches (and forwards to the inner
//! strategy). Because `Runtime::collect` calls `dispatch.on_gc()`, and this IS
//! the dispatch strategy, speculation gets correct GC invalidation with no extra
//! wiring.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::dispatch::{Dispatch, DispatchStats, MethodRegistry};
use crate::value::Sym;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Respeculate,
    Blacklist,
}

/// The replaceable strategy: WHAT to speculate and WHEN to stop.
pub trait SpeculationPolicy {
    fn speculate(&self, site: usize, ty: Sym) -> bool;
    fn after_deopt(&self, site: usize, deopts: u32) -> Decision;
    fn name(&self) -> &'static str;
}

pub struct NeverSpeculate;
impl SpeculationPolicy for NeverSpeculate {
    fn speculate(&self, _: usize, _: Sym) -> bool {
        false
    }
    fn after_deopt(&self, _: usize, _: u32) -> Decision {
        Decision::Blacklist
    }
    fn name(&self) -> &'static str {
        "NeverSpeculate"
    }
}

pub struct AlwaysMonomorphic;
impl SpeculationPolicy for AlwaysMonomorphic {
    fn speculate(&self, _: usize, _: Sym) -> bool {
        true
    }
    fn after_deopt(&self, _: usize, _: u32) -> Decision {
        Decision::Respeculate
    }
    fn name(&self) -> &'static str {
        "AlwaysMonomorphic"
    }
}

pub struct BlacklistAfter(pub u32);
impl SpeculationPolicy for BlacklistAfter {
    fn speculate(&self, _: usize, _: Sym) -> bool {
        true
    }
    fn after_deopt(&self, _: usize, deopts: u32) -> Decision {
        if deopts >= self.0 {
            Decision::Blacklist
        } else {
            Decision::Respeculate
        }
    }
    fn name(&self) -> &'static str {
        "BlacklistAfter"
    }
}

#[derive(Clone, Copy)]
enum Spec {
    Cold,
    Guarded { ty: Sym, target: u64 },
    Blacklisted,
}

#[derive(Clone, Copy)]
struct SiteState {
    spec: Spec,
    deopts: u32,
}
impl Default for SiteState {
    fn default() -> Self {
        SiteState { spec: Spec::Cold, deopts: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpecStats {
    /// Guard hits: cached target used, inner dispatch skipped.
    pub spec_hits: u64,
    /// Guard failures: reconciled with the real type via the inner dispatch.
    pub deopts: u64,
    /// Cold or blacklisted sites resolved via the inner dispatch.
    pub fallbacks: u64,
}

/// Shared, cloneable counters so a caller can read stats after the strategy has
/// been boxed into the runtime.
#[derive(Default)]
pub struct SpecCounters {
    spec_hits: Cell<u64>,
    deopts: Cell<u64>,
    fallbacks: Cell<u64>,
}
impl SpecCounters {
    pub fn snapshot(&self) -> SpecStats {
        SpecStats {
            spec_hits: self.spec_hits.get(),
            deopts: self.deopts.get(),
            fallbacks: self.fallbacks.get(),
        }
    }
}

/// A speculative dispatch strategy: wraps an inner dispatch + a policy.
pub struct Speculative {
    inner: Box<dyn Dispatch>,
    policy: Box<dyn SpeculationPolicy>,
    sites: RefCell<Vec<SiteState>>,
    counters: Arc<SpecCounters>,
}

impl Speculative {
    pub fn new(inner: impl Dispatch + 'static, policy: impl SpeculationPolicy + 'static) -> Self {
        Speculative {
            inner: Box::new(inner),
            policy: Box::new(policy),
            sites: RefCell::new(Vec::new()),
            counters: Arc::new(SpecCounters::default()),
        }
    }
    /// A handle to read stats after this has been boxed into the runtime.
    pub fn counters(&self) -> Arc<SpecCounters> {
        self.counters.clone()
    }

    fn site(&self, site: usize) -> SiteState {
        self.sites.borrow().get(site).copied().unwrap_or_default()
    }
    fn set_site(&self, site: usize, s: SiteState) {
        let mut v = self.sites.borrow_mut();
        if site >= v.len() {
            v.resize(site + 1, SiteState::default());
        }
        v[site] = s;
    }
}

impl Dispatch for Speculative {
    fn resolve(&self, reg: &MethodRegistry, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        let cur = self.site(site);
        match cur.spec {
            Spec::Guarded { ty: aty, target } if aty == ty => {
                // Guard hit: use the cached target, skip the inner dispatch.
                self.counters.spec_hits.set(self.counters.spec_hits.get() + 1);
                Some(target)
            }
            Spec::Guarded { .. } => {
                // Guard fail -> DEOPT: reconcile via the inner dispatch with the
                // ACTUAL type (this is what keeps results correct), then let the
                // policy decide the site's fate.
                self.counters.deopts.set(self.counters.deopts.get() + 1);
                let imp = self.inner.resolve(reg, site, method, ty)?;
                let deopts = cur.deopts + 1;
                let spec = match self.policy.after_deopt(site, deopts) {
                    Decision::Respeculate => Spec::Guarded { ty, target: imp },
                    Decision::Blacklist => Spec::Blacklisted,
                };
                self.set_site(site, SiteState { spec, deopts });
                Some(imp)
            }
            Spec::Blacklisted => {
                self.counters.fallbacks.set(self.counters.fallbacks.get() + 1);
                self.inner.resolve(reg, site, method, ty)
            }
            Spec::Cold => {
                self.counters.fallbacks.set(self.counters.fallbacks.get() + 1);
                let imp = self.inner.resolve(reg, site, method, ty)?;
                if self.policy.speculate(site, ty) {
                    self.set_site(
                        site,
                        SiteState { spec: Spec::Guarded { ty, target: imp }, deopts: cur.deopts },
                    );
                }
                Some(imp)
            }
        }
    }

    fn on_gc(&self) {
        // Cached targets just moved; clear them and forward to the inner cache.
        self.sites.borrow_mut().iter_mut().for_each(|s| {
            *s = SiteState::default();
        });
        self.inner.on_gc();
    }

    fn stats(&self) -> DispatchStats {
        let s = self.counters.snapshot();
        DispatchStats { hits: s.spec_hits, misses: s.deopts + s.fallbacks }
    }

    fn name(&self) -> &'static str {
        "Speculative"
    }
}
