//! Process-wide performance counters — always on (a relaxed `fetch_add` on an
//! uncontended cache line is effectively free next to the operations being
//! counted), read by the `%stats` prim and dumped at exit under
//! `MICROLANG_STATS=1`.
//!
//! These are the gauntlet's attribution signal: a slow library names its
//! bottleneck category on its FIRST run — high interp share ⇒ bodies fell off
//! the JIT, high dispatch-shim count ⇒ megamorphic/uncached protocol sites,
//! high bytes/op + minor GCs ⇒ allocation-bound — before anyone reaches for a
//! profiler. (Heap bytes and GC counts live on `Heap`, per-runtime; these are
//! process-wide because the tiers and shims have no runtime handle spare.)
//!
//! The adaptive-tier counters (`REOPT_*`/`SPEC_DISPATCH_*`) are the "Always
//! Fast" Phase 1 signal: how many hot bodies were recompiled with type feedback
//! and how the feedback-driven dispatch speculation fared (sites emitted vs.
//! guard deopts). Same discipline — DIAGNOSTICS, never semantics, approximate
//! under races is fine.

use std::sync::atomic::{AtomicU64, Ordering};

/// TIER-BOUNDARY calls routed to native code by the tiered dispatcher. This is
/// NOT a per-call execution count: JIT'd code calls JIT'd code through its own
/// inline fast path and never crosses `Tiered::invoke` (per-body invocation
/// counters arrive with the baseline tier). What the pair DOES tell you is
/// whether the CEK interpreter is in play at all — interp > 0 means bodies
/// fell off the JIT and every such call pays the boundary.
pub static NATIVE_INVOKES: AtomicU64 = AtomicU64::new(0);
/// Tier-boundary calls routed to the CEK interpreter (non-compilable bodies).
pub static INTERP_INVOKES: AtomicU64 = AtomicU64::new(0);
/// Entries into the generic dispatch shim — the COLD path (the hot path is the
/// inline 2-way IC that never calls the shim), so this count localizes
/// megamorphic / cache-missing protocol sites.
pub static DISPATCH_SHIM_CALLS: AtomicU64 = AtomicU64::new(0);
/// Bodies compiled to native code (compile-once, so also the body count).
pub static JIT_COMPILES: AtomicU64 = AtomicU64::new(0);

/// Bodies recompiled from a first-invoke compile into a feedback-speculating
/// v2 (or later). Bounded per body by the reopt cap, so this grows with how
/// many DISTINCT hot bodies were re-optimized, plus the rare re-speculation.
pub static REOPT_COMPILES: AtomicU64 = AtomicU64::new(0);
/// Feedback-driven dispatch-site speculations EMITTED across all recompiles
/// (one per dispatch site that had a confident dominant type at recompile).
pub static SPEC_DISPATCH_SITES: AtomicU64 = AtomicU64::new(0);
/// Runtime deopts off a feedback-speculated dispatch guard: the receiver type
/// or dispatch version diverged from what was baked, so the generic edge ran.
pub static SPEC_DISPATCH_DEOPTS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// A snapshot of the adaptive-tier counters — the `# STATS` line the bench
/// harness reads to see how often the Phase 1 machinery fired.
#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub reopt_compiles: u64,
    pub spec_dispatch_sites: u64,
    pub spec_dispatch_deopts: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        reopt_compiles: REOPT_COMPILES.load(Ordering::Relaxed),
        spec_dispatch_sites: SPEC_DISPATCH_SITES.load(Ordering::Relaxed),
        spec_dispatch_deopts: SPEC_DISPATCH_DEOPTS.load(Ordering::Relaxed),
    }
}

impl std::fmt::Display for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reopt_compiles={} spec_dispatch_sites={} spec_dispatch_deopts={}",
            self.reopt_compiles, self.spec_dispatch_sites, self.spec_dispatch_deopts
        )
    }
}
