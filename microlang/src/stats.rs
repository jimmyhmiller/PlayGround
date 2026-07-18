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

use std::sync::atomic::AtomicU64;

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
