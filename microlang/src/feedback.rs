//! Type-feedback for the JIT's warmup window (the "Always Fast" Phase 1).
//!
//! The first-invoke snapshot the speculative inliner planned from (`SpecEnv`)
//! saw the FIRST call's argument values and an almost-empty per-thread
//! `site_ic` — so a body compiled on first invocation could not know which
//! receiver types actually flow through its dispatch sites. This table gives
//! the JIT a real warmup signal: every dispatch that reaches a SLOW path (the
//! JIT's `shim_dispatch`, the interpreter's `resolve_or_default`) records the
//! receiver type here, UNCONDITIONALLY — NOT gated on `Dispatch::thread_cacheable`
//! the way the per-thread `site_ic` is. That gating is what made compile-time
//! observation go dark under any observing dispatch strategy; the histogram
//! must see every strategy, because a recompile reads it to decide what to
//! speculate on.
//!
//! Cost model: the bump lives on the SLOW path only. The JIT's emitted 2-way
//! dispatch IC and the interpreter's `site_ic` serve the hot repeat dispatches
//! without ever reaching here, so a monomorphic hot site contributes only the
//! handful of slow-path samples that seeded those caches (one per epoch change)
//! — enough to name its dominant type, which is all a recompile needs.
//!
//! GC: receiver types are interned `Sym`s (immediate, GC-stable), so the
//! histogram needs no collector hook. The advisory `last_impl` hint IS a heap
//! pointer a moving collection would dangle, so it is stored with the fold-epoch
//! at record time and only handed back when the caller's current epoch matches
//! (`impl_hint`) — the same revalidation `resolve_or_default` applies to its own
//! cached impl.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;

use crate::value::Sym;

/// How many distinct receiver types one site tracks before it stops widening
/// the histogram (`total` still counts every dispatch). Four covers the
/// monomorphic/bimorphic sites speculation targets and the tri/quad-morphic
/// ones it deliberately leaves generic.
const HIST_WAYS: usize = 4;

/// Per-dispatch-site feedback: a tiny receiver-type histogram plus a deopt
/// tally and an advisory last-impl hint. Every field is atomic so the slow
/// paths of multiple JIT worker threads can bump it lock-free under a shared
/// read guard; the counts are APPROXIMATE under races (claimed-slot / bump
/// interleavings can drop or double a count), which is fine — only the ROUTING
/// decision reads them, never a correctness-bearing one.
pub struct SiteFeedback {
    ty: [AtomicU32; HIST_WAYS],
    cnt: [AtomicU32; HIST_WAYS],
    total: AtomicU32,
    deopts: AtomicU32,
    last_impl: AtomicU64,
    last_epoch: AtomicU64,
}

impl SiteFeedback {
    fn new() -> Self {
        SiteFeedback {
            ty: std::array::from_fn(|_| AtomicU32::new(0)),
            cnt: std::array::from_fn(|_| AtomicU32::new(0)),
            total: AtomicU32::new(0),
            deopts: AtomicU32::new(0),
            last_impl: AtomicU64::new(0),
            last_epoch: AtomicU64::new(0),
        }
    }

    fn record(&self, ty: Sym, imp: u64, epoch: u64) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.last_impl.store(imp, Ordering::Relaxed);
        self.last_epoch.store(epoch, Ordering::Relaxed);
        // A slot is EMPTY when its count is 0 (a real type always has count > 0
        // once claimed). Sym 0 is a valid interned symbol, so emptiness cannot
        // key on the type word — it keys on the count.
        for i in 0..HIST_WAYS {
            let c = self.cnt[i].load(Ordering::Relaxed);
            if c == 0 {
                // Claim an empty way. Racy (two threads may claim the same way
                // for different types) but self-correcting: the loser's type
                // just re-appears on its next dispatch and finds/claims a way.
                self.ty[i].store(ty, Ordering::Relaxed);
                self.cnt[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
            if self.ty[i].load(Ordering::Relaxed) == ty {
                self.cnt[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // Histogram full (> HIST_WAYS distinct types): a megamorphic site. Stop
        // widening — `total` still counts, so `dominant_type`'s fraction falls
        // and speculation correctly declines.
    }

    /// The most-seen receiver type and its fraction of all recorded dispatches.
    fn dominant(&self) -> Option<(Sym, f32)> {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let mut best_ty = 0u32;
        let mut best_cnt = 0u32;
        for i in 0..HIST_WAYS {
            let c = self.cnt[i].load(Ordering::Relaxed);
            if c > best_cnt {
                best_cnt = c;
                best_ty = self.ty[i].load(Ordering::Relaxed);
            }
        }
        if best_cnt == 0 {
            return None;
        }
        Some((best_ty, best_cnt as f32 / total as f32))
    }

    /// The two most-seen receiver types (type, count), most-frequent first;
    /// missing ways come back as `(0, 0)`.
    fn top2(&self) -> [(Sym, u32); 2] {
        let mut ranked: [(Sym, u32); HIST_WAYS] = std::array::from_fn(|i| {
            (self.ty[i].load(Ordering::Relaxed), self.cnt[i].load(Ordering::Relaxed))
        });
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        [ranked[0], ranked[1]]
    }
}

/// The process-shared feedback table, one `SiteFeedback` per dispatch `site`
/// id. Grown lazily to the highest site id seen (a real program uses hundreds
/// to a few thousand of the `1<<17`-wide site space, so eager allocation would
/// waste megabytes). Hot-path bumps take only a SHARED read guard + atomics;
/// the exclusive write guard is taken solely to extend the vector for a
/// never-before-seen site.
pub struct FeedbackTable {
    sites: RwLock<Vec<SiteFeedback>>,
}

impl Default for FeedbackTable {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackTable {
    pub fn new() -> Self {
        FeedbackTable { sites: RwLock::new(Vec::new()) }
    }

    /// Record one slow-path dispatch: bump `ty`'s histogram way, remember the
    /// resolved impl (advisory) with its fold-epoch. `epoch` is the same
    /// `reloc*MIX ^ ver` fold `resolve_or_default` uses.
    pub fn record(&self, site: usize, ty: Sym, imp: u64, epoch: u64) {
        {
            let g = self.sites.read().unwrap();
            if let Some(sf) = g.get(site) {
                sf.record(ty, imp, epoch);
                return;
            }
        }
        // First time this site id is seen: grow under the exclusive guard, then
        // record. (Re-check the length — another thread may have grown it.)
        let mut g = self.sites.write().unwrap();
        if g.len() <= site {
            g.resize_with(site + 1, SiteFeedback::new);
        }
        g[site].record(ty, imp, epoch);
    }

    /// Count a deoptimization at `site`: a compile-time speculation's guard
    /// failed and the generic edge ran instead. Drives the blacklist policy.
    pub fn record_deopt(&self, site: usize) {
        let g = self.sites.read().unwrap();
        if let Some(sf) = g.get(site) {
            sf.deopts.fetch_add(1, Ordering::Relaxed);
        }
        // A deopt at an unseen site is impossible (the site was speculated on,
        // so it was recorded first) — no grow path needed.
    }

    /// The dominant receiver type at `site` and its fraction of all dispatches,
    /// or `None` if the site was never recorded.
    pub fn dominant_type(&self, site: usize) -> Option<(Sym, f32)> {
        self.sites.read().unwrap().get(site)?.dominant()
    }

    /// The two hottest receiver types at `site` (see `SiteFeedback::top2`).
    pub fn top2(&self, site: usize) -> Option<[(Sym, u32); 2]> {
        Some(self.sites.read().unwrap().get(site)?.top2())
    }

    /// Deopts recorded at `site` so far (0 if unseen).
    pub fn deopts(&self, site: usize) -> u32 {
        self.sites
            .read()
            .unwrap()
            .get(site)
            .map_or(0, |sf| sf.deopts.load(Ordering::Relaxed))
    }

    /// The advisory last-resolved impl at `site`, but ONLY if it is still valid
    /// at `cur_epoch` (no GC move / redefinition since it was recorded). A heap
    /// pointer, so a stale epoch must never hand it back.
    pub fn impl_hint(&self, site: usize, cur_epoch: u64) -> Option<u64> {
        let g = self.sites.read().unwrap();
        let sf = g.get(site)?;
        if sf.last_epoch.load(Ordering::Relaxed) == cur_epoch {
            let imp = sf.last_impl.load(Ordering::Relaxed);
            if imp != 0 {
                return Some(imp);
            }
        }
        None
    }
}
