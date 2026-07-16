//! The MOVING (semi-space copying) collector's runtime face: root
//! enumeration, the stop-the-world rendezvous, and the shadow-stack handle
//! discipline. The copying machinery itself lives in `heap.rs` (Cheney
//! evacuation over the `TypeInfo` table); this file's job is to hand it every
//! live root slot.
//!
//! Reachable objects are evacuated to the other space; every root slot is
//! rewritten in place: the globals, the constant pool (which is why `Ir`
//! holds no embedded heap pointers), the shadow stacks, dynamic-var bindings,
//! activation frames (slots + the running closure's `caps_src`), reified
//! continuations, future results, method impls, and the `()` singleton.
//! Lexical frames stay `Arc`-managed with atomic slots, so their heap
//! pointers are rewritten in place and the mutator's `locals` reference stays
//! valid across a collection.
//!
//! The one thing a moving collector cannot fix for free: a bare `u64` the
//! mutator holds directly in a Rust local. After a move it points into the
//! evacuated space and dereferencing it errors loudly (verify mode poisons
//! the space; the header type_id check catches it). The fix is the handle:
//! publish the value to the shadow stack and re-read it (`root_get`) after
//! anything that may allocate.

use std::sync::Arc;

use crate::cek::Kont;
use crate::model::ValueModel;
use crate::runtime::{Runtime, GLOBAL_UNBOUND};
use crate::value::Locals;

/// What a collection is being asked for (Stage I2).
#[derive(Clone, Copy, PartialEq, Eq)]
enum GcKind {
    /// Collect EVERYTHING: a major over the old gen with the nursery as a
    /// second from-space. What the explicit `(gc)` prim means, and what the
    /// tests and the bench battery are written against.
    Major,
    /// The GC's own choice — a minor, escalating to a major per the policy in
    /// `collect_inner`. This is what allocation pressure asks for.
    Auto,
}

/// gc-stress: every safepoint runs a MINOR (each one ending in the
/// missed-barrier walk — that is the hammer this mode exists to swing), and
/// every 8th ALSO runs a major. Majors have to keep getting hit because they
/// are what exercises the old gen's Cheney, the semi-space flip, and the card
/// table's re-point + start-index rebuild against a new base — none of which a
/// minor touches. 8 is chosen so the battery's cost stays dominated by the
/// minors (the thing under test) while a major still lands inside every
/// short-lived program the battery runs.
const STRESS_MAJOR_EVERY: u64 = 8;

/// A handle to a rooted value. `get` re-reads the shadow slot, so it yields the
/// value's CURRENT address even after the collector relocated it.
pub struct Root(usize);

impl Root {
    pub fn get<M: ValueModel>(&self, rt: &Runtime<M>) -> u64 {
        rt.root_get(self.0)
    }
}

impl<M: ValueModel> Runtime<M> {
    pub fn push_root(&mut self, v: u64) -> usize {
        self.shadow.push(v);
        self.shadow.len() - 1
    }
    pub fn pop_root(&mut self) {
        self.shadow.pop();
    }
    pub fn root_get(&self, slot: usize) -> u64 {
        self.shadow[slot]
    }
    pub fn set_root(&mut self, slot: usize, v: u64) {
        self.shadow[slot] = v;
    }
    pub fn root_depth(&self) -> usize {
        self.shadow.len()
    }
    /// Drop all roots at or above `depth` (a non-LIFO bulk `pop_root`).
    pub fn truncate_roots(&mut self, depth: usize) {
        self.shadow.truncate(depth);
    }

    /// Root a value and get a handle back. Caller balances with `pop_root`.
    pub fn root(&mut self, v: u64) -> Root {
        Root(self.push_root(v))
    }

    /// A FULL moving collection — the explicit `(gc)` prim. `live_env` is the
    /// currently executing environment (the safepoint's live frame); its slots
    /// are rewritten in place. STOP-THE-WORLD: every OTHER live mutator is
    /// brought to a safepoint and parked (publishing its roots) before any
    /// object moves.
    ///
    /// Stage I keeps this a MAJOR: `(gc)` has always meant "collect everything
    /// now", which is what makes it usable as a test instrument (a value that
    /// survives it survived a real evacuation) — a minor would quietly promote
    /// the live set and reclaim nothing old.
    pub fn collect(&mut self, live_env: &Locals) {
        self.stw_collect(live_env, None, GcKind::Major);
    }

    /// A moving collection from a `CekMachine` safepoint: additionally roots the
    /// live continuation `live_kont` (its `done` cells and captured frames).
    pub fn collect_cek(&mut self, live_env: &Locals, live_kont: &Arc<Kont>) {
        self.stw_collect(live_env, Some(live_kont), GcKind::Major);
    }

    /// The stop-the-world rendezvous around a collection: request a stop, wait
    /// for all sibling mutators to park at a safepoint (so no one is reading or
    /// mutating the heap), collect, then release them. One collector at a time.
    /// The requesting thread scans its OWN live roots directly; the parked
    /// threads' roots come from their published slots.
    fn stw_collect(&mut self, live_env: &Locals, live_kont: Option<&Arc<Kont>>, kind: GcKind) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
        let shared = self.shared.clone();
        // Claim the collector role with ONE atomic step. If another thread already
        // owns it (gc_requested was already true), we are a sibling mutator that
        // happened to ask for a GC too — just park and participate in theirs. This
        // is what makes concurrent `(gc)` calls safe instead of deadlocking on a
        // lock while un-parked.
        // (A `(gc)` that loses this race participates in the winner's
        // collection, which may be a minor rather than the major it asked for.
        // That has always been the deal — "someone else is collecting, join
        // them" — and `(gc)` is a test/bench instrument driven single-threaded,
        // where it always wins.)
        if shared
            .gc_requested
            .compare_exchange(false, true, AcqRel, Acquire)
            .is_err()
        {
            self.park_with_kont(live_env, live_kont);
            return;
        }
        // Mirror the request into the one-byte poll word the JIT polls and
        // every tier's safepoint checks first (Stage E).
        shared.heap.poll.fetch_or(crate::heap::POLL_REQUESTED, Release);
        // We are the sole collector. Wait until every other mutator is parked
        // (published its roots, stopped touching the heap).
        loop {
            let all_parked = {
                let ms = shared.mutators.lock().unwrap();
                ms.iter()
                    .filter(|m| !Arc::ptr_eq(m, &self.me))
                    .all(|m| m.parked.load(Acquire))
            };
            if all_parked {
                break;
            }
            std::thread::yield_now();
        }
        self.collect_inner(live_env, live_kont, kind);
        // Release the world; parked threads take their rewritten roots back.
        shared.heap.poll.fetch_and(!crate::heap::POLL_REQUESTED, Release);
        shared.gc_requested.store(false, Release);
    }

    /// Await a future, publishing this thread's roots and marking it parked while
    /// blocked on the worker's join — so a collection requested by another thread
    /// (possibly the worker) can proceed instead of deadlocking. The worker stores
    /// its result into the future slot (GC-rooted via the registry), so we read it
    /// back after the join even if a collection relocated it.
    pub fn await_future(&mut self, fut: u64, locals: &Locals) -> u64 {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let slot = match self.view(fut) {
            crate::runtime::ObjView::Future(s) => s,
            _ => panic!("await: not a future"),
        };
        if let Some(r) = slot.lock().unwrap().result {
            return r;
        }
        // Publish roots + the whole env chain (and any native frames below us,
        // Stage E) and mark parked before blocking.
        *self.me.roots.lock().unwrap() = std::mem::take(&mut self.shadow);
        *self.me.envs.lock().unwrap() = self.published_envs(locals);
        *self.me.dyn_roots.lock().unwrap() = std::mem::take(&mut self.dyn_stack);
        *self.me.native_roots.lock().unwrap() = crate::runtime::native_roots_now();
        self.me.parked.store(true, Release);
        let handle = slot.lock().unwrap().handle.take();
        if let Some(h) = handle {
            let _ = h.join();
        }
        // Wait out any in-progress collection before un-parking (our roots stay
        // published so it can complete), then take our (rewritten) roots back.
        while self.shared.gc_requested.load(Acquire) {
            std::thread::yield_now();
        }
        self.shadow = std::mem::take(&mut *self.me.roots.lock().unwrap());
        self.dyn_stack = std::mem::take(&mut *self.me.dyn_roots.lock().unwrap());
        self.me.envs.lock().unwrap().clear();
        self.me.native_roots.lock().unwrap().clear();
        self.me.parked.store(false, Release);
        let r = slot.lock().unwrap().result.expect("future produced no result");
        r
    }

    /// The full set of environments to publish as roots: this thread's dynamic
    /// call chain plus the innermost live env.
    fn published_envs(&self, locals: &Locals) -> Vec<Locals> {
        let mut envs = self.env_stack.clone();
        envs.push(locals.clone());
        envs
    }

    /// Poll a safepoint: one cheap poll-word load; on REQUESTED park until the
    /// sibling's collection finishes, on allocation PRESSURE (Stage E, when
    /// enabled) run a collection ourselves — exactly what the explicit `(gc)`
    /// prim does. Call this where the mutator holds no heap borrow and no
    /// unrooted value bits across it (function entry, call boundaries).
    /// `locals` is published so the collector can trace this thread's env.
    #[inline]
    pub fn safepoint(&mut self, locals: &Locals) {
        use std::sync::atomic::Ordering::Acquire;
        let poll = self.shared.heap.poll.load(Acquire);
        if poll == 0 {
            return;
        }
        self.safepoint_slow(poll, locals, None);
    }

    /// The CEK step loop's safepoint: same poll, but the live continuation
    /// rides along — it is rooted by a self-triggered collection and published
    /// when parking (its `done` cells hold value bits the collector must see).
    #[inline]
    pub fn safepoint_cek(&mut self, locals: &Locals, kont: &Arc<Kont>) {
        use std::sync::atomic::Ordering::Acquire;
        let poll = self.shared.heap.poll.load(Acquire);
        if poll == 0 {
            return;
        }
        self.safepoint_slow(poll, locals, Some(kont));
    }

    #[cold]
    fn safepoint_slow(&mut self, poll: u8, locals: &Locals, kont: Option<&Arc<Kont>>) {
        if poll & crate::heap::POLL_REQUESTED != 0 {
            self.park_with_kont(locals, kont);
            return;
        }
        // PRESSURE: allocation crossed the soft threshold since the last
        // collection. Collect here (this thread claims the collector role; a
        // race with a sibling degenerates into parking for theirs). Auto, not
        // Major — this is the nursery filling up, which is exactly what a minor
        // answers, and answering it with a full evacuation of the accumulated
        // live set is the cost Stage I exists to stop paying.
        if self.shared.pressure_gc {
            self.stw_collect(locals, kont, GcKind::Auto);
        }
    }

    /// Publish this thread's roots and block until the pending collection clears,
    /// then take the (rewritten) roots back. The env's frame slots are rewritten
    /// in place (shared `Arc` frames), so `locals` sees the new addresses without
    /// a copy-back.
    pub fn park(&mut self, locals: &Locals) {
        self.park_with_kont(locals, None);
    }

    pub(crate) fn park_with_kont(&mut self, locals: &Locals, kont: Option<&Arc<Kont>>) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        *self.me.roots.lock().unwrap() = std::mem::take(&mut self.shadow);
        *self.me.envs.lock().unwrap() = self.published_envs(locals);
        *self.me.dyn_roots.lock().unwrap() = std::mem::take(&mut self.dyn_stack);
        *self.me.kont.lock().unwrap() = kont.cloned();
        // Stage E: publish this thread's NATIVE roots — the JIT stack-map spill
        // slots of every native frame below us on the stack (empty without a
        // JIT). The collector rewrites the slots in place; the emitted code
        // reloads them when its call resumes.
        *self.me.native_roots.lock().unwrap() = crate::runtime::native_roots_now();
        self.me.parked.store(true, Release);
        while self.shared.gc_requested.load(Acquire) {
            std::thread::yield_now();
        }
        self.shadow = std::mem::take(&mut *self.me.roots.lock().unwrap());
        self.dyn_stack = std::mem::take(&mut *self.me.dyn_roots.lock().unwrap());
        self.me.envs.lock().unwrap().clear();
        *self.me.kont.lock().unwrap() = None;
        self.me.native_roots.lock().unwrap().clear();
        self.me.parked.store(false, Release);
    }

    /// Enumerate every root slot and run the evacuation (`heap.rs`). Runs with
    /// the world stopped (all sibling mutators parked).
    ///
    /// The root enumeration is the same for a minor and a major — the ONLY
    /// difference is what the heap does with a slot it is handed (promote out
    /// of the nursery, versus evacuate from-space) and that a minor asks the
    /// card table for the old→young edges no root names. So it is written once
    /// and handed to whichever collection(s) run.
    fn collect_inner(&mut self, live_env: &Locals, live_kont: Option<&Arc<Kont>>, kind: GcKind) {
        use std::sync::atomic::Ordering::Relaxed;
        // Serialize against in-place heap mutators (array extend) that might
        // hold the lock right now on THIS thread's behalf — cheap insurance;
        // the real exclusion is the park rendezvous.
        let _g = self.shared.heap_lock.lock().unwrap();
        let shared = self.shared.clone();
        let shadow = &mut self.shadow;
        let dyn_stack = &mut self.dyn_stack;
        let env_stack = &self.env_stack;
        let me = &self.me;
        let signal = &mut self.signal;
        let mut enumerate_roots = |visit: &mut dyn FnMut(*mut u64)| {
            unsafe {
                // 1. Globals: the dense array IS the store; skip unbound slots.
                for a in shared.global_slots.iter() {
                    if a.load(Relaxed) != GLOBAL_UNBOUND {
                        visit(a.as_ptr());
                    }
                }
                // 2. This thread's shadow stack + dynamic-var bindings.
                for s in shadow.iter_mut() {
                    visit(s as *mut u64);
                }
                for (_, v) in dyn_stack.iter_mut() {
                    visit(v as *mut u64);
                }
                // 3. Every OTHER (parked) thread's PUBLISHED roots — including
                //    its live CEK continuation (if it parked from the step
                //    loop) and its NATIVE stack-map slots (Stage E).
                {
                    let ms = shared.mutators.lock().unwrap();
                    for m in ms.iter() {
                        if Arc::ptr_eq(m, me) {
                            continue;
                        }
                        for r in m.roots.lock().unwrap().iter_mut() {
                            visit(r as *mut u64);
                        }
                        for (_, v) in m.dyn_roots.lock().unwrap().iter_mut() {
                            visit(v as *mut u64);
                        }
                        for env in m.envs.lock().unwrap().iter() {
                            visit_env(env, visit);
                        }
                        if let Some(k) = m.kont.lock().unwrap().as_ref() {
                            visit_kont(k, visit);
                        }
                        for &slot in m.native_roots.lock().unwrap().iter() {
                            visit(slot as *mut u64);
                        }
                    }
                }
                // 3b. THIS (collector) thread's own native frames: a JIT poll
                //     or shim triggered this collection, so our stack below
                //     the trigger point holds live stack-map slots too.
                for slot in crate::runtime::native_roots_now() {
                    visit(slot as *mut u64);
                }
                // 4. Constant pool.
                for c in (*shared.consts.get()).iter_mut() {
                    visit(c as *mut u64);
                }
                // 5. Method impls (the dispatch registry is truth) + arglists.
                for imp in shared.tables.lock().unwrap().methods.values_mut() {
                    visit(imp as *mut u64);
                }
                for v in shared.var_arglists.lock().unwrap().values_mut() {
                    visit(v as *mut u64);
                }
                // 6. The `()` singleton.
                if shared.empty_list.load(Relaxed) != 0 {
                    visit(shared.empty_list.as_ptr());
                }
                // 7. Reified continuations (registry = root set; append-only).
                for k in shared.konts.lock().unwrap().iter() {
                    visit_kont(k, visit);
                }
                // 8. Future results (the registry is the OS-resource table).
                for slot in shared.futures.lock().unwrap().iter() {
                    if let Ok(mut s) = slot.lock() {
                        if let Some(r) = s.result.as_mut() {
                            visit(r as *mut u64);
                        }
                    }
                }
                // 9. This (collector) thread's OWN live environments.
                visit_env(live_env, visit);
                for env in env_stack.iter() {
                    visit_env(env, visit);
                }
                // 10. The live continuation, if we are at a CEK safepoint.
                if let Some(k) = live_kont {
                    visit_kont(k, visit);
                }
                // 11. A PENDING signal's payload: the thrown value / the escape's
                //     result. While a throw propagates it is a live heap pointer
                //     that NO other root names, so a safepoint reached with the
                //     signal up would leave it dangling. (`tag` is a plain
                //     counter, not a value — visiting it would hand the collector
                //     a non-pointer.)
                if signal.kind != 0 {
                    visit(&mut signal.value as *mut u64);
                }
            }
        };

        let heap = &shared.heap;
        let types = &shared.types;
        // THE POLICY. A minor is the answer to a full nursery; a major is the
        // answer to a full OLD gen, and the only thing that reclaims old
        // garbage. `minor_will_fit` is a precondition, not a preference:
        // promotion cannot fail half-way through an evacuation, so when the
        // nursery could not fit in what is left of the old space we go straight
        // to a major — which reclaims the old garbage AND the nursery in one
        // pass, rather than promoting survivors twice.
        let minor = match kind {
            GcKind::Auto if heap.minor_will_fit() => {
                Some(unsafe { heap.collect_minor::<M::R>(types, &mut enumerate_roots) })
            }
            _ => None,
        };
        // After a minor the nursery is empty, so a major here only sees old
        // objects — the spec's "minor first, then major if over threshold".
        let needs_major = match minor {
            None => true,
            // gc-stress additionally forces a periodic major so the whole
            // major path stays hammered, not just the minor path.
            Some(out) => {
                out.needs_major
                    || (heap.stress_mode()
                        && heap.minor_collections.load(Relaxed) % STRESS_MAJOR_EVERY == 0)
            }
        };
        if needs_major {
            unsafe { heap.collect::<M::R>(types, &mut enumerate_roots) };
        }

        // Dispatch caches hold impl pointers that just moved: invalidate them
        // (they refill on the next call). The registry, forwarded above, is truth.
        shared.tables.lock().unwrap().dispatch.on_gc();
        // Epoch for per-site ICs: any cached heap pointer is now stale. A MINOR
        // moves objects too — promotion is a copy — so it must bump this exactly
        // like a major does, and one bump covers a minor+major pair because
        // nothing reads the epoch between them (the world is stopped).
        shared.relocated.fetch_add(1, Relaxed);
    }
}

/// Visit an activation frame's root slots: every local plus the running
/// closure's `caps_src`. Idempotent (a re-visited slot forwards to itself), so
/// frames shared with continuations need no visited set.
fn visit_env(env: &Locals, visit: &mut dyn FnMut(*mut u64)) {
    if let Some(f) = env {
        for cell in &f.slots {
            visit(cell.as_ptr());
        }
        visit(f.caps_src.as_ptr());
    }
}

/// Trace a `Kont` chain: every `done`-slot cell and every captured frame.
/// In-place and idempotent (like `visit_env`), so shared continuation tails
/// and multi-shot resumptions need no visited set.
fn visit_kont(k: &Arc<Kont>, visit: &mut dyn FnMut(*mut u64)) {
    let mut cur = k.clone();
    loop {
        let next = match &*cur {
            Kont::Done => return,
            Kont::CallK { done, env, next, .. } | Kont::PrimK { done, env, next, .. } => {
                for cell in done {
                    visit(cell.as_ptr());
                }
                visit_env(env, visit);
                next.clone()
            }
            Kont::If { env, next, .. }
            | Kont::Seq { env, next, .. }
            | Kont::SetLoc { env, next, .. } => {
                visit_env(env, visit);
                next.clone()
            }
            Kont::Def { next, .. }
            | Kont::SetGlob { next, .. }
            | Kont::CallCc { next, .. }
            | Kont::Prompt { next, .. }
            | Kont::ShiftK { next, .. } => next.clone(),
        };
        cur = next;
    }
}
