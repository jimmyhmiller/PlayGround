//! A MOVING (semi-space copying) collector, plus the shadow-stack handle
//! discipline. This is where the value and execution axes fully fuse.
//!
//! Reachable objects are copied to fresh addresses; the old slots become
//! `Obj::Moved` forwarding markers. Every root is rewritten to the new address:
//! the globals, the constant pool (which is why `Ir` holds no embedded heap
//! pointers), the shadow stack, and the live environment. Lexical frames stay
//! `Arc`-managed with `Cell` slots, so their heap pointers are rewritten in place
//! and the mutator's `locals` reference stays valid across a collection.
//!
//! The one thing a moving collector cannot fix for free: a bare `u64` the
//! mutator holds directly in a Rust local. After a move it points into
//! from-space (now a `Moved` marker) and dereferencing it is a loud
//! use-after-move. The fix is the handle: publish the value to the shadow stack
//! and re-read it (`root_get`) after anything that may allocate. The compiler
//! (`macroexpand`, `analyze`) does exactly this for the form it is expanding —
//! the direct remedy for the clojure-jvm form-609 relocation bug.
//!
//! Simplification vs. a production semi-space: we append to-space to one growing
//! arena and leave from-space as poisoned `Moved` markers (so stale reads stay
//! loud) instead of flipping two fixed buffers and reusing from-space. The
//! relocation semantics — copy, forward, rewrite roots, re-read via handle — are
//! faithful.

use std::sync::Arc;

use crate::cek::Kont;
use crate::model::{Repr, ValueModel};
use crate::runtime::Runtime;
use crate::value::{slot_load, slot_store, Locals, Obj, RawTag};

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

    /// A moving collection. `live_env` is the currently executing environment
    /// (the safepoint's live frame chain); its cells are rewritten in place.
    /// STOP-THE-WORLD: every OTHER live mutator is brought to a safepoint and
    /// parked (publishing its roots) before any object moves.
    pub fn collect(&mut self, live_env: &Locals) {
        self.stw_collect(live_env, None);
    }

    /// A moving collection from a `CekMachine` safepoint: additionally roots the
    /// live continuation `live_kont` (its `done` cells and captured frames).
    pub fn collect_cek(&mut self, live_env: &Locals, live_kont: &Arc<Kont>) {
        self.stw_collect(live_env, Some(live_kont));
    }

    /// The stop-the-world rendezvous around a collection: request a stop, wait
    /// for all sibling mutators to park at a safepoint (so no one is reading or
    /// mutating the heap), collect, then release them. One collector at a time
    /// (`gc_lock`). The requesting thread scans its OWN live roots directly; the
    /// parked threads' roots come from their published slots.
    fn stw_collect(&mut self, live_env: &Locals, live_kont: Option<&Arc<Kont>>) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
        let shared = self.shared.clone();
        // Claim the collector role with ONE atomic step. If another thread already
        // owns it (gc_requested was already true), we are a sibling mutator that
        // happened to ask for a GC too — just park and participate in theirs. This
        // is what makes concurrent `(gc)` calls safe instead of deadlocking on a
        // lock while un-parked.
        if shared
            .gc_requested
            .compare_exchange(false, true, AcqRel, Acquire)
            .is_err()
        {
            self.park(live_env);
            return;
        }
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
        self.collect_inner(live_env, live_kont);
        // Release the world; parked threads take their rewritten roots back.
        shared.gc_requested.store(false, Release);
    }

    /// Await a future, publishing this thread's roots and marking it parked while
    /// blocked on the worker's join — so a collection requested by another thread
    /// (possibly the worker) can proceed instead of deadlocking. The worker stores
    /// its result into the future slot (GC-rooted via the reachable object), so we
    /// read it back after the join even if a collection relocated it.
    pub fn await_future(&mut self, fut: u64, locals: &Locals) -> u64 {
        use std::sync::atomic::Ordering::{Acquire, Release};
        let id = M::R::as_ref(fut) as usize;
        let slot = match &self.heap()[id] {
            Obj::Future(s) => s.clone(),
            _ => panic!("await: not a future"),
        };
        if let Some(r) = slot.lock().unwrap().result {
            return r;
        }
        // Publish roots + the whole env chain and mark parked before blocking.
        *self.me.roots.lock().unwrap() = std::mem::take(&mut self.shadow);
        *self.me.envs.lock().unwrap() = self.published_envs(locals);
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
        self.me.envs.lock().unwrap().clear();
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

    /// Poll a safepoint: if a collection is pending, park until it finishes. Call
    /// this where the mutator holds no `&Obj` borrow across it (function entry,
    /// alloc). `locals` is published so the collector can trace this thread's env.
    #[inline]
    pub fn safepoint(&mut self, locals: &Locals) {
        use std::sync::atomic::Ordering::Acquire;
        if self.shared.gc_requested.load(Acquire) {
            self.park(locals);
        }
    }

    /// Publish this thread's roots and block until the pending collection clears,
    /// then take the (rewritten) roots back. The env's frame cells are rewritten
    /// in place (shared `Arc` frames), so `locals` sees the new addresses without
    /// a copy-back.
    pub fn park(&mut self, locals: &Locals) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        *self.me.roots.lock().unwrap() = std::mem::take(&mut self.shadow);
        *self.me.envs.lock().unwrap() = self.published_envs(locals);
        self.me.parked.store(true, Release);
        while self.shared.gc_requested.load(Acquire) {
            std::thread::yield_now();
        }
        self.shadow = std::mem::take(&mut *self.me.roots.lock().unwrap());
        self.me.envs.lock().unwrap().clear();
        self.me.parked.store(false, Release);
    }

    fn collect_inner(&mut self, live_env: &Locals, live_kont: Option<&Arc<Kont>>) {
        use std::sync::atomic::Ordering::Relaxed;
        // Serialize heap mutation. (Under the full safepoint protocol this is where
        // all mutators are already parked; here it is the single heap-write lock.)
        let _g = self.shared.heap_lock.lock().unwrap();
        // SAFETY: exclusive heap access is held via `heap_lock`; the resulting
        // `&mut Heap` is raw-pointer-derived, so it does not alias the other
        // `self.shared` field borrows below.
        let heap: &mut crate::runtime::Heap = unsafe { &mut *self.shared.heap.get() };
        let from_len = heap.len();
        let real_before = heap.iter().filter(|o| !matches!(o, Obj::Moved(_))).count();
        let mut to: Vec<Obj> = Vec::new();
        let mut reloc = 0u64;

        // 1. Forward the direct roots. The dense global array IS the global store;
        //    forward every bound slot (skip the unbound sentinel).
        for a in self.shared.global_slots.iter() {
            let v = a.load(Relaxed);
            if v != crate::runtime::GLOBAL_UNBOUND {
                a.store(fw::<M>(heap, from_len, &mut to, &mut reloc, v), Relaxed);
            }
        }
        // This thread's shadow stack (its transient roots).
        for s in self.shadow.iter_mut() {
            *s = fw::<M>(heap, from_len, &mut to, &mut reloc, *s);
        }
        // Every OTHER (parked) thread's PUBLISHED roots: its shadow snapshot
        // (rewritten in place, taken back on resume) and its live env frames.
        {
            let ms = self.shared.mutators.lock().unwrap();
            for m in ms.iter() {
                if Arc::ptr_eq(m, &self.me) {
                    continue;
                }
                let mut roots = m.roots.lock().unwrap();
                for r in roots.iter_mut() {
                    *r = fw::<M>(heap, from_len, &mut to, &mut reloc, *r);
                }
                let envs = m.envs.lock().unwrap();
                for env in envs.iter() {
                    update_env::<M>(heap, from_len, &mut to, &mut reloc, env);
                }
            }
        }
        // Constant pool.
        let consts = unsafe { &mut *self.shared.consts.get() };
        for c in consts.iter_mut() {
            *c = fw::<M>(heap, from_len, &mut to, &mut reloc, *c);
        }
        // Method impls are roots (the dispatch registry is truth).
        {
            let mut t = self.shared.tables.lock().unwrap();
            for imp in t.methods.values_mut() {
                *imp = fw::<M>(heap, from_len, &mut to, &mut reloc, *imp);
            }
        }
        // This (collector) thread's OWN live environments: the innermost env plus
        // every frame in its dynamic call chain.
        update_env::<M>(heap, from_len, &mut to, &mut reloc, live_env);
        for env in self.env_stack.iter() {
            update_env::<M>(heap, from_len, &mut to, &mut reloc, env);
        }
        // The live continuation, if we are at a CEK safepoint.
        if let Some(k) = live_kont {
            walk_kont::<M>(heap, from_len, &mut to, &mut reloc, k);
        }

        // 2. Cheney scan.
        let mut scan = 0;
        while scan < to.len() {
            scan_obj::<M>(heap, from_len, &mut to, &mut reloc, scan);
            scan += 1;
        }

        // 3. Commit to-space; poison remaining from-space.
        for o in to.drain(..) {
            heap.push(o);
        }
        for i in 0..from_len {
            if !matches!(heap[i], Obj::Moved(_)) {
                heap[i] = Obj::Moved(u32::MAX);
            }
        }

        // Dispatch caches hold impl pointers that just moved: invalidate them
        // (they refill on the next call). The registry, forwarded above, is truth.
        self.shared.tables.lock().unwrap().dispatch.on_gc();

        self.shared.relocated.fetch_add(reloc, Relaxed);
        self.shared.freed.fetch_add((real_before as u64).saturating_sub(reloc), Relaxed);
    }
}

/// Copy `bits`'s object to to-space if needed, returning its new ref. Idempotent
/// via `Moved` markers, so shared objects are copied once.
fn fw<M: ValueModel>(
    heap: &mut crate::runtime::Heap,
    from_len: usize,
    to: &mut Vec<Obj>,
    reloc: &mut u64,
    bits: u64,
) -> u64 {
    if M::R::tag_of(bits) != RawTag::Ref {
        return bits;
    }
    let idx = M::R::as_ref(bits) as usize;
    if idx >= from_len {
        return bits; // already in to-space
    }
    if let Obj::Moved(n) = heap[idx] {
        return M::R::enc_ref(n);
    }
    let abs = (from_len + to.len()) as u32;
    let obj = heap[idx].clone();
    heap[idx] = Obj::Moved(abs);
    to.push(obj);
    *reloc += 1;
    M::R::enc_ref(abs)
}

/// Forward the internal pointers of the just-copied object `to[i]`.
fn scan_obj<M: ValueModel>(
    heap: &mut crate::runtime::Heap,
    from_len: usize,
    to: &mut Vec<Obj>,
    reloc: &mut u64,
    i: usize,
) {
    // Cons: forward head and tail (extract first to release the borrow).
    if let Obj::Cons { head, tail } = &to[i] {
        let (h, t) = (*head, *tail);
        let nh = fw::<M>(heap, from_len, to, reloc, h);
        let nt = fw::<M>(heap, from_len, to, reloc, t);
        if let Obj::Cons { head, tail } = &mut to[i] {
            *head = nh;
            *tail = nt;
        }
        return;
    }
    // Closure: rewrite the cells of its captured env chain.
    if let Obj::Closure { env, .. } = &to[i] {
        let env = env.clone();
        update_env::<M>(heap, from_len, to, reloc, &env);
        return;
    }
    // Record: forward each field.
    if let Obj::Record { fields, .. } = &to[i] {
        let fs = fields.clone();
        let forwarded: Vec<u64> = fs
            .into_iter()
            .map(|f| fw::<M>(heap, from_len, to, reloc, f))
            .collect();
        if let Obj::Record { fields, .. } = &mut to[i] {
            *fields = forwarded;
        }
        return;
    }
    // Atom: forward the value it currently holds (atomic load/store; the
    // collector runs stop-the-world, but keep it atomic for tidiness).
    if let Obj::Atom(a) = &to[i] {
        let a = a.clone();
        let v = a.load(std::sync::atomic::Ordering::Relaxed);
        a.store(fw::<M>(heap, from_len, to, reloc, v), std::sync::atomic::Ordering::Relaxed);
        return;
    }
    // Future: forward its cached result (a worker stores the value here before
    // ending, so it stays rooted via the reachable `Future` object).
    if let Obj::Future(slot) = &to[i] {
        let slot = slot.clone();
        let mut g = slot.lock().unwrap();
        if let Some(r) = g.result {
            g.result = Some(fw::<M>(heap, from_len, to, reloc, r));
        }
        return;
    }
    // Vector / multiple-values packet: forward each element.
    if let Obj::Vector(elems) | Obj::Values(elems) = &to[i] {
        let es = elems.clone();
        let forwarded: Vec<u64> = es
            .into_iter()
            .map(|e| fw::<M>(heap, from_len, to, reloc, e))
            .collect();
        if let Obj::Vector(elems) | Obj::Values(elems) = &mut to[i] {
            *elems = forwarded;
        }
        return;
    }
    // A reified continuation (full or delimited): walk its whole `Kont` chain,
    // forwarding the `done` cells and captured frames it holds. `Ir` inside the
    // chain holds no heap pointers (const pool), so only cells and frames move.
    if let Obj::Cont(k) | Obj::PartialCont(k) = &to[i] {
        let k = k.clone();
        walk_kont::<M>(heap, from_len, to, reloc, &k);
    }
}

/// Trace a `Kont` chain: forward every `done`-slot cell in place and rewrite the
/// cells of every captured frame. In-place and idempotent (like `update_env`), so
/// shared continuation tails and multi-shot resumptions need no visited set.
fn walk_kont<M: ValueModel>(
    heap: &mut crate::runtime::Heap,
    from_len: usize,
    to: &mut Vec<Obj>,
    reloc: &mut u64,
    k: &Arc<Kont>,
) {
    let mut cur = k.clone();
    loop {
        let next = match &*cur {
            Kont::Done => return,
            Kont::CallK { done, env, next, .. } | Kont::PrimK { done, env, next, .. } => {
                for cell in done {
                    slot_store(cell, fw::<M>(heap, from_len, to, reloc, slot_load(cell)));
                }
                update_env::<M>(heap, from_len, to, reloc, env);
                next.clone()
            }
            Kont::If { env, next, .. }
            | Kont::Seq { env, next, .. }
            | Kont::SetLoc { env, next, .. } => {
                update_env::<M>(heap, from_len, to, reloc, env);
                next.clone()
            }
            Kont::LetSlot { frame, next, .. } => {
                update_env::<M>(heap, from_len, to, reloc, frame);
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

/// Walk a frame chain, rewriting each `Cell` slot to the forwarded address.
/// Idempotent (a re-visited, already-forwarded slot forwards to itself), so
/// frames shared between closures need no visited set.
fn update_env<M: ValueModel>(
    heap: &mut crate::runtime::Heap,
    from_len: usize,
    to: &mut Vec<Obj>,
    reloc: &mut u64,
    env: &Locals,
) {
    let mut cur = env.clone();
    while let Some(f) = cur {
        for cell in &f.slots {
            slot_store(cell, fw::<M>(heap, from_len, to, reloc, slot_load(cell)));
        }
        cur = f.parent.clone();
    }
}
