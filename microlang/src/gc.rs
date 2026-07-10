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
    pub fn collect(&mut self, live_env: &Locals) {
        self.collect_inner(live_env, None);
    }

    /// A moving collection from a `CekMachine` safepoint: additionally roots the
    /// live continuation `live_kont` (its `done` cells and captured frames), so a
    /// collection that happens while a continuation is live or captured relocates
    /// it correctly. This is the moving GC composed with the full-continuation
    /// tier — the combination the 45-way matrix deliberately left out.
    pub fn collect_cek(&mut self, live_env: &Locals, live_kont: &Arc<Kont>) {
        self.collect_inner(live_env, Some(live_kont));
    }

    fn collect_inner(&mut self, live_env: &Locals, live_kont: Option<&Arc<Kont>>) {
        let from_len = self.heap.len();
        let real_before = self
            .heap
            .iter()
            .filter(|o| !matches!(o, Obj::Moved(_)))
            .count();
        let mut to: Vec<Obj> = Vec::new();
        let mut reloc = 0u64;

        {
            let Self {
                heap,
                global_slots,
                shadow,
                consts,
                methods,
                ..
            } = &mut *self;

            // 1. Forward the direct roots. The dense global array IS the global
            //    store; forward every bound slot (skip the unbound sentinel).
            for v in global_slots.iter_mut() {
                if *v != crate::runtime::GLOBAL_UNBOUND {
                    *v = fw::<M>(heap, from_len, &mut to, &mut reloc, *v);
                }
            }
            for s in shadow.iter_mut() {
                *s = fw::<M>(heap, from_len, &mut to, &mut reloc, *s);
            }
            for c in consts.iter_mut() {
                *c = fw::<M>(heap, from_len, &mut to, &mut reloc, *c);
            }
            // Method impls are roots (the dispatch registry is truth).
            for imp in methods.values_mut() {
                *imp = fw::<M>(heap, from_len, &mut to, &mut reloc, *imp);
            }
            // The live environment: rewrite the cells of its frame chain.
            update_env::<M>(heap, from_len, &mut to, &mut reloc, live_env);
            // The live continuation, if we are at a CEK safepoint: forward the
            // `done` cells and captured frames along its whole chain.
            if let Some(k) = live_kont {
                walk_kont::<M>(heap, from_len, &mut to, &mut reloc, k);
            }

            // 2. Cheney scan: forward the internal pointers of copied objects
            //    (which may copy more), until the scan pointer catches up.
            let mut scan = 0;
            while scan < to.len() {
                scan_obj::<M>(heap, from_len, &mut to, &mut reloc, scan);
                scan += 1;
            }

            // 3. Commit to-space; poison remaining from-space so any stale
            //    pointer into it is a loud use-after-move.
            for o in to.drain(..) { heap.push(o); }
            for i in 0..from_len {
                if !matches!(heap[i], Obj::Moved(_)) {
                    heap[i] = Obj::Moved(u32::MAX);
                }
            }
        }

        // Dispatch caches hold impl pointers that just moved: invalidate them
        // (they refill on the next call). The registry, forwarded above, is truth.
        self.dispatch.on_gc();

        self.relocated += reloc;
        self.freed += (real_before as u64).saturating_sub(reloc);
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
