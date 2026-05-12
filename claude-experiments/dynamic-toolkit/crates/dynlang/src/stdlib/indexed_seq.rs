//! Growable indexed sequence — the canonical "vector of NanBox values"
//! type that dynamic-language frontends keep reinventing.
//!
//! ## Layout
//! Single Raw64 `len` field followed by a varlen-values backing region
//! (GC-traced). Capacity is fixed at allocation time: every `push`
//! allocates a fresh sequence of size `old_len + 1` and copies. This
//! "copy-on-push" semantics matches beagle's first-cut behavior; a
//! doubling/amortizing growth policy is a future addition (see
//! `docs/toolkit-improvements/07-indexed-seq.md`).
//!
//! ## Usage
//! ```ignore
//! let seq = IndexedSeq::register(&mut dyn_module, "Array");
//!
//! // During lowering:
//! let v = seq.emit_literal(f, &[a, b, c]);    // [a, b, c]
//! let pushed = seq.emit_push(f, src, val);    // push(src, val)
//! let elem = seq.emit_get(f, src, idx);       // src[idx]
//! let n = seq.emit_length_unboxed(f, src);    // length(src) as i64
//!
//! // Host-side reading (e.g. extern thunks):
//! if let Some(view) = seq.view(nanbox_bits) {
//!     let n = view.len();
//!     let elem_bits = view.get(0);
//! }
//! ```

use dynalloc::follow_forwarding;
use dynir::{CmpOp, Type, Value};
use dynvalue::{Decoded, NanBox, TagScheme};

use crate::{DynFunc, DynModule, FieldKind, ObjTypeId};

/// Handle to a registered indexed-sequence object type. Cheap to copy.
/// Owns the layout constants the lowerer + host accessor both need.
#[derive(Clone, Copy, Debug)]
pub struct IndexedSeq {
    type_id: ObjTypeId,
    /// Byte offset of the `len` Raw64 field (relative to the raw header
    /// pointer).
    len_offset: i32,
    /// Byte offset of element 0 in the varlen-values section.
    elem_base: i64,
    /// u16 type id baked into the object header. Used by `view` to
    /// recognize sequences vs other ptr-tagged NanBoxes.
    type_id_u16: u16,
    /// NanBox tag used for heap pointers in this module's tag scheme.
    /// Captured at registration so `view` can decode without consulting
    /// global state.
    ptr_tag: u32,
}

impl IndexedSeq {
    /// Register an indexed-sequence obj type on `dyn_module`. The `name`
    /// is user-visible only in obj-type-table dumps — the language-level
    /// name is whatever the embedder calls it.
    pub fn register(dyn_module: &mut DynModule, name: &str) -> Self {
        let type_id = dyn_module
            .obj_type(name)
            .field("len", FieldKind::Raw64)
            .varlen_values()
            .build();
        let ty = dyn_module.get_obj_type(type_id);
        let len_offset = ty
            .field_offsets
            .get("len")
            .map(|(o, _)| *o)
            .expect("IndexedSeq must have len field");
        let elem_base = ty.type_info.varlen_element_offset(0) as i64;
        let type_id_u16 = ty.type_info.type_id;
        let ptr_tag = dyn_module.tags().ptr;
        IndexedSeq {
            type_id,
            len_offset,
            elem_base,
            type_id_u16,
            ptr_tag,
        }
    }

    /// The object-type id, e.g. for the embedder to compare in custom
    /// type-checks.
    pub fn type_id(&self) -> ObjTypeId {
        self.type_id
    }

    /// Byte offset of the `len` field (host-side reads).
    pub fn len_offset(&self) -> i32 {
        self.len_offset
    }

    // ── Lowering helpers ──────────────────────────────────────────

    /// Lower a literal `[a, b, c, ...]`: alloc + len-store + per-element
    /// store. Each element is GC-rooted across the alloc via
    /// [`DynFunc::with_rooted`]; reloads after the alloc pick up any
    /// forwarding the GC may have done.
    pub fn emit_literal(&self, f: &mut DynFunc, elements: &[Value]) -> Value {
        let layout = *self;
        f.with_rooted(elements, |f, slots| {
            let n = slots.len() as i64;
            let n_const = f.fb.iconst(Type::I64, n);
            f.fb.safepoint(&[]);
            let raw = f.gc_alloc(layout.type_id, n_const);
            f.fb.store(n_const, raw, layout.len_offset);
            for (i, slot) in slots.iter().enumerate() {
                let v = slot.get(f);
                let idx = f.fb.iconst(Type::I64, i as i64);
                store_elem(f, layout.elem_base, raw, idx, v);
            }
            f.obj_wrap(raw)
        })
    }

    /// Lower `push(src, val)`: allocate a new sequence of capacity
    /// `old_len + 1`, copy the elements over, append `val`. Both `src`
    /// and `val` are pinned across the alloc. Result is the new
    /// (NanBox-wrapped) sequence.
    pub fn emit_push(&self, f: &mut DynFunc, src: Value, val: Value) -> Value {
        let layout = *self;
        f.with_rooted(&[src, val], |f, slots| {
            // Read old length BEFORE alloc.
            let src0 = slots[0].get(f);
            let src_raw0 = f.obj_unwrap(src0);
            let old_len = f.fb.load(Type::I64, src_raw0, layout.len_offset);
            let one = f.fb.iconst(Type::I64, 1);
            let new_len = f.fb.add(old_len, one);

            // Allocate the new sequence. After this, src may be
            // forwarded — reload through the slot.
            f.fb.safepoint(&[]);
            let new_raw = f.gc_alloc(layout.type_id, new_len);
            f.fb.store(new_len, new_raw, layout.len_offset);

            let src1 = slots[0].get(f);
            let src_raw1 = f.obj_unwrap(src1);

            // Copy loop: for i in 0..old_len, new[i] = src[i].
            let i_slot = f.fb.create_stack_slot(8, false);
            let zero = f.fb.iconst(Type::I64, 0);
            let i_addr0 = f.fb.stack_addr(i_slot);
            f.fb.store(zero, i_addr0, 0);

            let header_bb = f.fb.create_block(&[]);
            let body_bb = f.fb.create_block(&[]);
            let exit_bb = f.fb.create_block(&[]);
            f.fb.jump(header_bb, &[]);

            f.fb.switch_to_block(header_bb);
            let i_addr_h = f.fb.stack_addr(i_slot);
            let i_h = f.fb.load(Type::I64, i_addr_h, 0);
            let cond = f.fb.icmp(CmpOp::Slt, i_h, old_len);
            f.fb.br_if(cond, body_bb, &[], exit_bb, &[]);

            f.fb.switch_to_block(body_bb);
            let i_addr_b = f.fb.stack_addr(i_slot);
            let i_b = f.fb.load(Type::I64, i_addr_b, 0);
            let elem = load_elem(f, layout.elem_base, src_raw1, i_b);
            store_elem(f, layout.elem_base, new_raw, i_b, elem);
            let i_inc = f.fb.add(i_b, one);
            f.fb.store(i_inc, i_addr_b, 0);
            f.fb.jump(header_bb, &[]);

            f.fb.switch_to_block(exit_bb);
            // Append the new value at index old_len.
            let val_v = slots[1].get(f);
            store_elem(f, layout.elem_base, new_raw, old_len, val_v);

            f.obj_wrap(new_raw)
        })
    }

    /// Lower `seq[idx]` where `idx` is a NanBox-encoded float.
    /// Caller is responsible for ensuring `src` actually wraps a sequence
    /// (no out-of-bounds or type checking — matches beagle's
    /// benchmark-grade contract).
    pub fn emit_get(&self, f: &mut DynFunc, src: Value, idx_box: Value) -> Value {
        let raw = f.obj_unwrap(src);
        let idx_int = f.nanbox_to_int(idx_box);
        load_elem(f, self.elem_base, raw, idx_int)
    }

    /// Lower `length(seq)` — load the `len` field as a raw I64
    /// (unboxed). Caller is responsible for boxing if it's exposed at
    /// the language level as a number value.
    pub fn emit_length_unboxed(&self, f: &mut DynFunc, src: Value) -> Value {
        let raw = f.obj_unwrap(src);
        f.fb.load(Type::I64, raw, self.len_offset)
    }

    // ── Host-side accessor ────────────────────────────────────────

    /// Try to view `bits` as a sequence. Returns `None` if `bits` isn't
    /// a heap-pointer NanBox or the pointed-at object's header doesn't
    /// match this sequence type's id. Walks any forwarding pointer
    /// before reading the type id.
    ///
    /// # Safety
    /// Calling this on bits that *look* like a heap pointer but actually
    /// reference unrelated memory is UB. In practice, NanBox payloads
    /// either come from JIT code (always real heap pointers) or from
    /// untrusted sources that the embedder must filter first.
    pub fn view(&self, bits: u64) -> Option<SeqView> {
        let raw = match NanBox::decode(bits) {
            Decoded::Tagged { tag, payload } if tag == self.ptr_tag => {
                payload as usize as *const u8
            }
            _ => return None,
        };
        let raw = unsafe { follow_forwarding(raw) };
        let header = unsafe { *(raw as *const u64) };
        if (header as u16) != self.type_id_u16 {
            return None;
        }
        Some(SeqView {
            raw,
            len_offset: self.len_offset,
            elem_base: self.elem_base,
        })
    }
}

/// Host-side read-only view of a live sequence. Borrowed from an
/// [`IndexedSeq::view`] call; valid only until the next allocation
/// (which may forward the underlying object).
pub struct SeqView {
    raw: *const u8,
    len_offset: i32,
    elem_base: i64,
}

impl SeqView {
    /// Number of elements.
    pub fn len(&self) -> usize {
        unsafe { *(self.raw.add(self.len_offset as usize) as *const u64) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read element `idx` as a raw NanBox `u64`. Out-of-bounds reads
    /// are UB.
    pub fn get(&self, idx: usize) -> u64 {
        let addr = unsafe { self.raw.add(self.elem_base as usize + idx * 8) };
        unsafe { *(addr as *const u64) }
    }
}

// ── Internal helpers ──────────────────────────────────────────────

fn elem_addr(f: &mut DynFunc, elem_base: i64, raw: Value, idx_int: Value) -> Value {
    let base = f.fb.iconst(Type::I64, elem_base);
    let eight = f.fb.iconst(Type::I64, 8);
    let byte_off = f.fb.mul(idx_int, eight);
    let off = f.fb.add(base, byte_off);
    f.fb.add(raw, off)
}

fn load_elem(f: &mut DynFunc, elem_base: i64, raw: Value, idx_int: Value) -> Value {
    let addr = elem_addr(f, elem_base, raw, idx_int);
    f.fb.load(Type::I64, addr, 0)
}

fn store_elem(f: &mut DynFunc, elem_base: i64, raw: Value, idx_int: Value, val: Value) {
    let addr = elem_addr(f, elem_base, raw, idx_int);
    f.fb.store(val, addr, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GcConfig, NanBoxTags};

    #[test]
    fn register_captures_layout() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let seq = IndexedSeq::register(&mut dyn_module, "Array");
        let ty = dyn_module.get_obj_type(seq.type_id());
        // type_id matches what the obj_types table assigned.
        assert_eq!(seq.type_id_u16, ty.type_info.type_id);
        // elem_base comes from dynobj's varlen_element_offset(0), which
        // accounts for the varlen count word — it sits at
        // `varlen_count_offset + 8`, not `len_offset + 8`.
        assert_eq!(
            seq.elem_base as usize,
            ty.type_info.varlen_element_offset(0)
        );
        // Ptr tag matches the module's NanBoxTags.
        assert_eq!(seq.ptr_tag, dyn_module.tags().ptr);
    }

    #[test]
    fn view_rejects_non_pointer_nanbox() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let seq = IndexedSeq::register(&mut dyn_module, "Array");
        // A raw float NanBox shouldn't match.
        let f_bits = (1.5_f64).to_bits();
        assert!(seq.view(f_bits).is_none());
    }

    // ── End-to-end: lower IR + run via interpreter ────────────────

    use dynalloc::LowBitPtrPolicy;
    use dynir::gc_runtime::GcInterpCtx;
    use dynir::interp::{InterpResult, ModuleInterpreter};
    use dynobj::Compact;

    type TestRoots = GcInterpCtx<Compact, LowBitPtrPolicy<3>>;

    /// Build a function that emits an array literal of three floats and
    /// returns the array's NanBox, then read it back via SeqView.
    #[test]
    fn emit_literal_then_view() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let seq = IndexedSeq::register(&mut dyn_module, "Array");
        let main = dyn_module.declare_func("main", 0);

        let mut f = dyn_module.start_func(main);
        let a = f.number(10.0);
        let b = f.number(20.0);
        let c = f.number(30.0);
        let arr = seq.emit_literal(&mut f, &[a, b, c]);
        f.fb.ret(arr);
        dyn_module.finish_func(f);

        // Need the GC runtime alive for the duration of the SeqView read
        // — the BumpAllocator owns the heap-backed memory.
        let gc = crate::gc::DynGcRuntime::new(
            &GcConfig::generational(64 * 1024),
            &NanBoxTags::default(),
            &dyn_module.obj_types,
        );
        let built = dyn_module.build();

        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name(crate::gc::GC_ALLOC_EXTERN, gc.interp_gc_alloc());

        let _thread = gc.install_thread();
        let bits = match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => v,
            other => panic!("unexpected: {:?}", other),
        };

        let view = seq.view(bits).expect("returned value should be a sequence");
        assert_eq!(view.len(), 3);
        assert_eq!(f64::from_bits(view.get(0)), 10.0);
        assert_eq!(f64::from_bits(view.get(1)), 20.0);
        assert_eq!(f64::from_bits(view.get(2)), 30.0);
    }

    /// Build a function that returns `length(arr)` as a raw I64.
    #[test]
    fn emit_length_unboxed_returns_raw_i64() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let seq = IndexedSeq::register(&mut dyn_module, "Array");
        let main = dyn_module.declare_func("main", 0);

        let mut f = dyn_module.start_func(main);
        let elems: Vec<Value> = (0..5).map(|i| f.number(i as f64)).collect();
        let arr = seq.emit_literal(&mut f, &elems);
        let len = seq.emit_length_unboxed(&mut f, arr);
        f.fb.ret(len);
        dyn_module.finish_func(f);

        let gc = crate::gc::DynGcRuntime::new(
            &GcConfig::generational(64 * 1024),
            &NanBoxTags::default(),
            &dyn_module.obj_types,
        );
        let built = dyn_module.build();

        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name(crate::gc::GC_ALLOC_EXTERN, gc.interp_gc_alloc());

        let _thread = gc.install_thread();
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v as i64, 5),
            other => panic!("unexpected: {:?}", other),
        }
    }

    /// Build a function that does `arr[1]` and returns the element.
    #[test]
    fn emit_get_returns_element_at_index() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let seq = IndexedSeq::register(&mut dyn_module, "Array");
        let main = dyn_module.declare_func("main", 0);

        let mut f = dyn_module.start_func(main);
        let a = f.number(11.0);
        let b = f.number(22.0);
        let c = f.number(33.0);
        let arr = seq.emit_literal(&mut f, &[a, b, c]);
        // Index 1 → 22.0. The index is a NanBox-encoded float.
        let idx = f.number(1.0);
        let elem = seq.emit_get(&mut f, arr, idx);
        f.fb.ret(elem);
        dyn_module.finish_func(f);

        let gc = crate::gc::DynGcRuntime::new(
            &GcConfig::generational(64 * 1024),
            &NanBoxTags::default(),
            &dyn_module.obj_types,
        );
        let built = dyn_module.build();

        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name(crate::gc::GC_ALLOC_EXTERN, gc.interp_gc_alloc());

        let _thread = gc.install_thread();
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(f64::from_bits(v), 22.0),
            other => panic!("unexpected: {:?}", other),
        }
    }

    /// Build a function that pushes onto a 2-element array and returns
    /// `length(pushed)` — should be 3.
    #[test]
    fn emit_push_extends_by_one() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let seq = IndexedSeq::register(&mut dyn_module, "Array");
        let main = dyn_module.declare_func("main", 0);

        let mut f = dyn_module.start_func(main);
        let a = f.number(7.0);
        let b = f.number(8.0);
        let arr = seq.emit_literal(&mut f, &[a, b]);
        let v = f.number(9.0);
        let pushed = seq.emit_push(&mut f, arr, v);
        let len = seq.emit_length_unboxed(&mut f, pushed);
        f.fb.ret(len);
        dyn_module.finish_func(f);

        let gc = crate::gc::DynGcRuntime::new(
            &GcConfig::generational(64 * 1024),
            &NanBoxTags::default(),
            &dyn_module.obj_types,
        );
        let built = dyn_module.build();

        let roots: TestRoots = GcInterpCtx::new_unallocating();
        let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        interp.bind_by_name(crate::gc::GC_ALLOC_EXTERN, gc.interp_gc_alloc());

        let _thread = gc.install_thread();
        match interp.run(main, &[]) {
            Ok(InterpResult::Value(v)) => assert_eq!(v as i64, 3),
            other => panic!("unexpected: {:?}", other),
        }
    }
}
