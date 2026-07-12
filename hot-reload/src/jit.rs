//! LLVM `step` backend — the native half of `RUNTIME_DESIGN.md`.
//!
//! Each Ready function version is JIT-compiled to a
//! `step(frame*, runtime*) -> outcome` function. Native code runs the pure ops
//! and the non-pausing runtime calls, then *hands back* to the Rust driver for
//! everything that owns continuation semantics: pushing a frame (`Call`),
//! ending one (`Return`), pausing (`Condition`), or reaching a recurring safe
//! point (`Yield`). LLVM therefore accelerates execution without owning
//! pause/resume — exactly the design's split.
//!
//! Two properties make this sound and simple:
//!   * **No SSA lives across a basic block.** Every register read loads from
//!     `frame->regs[i]`; every write stores back. Loops and branches need no
//!     phi nodes, and — the design's key claim — every live reference sits in a
//!     typed frame slot, so the GC has a complete precise root map.
//!   * **Recompile per run.** Each drive rebuilds the module/engine for the
//!     current world, so repaired code is simply the next compile; JIT frame
//!     register arrays live in [`JitActor`] and outlive any `Context`.

use crate::*;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::types::StructType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// C-ABI value / frame representation shared verbatim between Rust and LLVM.
// ---------------------------------------------------------------------------

pub const TAG_EMPTY: i64 = 0;
pub const TAG_UNIT: i64 = 1;
pub const TAG_I64: i64 = 2;
pub const TAG_BOOL: i64 = 3;
pub const TAG_REF: i64 = 4;
pub const TAG_FOREIGN: i64 = 5;
/// Low-byte mask for the tag. A `Foreign` slot carries its (small `u32`) kind in
/// the tag's high bits and its native pointer in the payload, so a two-word slot
/// still represents every value — no wider frame layout needed. Only `Foreign`
/// uses the high bits; every other tag is a bare low value, so the exact-`==`
/// tag guards the codegen emits for arithmetic/branch operands are unaffected.
pub const TAG_KIND_SHIFT: i64 = 8;
pub const TAG_MASK: i64 = 0xff;

/// One typed register slot. For `TAG_REF` the payload is the [`ObjectId`] — a
/// GC root the collector reads directly out of the frame. For `TAG_FOREIGN` the
/// payload is the native pointer and the kind is in `tag >> TAG_KIND_SHIFT`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawSlot {
    pub tag: i64,
    pub payload: i64,
}

impl RawSlot {
    pub const EMPTY: RawSlot = RawSlot {
        tag: TAG_EMPTY,
        payload: 0,
    };

    pub fn from_value(value: &Value) -> RawSlot {
        match value {
            Value::Unit => RawSlot {
                tag: TAG_UNIT,
                payload: 0,
            },
            Value::I64(n) => RawSlot {
                tag: TAG_I64,
                payload: *n,
            },
            Value::Bool(b) => RawSlot {
                tag: TAG_BOOL,
                payload: *b as i64,
            },
            Value::Ref(id) => RawSlot {
                tag: TAG_REF,
                payload: *id as i64,
            },
            Value::Foreign { kind, ptr } => RawSlot {
                tag: TAG_FOREIGN | ((*kind as i64) << TAG_KIND_SHIFT),
                payload: *ptr as i64,
            },
        }
    }

    pub fn to_value(self) -> Value {
        match self.tag & TAG_MASK {
            TAG_UNIT => Value::Unit,
            TAG_I64 => Value::I64(self.payload),
            TAG_BOOL => Value::Bool(self.payload != 0),
            TAG_REF => Value::Ref(self.payload as ObjectId),
            TAG_FOREIGN => Value::Foreign {
                kind: (self.tag >> TAG_KIND_SHIFT) as u32,
                ptr: self.payload as u64,
            },
            other => panic!("empty or unknown slot tag {other} escaped a step boundary"),
        }
    }
}

/// The heap-resident frame native code operates on. Its LLVM struct layout
/// (see [`frame_type`]) matches this `#[repr(C)]` exactly.
#[repr(C)]
pub struct RawFrame {
    pub func_id: i64,
    pub version: i64,
    pub pc: i64,
    pub n_regs: i64,
    pub regs: *mut RawSlot,
    pub scratch: RawSlot,
    pub return_reg: i64,
}

/// A constructor field passed to `lt_new` — laid out as three consecutive
/// `i64`s (`field_id`, then the slot's `tag`/`payload`), which is what the
/// codegen writes into its stack array.
#[repr(C)]
pub struct SuppliedField {
    pub field_id: i64,
    pub value: RawSlot,
}

// Native `step` outcomes (the function's `i64` return).
pub const OUT_RETURN: i64 = 0;
pub const OUT_CALL: i64 = 1;
pub const OUT_CONDITION: i64 = 2;
pub const OUT_YIELD: i64 = 3;
/// An operand-tag check failed (`SubI64`/`LtI64`/`Branch` saw a value of the
/// wrong representation — the con-freeness trap). The driver reconstructs the
/// exact condition from the instruction at `frame->pc` so it matches the
/// interpreter's.
pub const OUT_TYPE_ERROR: i64 = 4;

// ---------------------------------------------------------------------------
// The JIT host. The runtime side of a JIT step — allocation, migration, and
// effects — behind one type so there is a single set of externs for both
// executors: `Single` wraps the single-threaded `&mut Runtime`, `Concurrent`
// wraps the thread-safe `&Shared` used by JIT worker threads. Each is a thin
// bridge to the *same* `Heap`/`World` operations the interpreter uses, so the
// JIT cannot diverge on them. A trapped condition is stashed per call (per
// thread) for the driver to pick up on an `OUT_CONDITION`.
// ---------------------------------------------------------------------------

pub enum JitHost<'a> {
    Single { rt: &'a mut Runtime, pending: Option<Condition> },
    Concurrent { shared: &'a Shared, pending: Option<Condition> },
}

impl<'a> JitHost<'a> {
    fn single(rt: &'a mut Runtime) -> JitHost<'a> {
        JitHost::Single { rt, pending: None }
    }
    fn concurrent(shared: &'a Shared) -> JitHost<'a> {
        JitHost::Concurrent { shared, pending: None }
    }
    fn new_object(&mut self, type_id: DefId, supplied: &[(FieldId, Value)]) -> Result<ObjectId, Condition> {
        match self {
            JitHost::Single { rt, .. } => rt.jit_new(type_id, supplied),
            JitHost::Concurrent { shared, .. } => shared.jit_new(type_id, supplied),
        }
    }
    fn get_field(&mut self, id: ObjectId, field: FieldId) -> Result<Value, Condition> {
        match self {
            JitHost::Single { rt, .. } => rt.jit_get_field(id, field),
            JitHost::Concurrent { shared, .. } => shared.jit_get_field(id, field),
        }
    }
    fn emit(&mut self, value: Value) {
        match self {
            JitHost::Single { rt, .. } => rt.jit_emit(value),
            JitHost::Concurrent { shared, .. } => shared.jit_emit(value),
        }
    }
    fn call_foreign(&mut self, foreign: ForeignFnId, args: &[Value]) -> Result<Value, Condition> {
        match self {
            JitHost::Single { rt, .. } => rt.jit_call_foreign(foreign, args),
            JitHost::Concurrent { shared, .. } => shared.jit_call_foreign(foreign, args),
        }
    }
    fn load_global(&mut self, id: DefId) -> Result<Value, Condition> {
        match self {
            JitHost::Single { rt, .. } => rt.jit_load_global(id),
            JitHost::Concurrent { shared, .. } => shared.jit_load_global(id),
        }
    }
    fn set_pending(&mut self, condition: Condition) {
        match self {
            JitHost::Single { pending, .. } | JitHost::Concurrent { pending, .. } => {
                *pending = Some(condition)
            }
        }
    }
    fn take_pending(&mut self) -> Option<Condition> {
        match self {
            JitHost::Single { pending, .. } | JitHost::Concurrent { pending, .. } => pending.take(),
        }
    }
}

/// Returns 0 on success (writes `*out_objid`), 1 when construction trips the
/// soundness check (the condition is stashed in the host).
///
/// # Safety
/// `host` is a live `*mut JitHost`, `fields` points to `n` `SuppliedField`s,
/// `out_objid` is a writable `*mut i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_new(
    host: *mut JitHost,
    type_id: i64,
    fields: *const SuppliedField,
    n: i64,
    out_objid: *mut i64,
) -> i64 {
    let host = unsafe { &mut *host };
    let mut supplied = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        let f = unsafe { &*fields.offset(i) };
        supplied.push((f.field_id as FieldId, f.value.to_value()));
    }
    match host.new_object(type_id as DefId, &supplied) {
        Ok(id) => {
            unsafe { *out_objid = id as i64 };
            0
        }
        Err(condition) => {
            host.set_pending(condition);
            1
        }
    }
}

/// Returns 0 on success (writes `*out`), 1 when a migration barrier trips (the
/// condition is stashed in the host for the driver).
///
/// # Safety
/// `host` is a live `*mut JitHost`, `out` a writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_get_field(
    host: *mut JitHost,
    objid: i64,
    field: i64,
    out: *mut RawSlot,
) -> i64 {
    let host = unsafe { &mut *host };
    match host.get_field(objid as ObjectId, field as FieldId) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            host.set_pending(condition);
            1
        }
    }
}

/// # Safety
/// `host` is a live `*mut JitHost`, `value` a readable `*const RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_emit(host: *mut JitHost, value: *const RawSlot) {
    let host = unsafe { &mut *host };
    host.emit(unsafe { *value }.to_value());
}

/// Returns 0 on success (writes the result to `*out`), 1 when the call traps
/// (unregistered fn, or a native return that fails the type check) — the
/// condition is stashed in the host.
///
/// # Safety
/// `host` is a live `*mut JitHost`, `args` points to `n` `RawSlot`s, `out` is a
/// writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_call_foreign(
    host: *mut JitHost,
    foreign: i64,
    args: *const RawSlot,
    n: i64,
    out: *mut RawSlot,
) -> i64 {
    let host = unsafe { &mut *host };
    let mut values = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        values.push(unsafe { *args.offset(i) }.to_value());
    }
    match host.call_foreign(foreign as ForeignFnId, &values) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            host.set_pending(condition);
            1
        }
    }
}

/// Returns 0 on success (writes `*out`), 1 when the global is unset (condition
/// stashed in the host).
///
/// # Safety
/// `host` is a live `*mut JitHost`, `out` a writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_load_global(host: *mut JitHost, global: i64, out: *mut RawSlot) -> i64 {
    let host = unsafe { &mut *host };
    match host.load_global(global as DefId) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            host.set_pending(condition);
            1
        }
    }
}

// ---------------------------------------------------------------------------
// The compiler.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JitError(pub String);

/// A compiled world: the execution engine, the module it owns, and the
/// resolved address of every `step_<id>_<ver>`. The engine backs the JIT code
/// memory the addresses point at, so it must outlive every step call; the
/// module must outlive the engine. Field order matters — Rust drops fields top
/// to bottom, so `engine` (which references the module's code) is torn down
/// before `module`. Dropping the module first was a use-after-free that only
/// crashed under release optimization.
pub(crate) struct Compiled<'ctx> {
    _engine: ExecutionEngine<'ctx>,
    _module: Module<'ctx>,
    addrs: HashMap<(DefId, Version), usize>,
}

type StepFn = unsafe extern "C" fn(*mut RawFrame, *mut JitHost) -> i64;

impl<'ctx> Compiled<'ctx> {
    fn step_of(&self, func: DefId, version: Version) -> Option<StepFn> {
        self.addrs
            .get(&(func, version))
            .map(|addr| unsafe { std::mem::transmute::<usize, StepFn>(*addr) })
    }

    /// A `Send` copy of the compiled addresses, for handing to worker threads.
    /// The addresses are raw code pointers — valid to call from any thread as
    /// long as this `Compiled` (which owns the engine backing them) outlives the
    /// threads. The concurrent runner guarantees that by joining before drop.
    fn addr_map(&self) -> HashMap<(DefId, Version), usize> {
        self.addrs.clone()
    }
}

/// Transmute a compiled step address to a callable. See [`Compiled::addr_map`]
/// for the lifetime guarantee that makes this safe.
fn step_at(addr: usize) -> StepFn {
    unsafe { std::mem::transmute::<usize, StepFn>(addr) }
}

struct Codegen<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
}

impl<'ctx> Codegen<'ctx> {
    fn slot_type(&self) -> StructType<'ctx> {
        let i64t = self.ctx.i64_type();
        self.ctx.struct_type(&[i64t.into(), i64t.into()], false)
    }

    /// `{ i64 func_id, version, pc, n_regs; ptr regs; i64 scratch_tag,
    /// scratch_payload, return_reg }` — byte-identical to [`RawFrame`] (the
    /// embedded `scratch: RawSlot` flattens to fields 5 and 6).
    fn frame_type(&self) -> StructType<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        self.ctx.struct_type(
            &[
                i64t.into(), // 0 func_id
                i64t.into(), // 1 version
                i64t.into(), // 2 pc
                i64t.into(), // 3 n_regs
                ptr.into(),  // 4 regs
                i64t.into(), // 5 scratch.tag
                i64t.into(), // 6 scratch.payload
                i64t.into(), // 7 return_reg
            ],
            false,
        )
    }

    fn declare_externs(&self) {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let void = self.ctx.void_type();
        self.module.add_function(
            "lt_new",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), ptr.into(), i64t.into(), ptr.into()],
                false,
            ),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_get_field",
            i64t.fn_type(&[ptr.into(), i64t.into(), i64t.into(), ptr.into()], false),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_emit",
            void.fn_type(&[ptr.into(), ptr.into()], false),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_call_foreign",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), ptr.into(), i64t.into(), ptr.into()],
                false,
            ),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_load_global",
            i64t.fn_type(&[ptr.into(), i64t.into(), ptr.into()], false),
            Some(Linkage::External),
        );
    }

    fn declare_step(&self, func: &Function) -> FunctionValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let fn_ty = i64t.fn_type(&[ptr.into(), ptr.into()], false);
        self.module
            .add_function(&step_name(func.id, func.version), fn_ty, None)
    }

    /// Pointer to register slot `i` within `frame->regs`.
    fn slot_ptr(&self, frame: PointerValue<'ctx>, i: usize) -> PointerValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let frame_ty = self.frame_type();
        let slot_ty = self.slot_type();
        let regs_field = self
            .builder
            .build_struct_gep(frame_ty, frame, 4, "regs.f")
            .unwrap();
        let regs = self
            .builder
            .build_load(ptr, regs_field, "regs")
            .unwrap()
            .into_pointer_value();
        unsafe {
            self.builder
                .build_in_bounds_gep(slot_ty, regs, &[i64t.const_int(i as u64, false)], "slot")
                .unwrap()
        }
    }

    fn load_field(&self, slot: PointerValue<'ctx>, idx: u32, name: &str) -> IntValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let slot_ty = self.slot_type();
        let field = self.builder.build_struct_gep(slot_ty, slot, idx, name).unwrap();
        self.builder
            .build_load(i64t, field, name)
            .unwrap()
            .into_int_value()
    }

    fn store_slot(&self, frame: PointerValue<'ctx>, i: usize, tag: IntValue<'ctx>, payload: IntValue<'ctx>) {
        let slot_ty = self.slot_type();
        let slot = self.slot_ptr(frame, i);
        let tag_f = self.builder.build_struct_gep(slot_ty, slot, 0, "tag.f").unwrap();
        self.builder.build_store(tag_f, tag).unwrap();
        let pay_f = self.builder.build_struct_gep(slot_ty, slot, 1, "pay.f").unwrap();
        self.builder.build_store(pay_f, payload).unwrap();
    }

    /// Read register `i`'s payload — used for operands whose tag has already
    /// been guarded by [`Codegen::guard_tags`].
    fn payload_of(&self, frame: PointerValue<'ctx>, i: usize) -> IntValue<'ctx> {
        let slot = self.slot_ptr(frame, i);
        self.load_field(slot, 1, "payload")
    }

    fn tag_of(&self, frame: PointerValue<'ctx>, i: usize) -> IntValue<'ctx> {
        let slot = self.slot_ptr(frame, i);
        self.load_field(slot, 0, "tag")
    }

    /// Guard that each `(reg, expected_tag)` holds, before the current
    /// instruction consumes those operands. On any mismatch, branch to a block
    /// that parks `frame->pc` at this instruction and returns `OUT_TYPE_ERROR`
    /// — the native con-freeness trap. On success the builder is left
    /// positioned in a fresh "checks passed" block. Without this, native
    /// arithmetic would read a migrated `Ref`'s object id as an integer and
    /// silently diverge from the interpreter (which tag-checks its operands).
    fn guard_tags(
        &self,
        step: FunctionValue<'ctx>,
        frame: PointerValue<'ctx>,
        pc: usize,
        checks: &[(usize, i64)],
    ) {
        let i64t = self.ctx.i64_type();
        let err = self.ctx.append_basic_block(step, &format!("pc{pc}.badtag"));
        for (n, (reg, expected)) in checks.iter().enumerate() {
            let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.tagok{n}"));
            let tag = self.tag_of(frame, *reg);
            let good = self
                .builder
                .build_int_compare(IntPredicate::EQ, tag, i64t.const_int(*expected as u64, false), "tagok")
                .unwrap();
            self.builder.build_conditional_branch(good, ok, err).unwrap();
            self.builder.position_at_end(ok);
        }
        // Emit the error block after wiring all checks so it dominates nothing
        // on the success path. Save/restore the builder position.
        let resume = step.get_last_basic_block().unwrap();
        self.builder.position_at_end(err);
        self.set_pc(frame, pc as u64);
        self.ret_outcome(OUT_TYPE_ERROR);
        self.builder.position_at_end(resume);
    }

    fn set_pc(&self, frame: PointerValue<'ctx>, pc: u64) {
        let i64t = self.ctx.i64_type();
        let frame_ty = self.frame_type();
        let pc_f = self.builder.build_struct_gep(frame_ty, frame, 2, "pc.f").unwrap();
        self.builder.build_store(pc_f, i64t.const_int(pc, false)).unwrap();
    }

    fn ret_outcome(&self, outcome: i64) {
        let i64t = self.ctx.i64_type();
        self.builder
            .build_return(Some(&i64t.const_int(outcome as u64, false)))
            .unwrap();
    }

    fn define_step(&self, func: &Function, step: FunctionValue<'ctx>) {
        let i64t = self.ctx.i64_type();
        let frame_ty = self.frame_type();
        let frame = step.get_nth_param(0).unwrap().into_pointer_value();
        let rt = step.get_nth_param(1).unwrap().into_pointer_value();

        // One LLVM block per IR pc, created up front so branches/back-edges can
        // target any of them, plus a dispatch block that resumes at frame->pc.
        let entry = self.ctx.append_basic_block(step, "dispatch");
        let blocks: Vec<_> = (0..func.code.len())
            .map(|pc| self.ctx.append_basic_block(step, &format!("pc{pc}")))
            .collect();
        let trap = self.ctx.append_basic_block(step, "trap");

        self.builder.position_at_end(entry);
        let pc_f = self.builder.build_struct_gep(frame_ty, frame, 2, "pc.f").unwrap();
        let pc = self.builder.build_load(i64t, pc_f, "pc").unwrap().into_int_value();
        let cases: Vec<_> = blocks
            .iter()
            .enumerate()
            .map(|(pc, bb)| (i64t.const_int(pc as u64, false), *bb))
            .collect();
        self.builder.build_switch(pc, trap, &cases).unwrap();
        self.builder.position_at_end(trap);
        self.builder.build_unreachable().unwrap();

        for (pc, instruction) in func.code.iter().enumerate() {
            self.builder.position_at_end(blocks[pc]);
            let fallthrough = || {
                if pc + 1 < blocks.len() {
                    self.builder.build_unconditional_branch(blocks[pc + 1]).unwrap();
                }
            };
            match instruction {
                Instruction::Const { dst, value } => {
                    let slot = RawSlot::from_value(value);
                    self.store_slot(
                        frame,
                        *dst,
                        i64t.const_int(slot.tag as u64, false),
                        i64t.const_int(slot.payload as u64, true),
                    );
                    fallthrough();
                }
                Instruction::Copy { dst, src } => {
                    // Copy the whole slot (tag + payload) — works for any type,
                    // and the GC sees a copied Ref in dst.
                    let s = self.slot_ptr(frame, *src);
                    let tag = self.load_field(s, 0, "cp.tag");
                    let payload = self.load_field(s, 1, "cp.pay");
                    self.store_slot(frame, *dst, tag, payload);
                    fallthrough();
                }
                Instruction::AddI64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_I64), (*right, TAG_I64)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let r = self.builder.build_int_add(a, b, "add").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_I64 as u64, false), r);
                    fallthrough();
                }
                Instruction::SubI64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_I64), (*right, TAG_I64)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let r = self.builder.build_int_sub(a, b, "sub").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_I64 as u64, false), r);
                    fallthrough();
                }
                Instruction::MulI64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_I64), (*right, TAG_I64)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let r = self.builder.build_int_mul(a, b, "mul").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_I64 as u64, false), r);
                    fallthrough();
                }
                Instruction::LtI64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_I64), (*right, TAG_I64)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let lt = self
                        .builder
                        .build_int_compare(IntPredicate::SLT, a, b, "lt")
                        .unwrap();
                    let r = self.builder.build_int_z_extend(lt, i64t, "ltz").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_BOOL as u64, false), r);
                    fallthrough();
                }
                Instruction::EqI64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_I64), (*right, TAG_I64)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let eq = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, a, b, "eq")
                        .unwrap();
                    let r = self.builder.build_int_z_extend(eq, i64t, "eqz").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_BOOL as u64, false), r);
                    fallthrough();
                }
                Instruction::Not { dst, src } => {
                    self.guard_tags(step, frame, pc, &[(*src, TAG_BOOL)]);
                    // Bool payload is 0/1; flip it.
                    let v = self.payload_of(frame, *src);
                    let one = i64t.const_int(1, false);
                    let r = self.builder.build_xor(v, one, "not").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_BOOL as u64, false), r);
                    fallthrough();
                }
                Instruction::New {
                    dst,
                    type_id,
                    fields,
                } => {
                    // Marshal supplied (field_id, slot) triples into a stack
                    // array laid out as `SuppliedField[n]`, then call lt_new.
                    let n = fields.len();
                    let arr_ty = i64t.array_type(3 * n as u32);
                    let arr = self.builder.build_alloca(arr_ty, "supplied").unwrap();
                    for (k, (fid, reg)) in fields.iter().enumerate() {
                        let slot = self.slot_ptr(frame, *reg);
                        let tag = self.load_field(slot, 0, "s.tag");
                        let payload = self.load_field(slot, 1, "s.pay");
                        let store = |off: usize, v: IntValue<'ctx>| {
                            let elem = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        arr_ty,
                                        arr,
                                        &[i64t.const_zero(), i64t.const_int((3 * k + off) as u64, false)],
                                        "field",
                                    )
                                    .unwrap()
                            };
                            self.builder.build_store(elem, v).unwrap();
                        };
                        store(0, i64t.const_int(*fid, false));
                        store(1, tag);
                        store(2, payload);
                    }
                    let base = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                arr_ty,
                                arr,
                                &[i64t.const_zero(), i64t.const_zero()],
                                "supplied.base",
                            )
                            .unwrap()
                    };
                    let out_objid = self.builder.build_alloca(i64t, "out.objid").unwrap();
                    let lt_new = self.module.get_function("lt_new").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_new,
                            &[
                                rt.into(),
                                i64t.const_int(*type_id, false).into(),
                                base.into(),
                                i64t.const_int(n as u64, false).into(),
                                out_objid.into(),
                            ],
                            "status",
                        )
                        .unwrap();
                    let status = call_result(call).into_int_value();
                    // status != 0 → construction trapped the soundness check.
                    let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.newok"));
                    let bad = self.ctx.append_basic_block(step, &format!("pc{pc}.newbad"));
                    let is_ok = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, status, i64t.const_zero(), "newisok")
                        .unwrap();
                    self.builder.build_conditional_branch(is_ok, ok, bad).unwrap();
                    self.builder.position_at_end(bad);
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CONDITION);
                    self.builder.position_at_end(ok);
                    let objid = self.builder.build_load(i64t, out_objid, "objid").unwrap().into_int_value();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_REF as u64, false), objid);
                    if pc + 1 < blocks.len() {
                        self.builder.build_unconditional_branch(blocks[pc + 1]).unwrap();
                    }
                }
                Instruction::GetField { dst, object, field } => {
                    let objid = self.payload_of(frame, *object);
                    let out = self.slot_ptr(frame, *dst);
                    let lt_get = self.module.get_function("lt_get_field").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_get,
                            &[
                                rt.into(),
                                objid.into(),
                                i64t.const_int(*field, false).into(),
                                out.into(),
                            ],
                            "status",
                        )
                        .unwrap();
                    let status = call_result(call).into_int_value();
                    // status != 0 → migration barrier: retry this pc on resume.
                    let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.ok"));
                    let need = self.ctx.append_basic_block(step, &format!("pc{pc}.cond"));
                    let is_ok = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, status, i64t.const_zero(), "isok")
                        .unwrap();
                    self.builder.build_conditional_branch(is_ok, ok, need).unwrap();
                    self.builder.position_at_end(need);
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CONDITION);
                    self.builder.position_at_end(ok);
                    if pc + 1 < blocks.len() {
                        self.builder.build_unconditional_branch(blocks[pc + 1]).unwrap();
                    }
                }
                Instruction::Emit { value } => {
                    let slot = self.slot_ptr(frame, *value);
                    let lt_emit = self.module.get_function("lt_emit").unwrap();
                    self.builder
                        .build_call(lt_emit, &[rt.into(), slot.into()], "")
                        .unwrap();
                    fallthrough();
                }
                Instruction::Send { .. } | Instruction::Recv { .. } => {
                    // Message passing belongs to the concurrent tier; the JIT
                    // never runs programs containing it. Trap if one appears.
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_TYPE_ERROR);
                }
                Instruction::CallForeign { dst, foreign, args } => {
                    // Marshal argument slots into a stack `RawSlot[n]` (flat
                    // `i64[2n]`), then call the foreign extern, which writes the
                    // result into `dst`'s slot or stashes a trap condition.
                    let n = args.len();
                    let arr_ty = i64t.array_type(2 * n as u32);
                    let arr = self.builder.build_alloca(arr_ty, "cf.args").unwrap();
                    for (k, reg) in args.iter().enumerate() {
                        let slot = self.slot_ptr(frame, *reg);
                        let tag = self.load_field(slot, 0, "a.tag");
                        let payload = self.load_field(slot, 1, "a.pay");
                        let store = |off: usize, v: IntValue<'ctx>| {
                            let elem = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        arr_ty,
                                        arr,
                                        &[i64t.const_zero(), i64t.const_int((2 * k + off) as u64, false)],
                                        "arg",
                                    )
                                    .unwrap()
                            };
                            self.builder.build_store(elem, v).unwrap();
                        };
                        store(0, tag);
                        store(1, payload);
                    }
                    let base = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                arr_ty,
                                arr,
                                &[i64t.const_zero(), i64t.const_zero()],
                                "cf.args.base",
                            )
                            .unwrap()
                    };
                    let out = self.slot_ptr(frame, *dst);
                    let lt_cf = self.module.get_function("lt_call_foreign").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_cf,
                            &[
                                rt.into(),
                                i64t.const_int(*foreign as u64, false).into(),
                                base.into(),
                                i64t.const_int(n as u64, false).into(),
                                out.into(),
                            ],
                            "status",
                        )
                        .unwrap();
                    let status = call_result(call).into_int_value();
                    let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.cfok"));
                    let bad = self.ctx.append_basic_block(step, &format!("pc{pc}.cfbad"));
                    let is_ok = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, status, i64t.const_zero(), "cfisok")
                        .unwrap();
                    self.builder.build_conditional_branch(is_ok, ok, bad).unwrap();
                    self.builder.position_at_end(bad);
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CONDITION);
                    self.builder.position_at_end(ok);
                    fallthrough();
                }
                Instruction::LoadGlobal { dst, global } => {
                    let out = self.slot_ptr(frame, *dst);
                    let lt_lg = self.module.get_function("lt_load_global").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_lg,
                            &[rt.into(), i64t.const_int(*global, false).into(), out.into()],
                            "status",
                        )
                        .unwrap();
                    let status = call_result(call).into_int_value();
                    let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.lgok"));
                    let bad = self.ctx.append_basic_block(step, &format!("pc{pc}.lgbad"));
                    let is_ok = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, status, i64t.const_zero(), "lgisok")
                        .unwrap();
                    self.builder.build_conditional_branch(is_ok, ok, bad).unwrap();
                    self.builder.position_at_end(bad);
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CONDITION);
                    self.builder.position_at_end(ok);
                    fallthrough();
                }
                Instruction::Jump { target } => {
                    self.builder.build_unconditional_branch(blocks[*target]).unwrap();
                }
                Instruction::Branch {
                    cond,
                    then_pc,
                    else_pc,
                } => {
                    self.guard_tags(step, frame, pc, &[(*cond, TAG_BOOL)]);
                    let c = self.payload_of(frame, *cond);
                    let taken = self
                        .builder
                        .build_int_compare(IntPredicate::NE, c, i64t.const_zero(), "taken")
                        .unwrap();
                    self.builder
                        .build_conditional_branch(taken, blocks[*then_pc], blocks[*else_pc])
                        .unwrap();
                }
                Instruction::Yield => {
                    self.set_pc(frame, (pc + 1) as u64);
                    self.ret_outcome(OUT_YIELD);
                }
                Instruction::Call { .. } => {
                    // Hand back so the driver pushes the frame; leave pc here so
                    // it re-reads this Call (needed if the callee is broken and
                    // we pause, then resume after a repair).
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CALL);
                }
                Instruction::Return { value } => {
                    let slot = self.slot_ptr(frame, *value);
                    let tag = self.load_field(slot, 0, "r.tag");
                    let payload = self.load_field(slot, 1, "r.pay");
                    let tag_f = self.builder.build_struct_gep(frame_ty, frame, 5, "sc.tag").unwrap();
                    self.builder.build_store(tag_f, tag).unwrap();
                    let pay_f = self.builder.build_struct_gep(frame_ty, frame, 6, "sc.pay").unwrap();
                    self.builder.build_store(pay_f, payload).unwrap();
                    // Park pc at the Return so the driver's result-type check
                    // reports the same pc the interpreter does.
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_RETURN);
                }
            }
        }
    }
}

fn step_name(func: DefId, version: Version) -> String {
    format!("step_{}_{}", func, version.0)
}

/// Extract a call's basic-value result (this inkwell fork returns a `ValueKind`).
fn call_result(cs: inkwell::values::CallSiteValue<'_>) -> inkwell::values::BasicValueEnum<'_> {
    match cs.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => panic!("call produced no value"),
    }
}

/// Compile every Ready function version in the world (old pinned versions
/// included) into `step` functions and wire the runtime externs.
pub(crate) fn compile<'ctx>(ctx: &'ctx Context, world: &World) -> Result<Compiled<'ctx>, JitError> {
    let cg = Codegen {
        ctx,
        module: ctx.create_module("livetype"),
        builder: ctx.create_builder(),
    };
    cg.declare_externs();

    let ready: Vec<&Function> = world
        .functions
        .values()
        .filter_map(|state| match state {
            FunctionState::Ready(f) => Some(f),
            FunctionState::Broken { .. } => None,
        })
        .collect();

    let mut steps = Vec::new();
    for func in &ready {
        steps.push((func, cg.declare_step(func)));
    }
    for (func, step) in &steps {
        cg.define_step(func, *step);
    }

    cg.module
        .verify()
        .map_err(|e| JitError(format!("module verification failed: {e}")))?;
    optimize(&cg.module);

    let engine = cg
        .module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| JitError(e.to_string()))?;
    for (name, addr) in [
        ("lt_new", lt_new as *const () as usize),
        ("lt_get_field", lt_get_field as *const () as usize),
        ("lt_emit", lt_emit as *const () as usize),
        ("lt_call_foreign", lt_call_foreign as *const () as usize),
        ("lt_load_global", lt_load_global as *const () as usize),
    ] {
        if let Some(f) = cg.module.get_function(name) {
            engine.add_global_mapping(&f, addr);
        }
    }

    let mut addrs = HashMap::new();
    for (func, _) in &steps {
        let name = step_name(func.id, func.version);
        let addr = engine
            .get_function_address(&name)
            .map_err(|_| JitError(format!("{name} not found in engine")))?;
        addrs.insert((func.id, func.version), addr as usize);
    }
    // Hand the module to `Compiled` so it outlives the engine (see the struct's
    // drop-order note). `steps` borrowed it, so drop those borrows first.
    drop(steps);
    Ok(Compiled {
        _engine: engine,
        _module: cg.module,
        addrs,
    })
}

fn optimize(module: &Module) {
    use inkwell::passes::PassBuilderOptions;
    use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
    Target::initialize_native(&InitializationConfig::default()).ok();
    let triple = TargetMachine::get_default_triple();
    let Ok(target) = Target::from_triple(&triple) else {
        return;
    };
    let Some(machine) = target.create_target_machine(
        &triple,
        &TargetMachine::get_host_cpu_name().to_string(),
        &TargetMachine::get_host_cpu_features().to_string(),
        OptimizationLevel::Aggressive,
        RelocMode::Default,
        CodeModel::Default,
    ) else {
        return;
    };
    let _ = module.run_passes("default<O2>", &machine, PassBuilderOptions::create());
}

// ---------------------------------------------------------------------------
// The driver — owns continuation semantics; native `step` never does.
// ---------------------------------------------------------------------------

/// One heap-resident frame of a JIT actor. `regs` is a stable, boxed slice: its
/// pointer is what native code (and the GC) read.
pub struct JitFrame {
    pub func_id: DefId,
    pub version: Version,
    pub pc: usize,
    pub regs: Box<[RawSlot]>,
    /// Caller register to receive the result, or `None` for the entry frame.
    pub return_to: Option<usize>,
}

pub struct JitActor {
    pub id: ActorId,
    pub frames: Vec<JitFrame>,
    pub status: ActorStatus,
}

impl JitActor {
    /// Spawn an actor at `func`'s current version, mirroring [`Runtime::spawn`]:
    /// entering a broken entry is a `BrokenFunction` condition.
    pub fn spawn(rt: &Runtime, id: ActorId, func: DefId, args: Vec<Value>) -> Result<JitActor, Condition> {
        Self::spawn_in(&rt.world, id, func, args)
    }

    /// Spawn from a bare [`World`] — used by the concurrent JIT runner, which
    /// only has the shared world (behind its lock), not a `Runtime`.
    pub fn spawn_in(world: &World, id: ActorId, func: DefId, args: Vec<Value>) -> Result<JitActor, Condition> {
        let version = *world.current_functions.get(&func).ok_or_else(|| {
            Condition::BrokenFunction {
                function: func,
                diagnostics: vec!["unknown function".into()],
            }
        })?;
        let FunctionState::Ready(code) = &world.functions[&(func, version)] else {
            let FunctionState::Broken { diagnostics, .. } = &world.functions[&(func, version)]
            else {
                unreachable!()
            };
            return Err(Condition::BrokenFunction {
                function: func,
                diagnostics: diagnostics.clone(),
            });
        };
        let mut regs = vec![RawSlot::EMPTY; code.registers];
        for (slot, value) in args.iter().enumerate() {
            regs[slot] = RawSlot::from_value(value);
        }
        Ok(JitActor {
            id,
            frames: vec![JitFrame {
                func_id: func,
                version,
                pc: 0,
                regs: regs.into_boxed_slice(),
                return_to: None,
            }],
            status: ActorStatus::Runnable,
        })
    }

    /// Every live [`ObjectId`] reachable from this actor's frame slots — the
    /// precise root set the design promises for the native path.
    pub fn roots(&self) -> Vec<ObjectId> {
        let mut roots = Vec::new();
        for frame in &self.frames {
            for slot in frame.regs.iter() {
                if slot.tag == TAG_REF {
                    roots.push(slot.payload as ObjectId);
                }
            }
        }
        roots
    }
}

/// Is a paused actor's condition now satisfiable? Mirrors the interpreter's
/// `resume_repaired` predicate.
fn repaired(rt: &Runtime, condition: &Condition) -> bool {
    match condition {
        Condition::BrokenFunction { function, .. } => rt
            .world
            .current_functions
            .get(function)
            .is_some_and(|v| matches!(rt.world.functions.get(&(*function, *v)), Some(FunctionState::Ready(_)))),
        Condition::MissingMigration { type_id, from, .. } => {
            rt.world.migrations.contains_key(&(*type_id, *from))
        }
        Condition::RuntimeTypeError { .. } => false,
    }
}

/// Drive one actor natively over the current world. Runs until the actor
/// pauses or completes; if `stop_on_yield` is set, also returns (still
/// `Runnable`) at the next `Yield` safe point so the caller can land an update
/// before resuming (DESIGN.md T5).
pub fn drive(rt: &mut Runtime, actor: &mut JitActor, stop_on_yield: bool) -> Result<(), JitError> {
    if let ActorStatus::Paused(condition) = &actor.status {
        if repaired(rt, condition) {
            actor.status = ActorStatus::Runnable;
        } else {
            return Ok(());
        }
    }
    if !matches!(actor.status, ActorStatus::Runnable) {
        return Ok(());
    }

    let ctx = Context::create();
    let compiled = compile(&ctx, &rt.world)?;

    while matches!(actor.status, ActorStatus::Runnable) {
        let frame = actor.frames.last_mut().unwrap();
        let Some(step) = compiled.step_of(frame.func_id, frame.version) else {
            return Err(JitError(format!(
                "no compiled step for {}@{}",
                frame.func_id, frame.version.0
            )));
        };
        let mut raw = RawFrame {
            func_id: frame.func_id as i64,
            version: frame.version.0 as i64,
            pc: frame.pc as i64,
            n_regs: frame.regs.len() as i64,
            regs: frame.regs.as_mut_ptr(),
            scratch: RawSlot::EMPTY,
            return_reg: frame.return_to.map_or(-1, |r| r as i64),
        };
        // Build a fresh host (borrowing `rt`) for just this step call, then
        // release the borrow so the outcome dispatch below can use `rt` again.
        // A trap is stashed in the host and picked up on `OUT_CONDITION`.
        let (outcome, pending) = {
            let mut host = JitHost::single(rt);
            let out = unsafe { step(&mut raw, &mut host as *mut JitHost) };
            (out, host.take_pending())
        };
        frame.pc = raw.pc as usize;

        match outcome {
            OUT_RETURN => {
                let result_slot = raw.scratch;
                let result = result_slot.to_value();
                // Check the result against the returning frame's declared type
                // before it leaves the frame — the con-freeness trap for a
                // pinned old function returning a since-migrated value.
                let (fid, ver, ret_pc) = {
                    let f = actor.frames.last().unwrap();
                    (f.func_id, f.version, f.pc)
                };
                if let FunctionState::Ready(f) = &rt.world.functions[&(fid, ver)] {
                    let result_ty = f.result.clone();
                    if let Err(condition) =
                        rt.expect_value(&result, &result_ty, fid, ret_pc, "return value")
                    {
                        actor.status = ActorStatus::Paused(condition);
                        continue;
                    }
                }
                let done = actor.frames.pop().unwrap();
                match done.return_to {
                    Some(reg) => actor.frames.last_mut().unwrap().regs[reg] = result_slot,
                    None => actor.status = ActorStatus::Complete(result),
                }
            }
            OUT_CALL => handle_call(rt, actor)?,
            OUT_CONDITION => {
                let condition =
                    pending.expect("CONDITION outcome without a stashed condition");
                actor.status = ActorStatus::Paused(condition);
            }
            OUT_TYPE_ERROR => {
                // Native operand-tag trap. Rebuild the exact condition from the
                // trapping instruction so it matches the interpreter.
                let (fid, ver, trap_pc) = {
                    let f = actor.frames.last().unwrap();
                    (f.func_id, f.version, f.pc)
                };
                let instruction = match &rt.world.functions[&(fid, ver)] {
                    FunctionState::Ready(f) => f.code[trap_pc].clone(),
                    _ => return Err(JitError("type-error frame pins non-ready code".into())),
                };
                actor.status =
                    ActorStatus::Paused(rt.operand_type_error(fid, trap_pc, &instruction));
            }
            OUT_YIELD => {
                if stop_on_yield {
                    return Ok(());
                }
            }
            other => return Err(JitError(format!("unknown step outcome {other}"))),
        }
    }
    Ok(())
}

/// Resume a JIT actor's con-freeness trap by supplying a value — the native
/// half of the delimited-continuation repair (mirrors [`Runtime::resume_with`]).
/// The offering must be well-typed for the trap's expected type, so repair can
/// never reintroduce an ill-typed value. After this the actor is `Runnable`;
/// call [`drive`] to continue it.
pub fn resume_with(rt: &Runtime, actor: &mut JitActor, value: Value) -> Result<(), JitError> {
    if !matches!(actor.status, ActorStatus::Paused(Condition::RuntimeTypeError { .. })) {
        return Err(JitError("actor is not paused on a resumable type trap".into()));
    }
    let (fid, ver, pc) = {
        let f = actor.frames.last().unwrap();
        (f.func_id, f.version, f.pc)
    };
    let FunctionState::Ready(f) = &rt.world.functions[&(fid, ver)] else {
        return Err(JitError("frame pins non-ready code".into()));
    };
    let result_ty = f.result.clone();
    let instruction = f.code[pc].clone();
    let (expected, plan) = resume_shape(&instruction, &result_ty, &rt.world).map_err(JitError)?;
    if !rt.value_ok(&value, &expected) {
        return Err(JitError(format!(
            "supplied value does not have the expected type {expected:?}"
        )));
    }
    let slot = RawSlot::from_value(&value);
    match plan {
        ResumePlan::SetAdvance(dst) => {
            let frame = actor.frames.last_mut().unwrap();
            frame.regs[dst] = slot;
            frame.pc += 1;
            actor.status = ActorStatus::Runnable;
        }
        ResumePlan::Branch(then_pc, else_pc) => {
            let take = matches!(value, Value::Bool(true));
            actor.frames.last_mut().unwrap().pc = if take { then_pc } else { else_pc };
            actor.status = ActorStatus::Runnable;
        }
        ResumePlan::ReturnValue => {
            let done = actor.frames.pop().unwrap();
            match done.return_to {
                Some(reg) => {
                    actor.frames.last_mut().unwrap().regs[reg] = slot;
                    actor.status = ActorStatus::Runnable;
                }
                None => actor.status = ActorStatus::Complete(value),
            }
        }
    }
    Ok(())
}

/// Interleave several actors over the one shared [`Runtime`] (its heap, world,
/// and effects), round-robin at `Yield` granularity: each turn advances one
/// runnable actor until its next safe point, pause, or completion, then moves
/// on. Because the heap is shared, one actor's lazy migration of an object is
/// visible to the others — the setting in which the soundness invariant is
/// stressed by concurrency. Returns when no actor is runnable (all paused or
/// complete). Deterministic: this is *semantic* concurrency (interleaving over a
/// shared heap), not OS threads, so there are no data races to reason about —
/// only whether type soundness survives shared, mid-flight migration.
pub fn run_interleaved(rt: &mut Runtime, actors: &mut [JitActor]) -> Result<(), JitError> {
    loop {
        let mut progressed = false;
        for actor in actors.iter_mut() {
            if matches!(actor.status, ActorStatus::Runnable) {
                drive(rt, actor, true)?;
                progressed = true;
            }
        }
        if !progressed {
            return Ok(());
        }
    }
}

/// Every live [`ObjectId`] rooted by any actor's frame slots — the precise root
/// set for a garbage collection over a set of concurrent actors.
pub fn all_roots(actors: &[JitActor]) -> Vec<ObjectId> {
    actors.iter().flat_map(|a| a.roots()).collect()
}

/// Handle a `Call` hand-back: read the call site from the caller's IR, gather
/// arguments from its slots, and either pause (broken callee) or push the new
/// frame. The caller's pc still points at the `Call`; advance it before pushing
/// so the callee returns into the following instruction.
fn handle_call(rt: &mut Runtime, actor: &mut JitActor) -> Result<(), JitError> {
    let (caller_id, caller_ver, call_pc) = {
        let f = actor.frames.last().unwrap();
        (f.func_id, f.version, f.pc)
    };
    let FunctionState::Ready(caller) = &rt.world.functions[&(caller_id, caller_ver)] else {
        return Err(JitError("caller frame pins non-ready code".into()));
    };
    let Instruction::Call {
        dst,
        function: callee,
        args,
    } = caller.code[call_pc].clone()
    else {
        return Err(JitError("CALL outcome at a non-call pc".into()));
    };

    let version = match rt.world.current_functions.get(&callee) {
        Some(v) => *v,
        None => {
            actor.status = ActorStatus::Paused(Condition::BrokenFunction {
                function: callee,
                diagnostics: vec!["unknown function".into()],
            });
            return Ok(());
        }
    };
    let code = match &rt.world.functions[&(callee, version)] {
        FunctionState::Ready(code) => code.clone(),
        FunctionState::Broken { diagnostics, .. } => {
            actor.status = ActorStatus::Paused(Condition::BrokenFunction {
                function: callee,
                diagnostics: diagnostics.clone(),
            });
            return Ok(());
        }
    };

    let arg_slots: Vec<RawSlot> = {
        let caller_frame = actor.frames.last().unwrap();
        args.iter().map(|r| caller_frame.regs[*r]).collect()
    };
    // Check each argument against the callee's parameter type before the frame
    // is pushed — a pinned old caller passing a since-migrated value traps here.
    for (slot, expected) in arg_slots.iter().zip(&code.params) {
        if let Err(condition) =
            rt.expect_value(&slot.to_value(), expected, callee, call_pc, "call argument")
        {
            actor.status = ActorStatus::Paused(condition);
            return Ok(());
        }
    }
    let mut regs = vec![RawSlot::EMPTY; code.registers];
    for (slot, value) in arg_slots.into_iter().enumerate() {
        regs[slot] = value;
    }
    actor.frames.last_mut().unwrap().pc = call_pc + 1;
    actor.frames.push(JitFrame {
        func_id: callee,
        version,
        pc: 0,
        regs: regs.into_boxed_slice(),
        return_to: Some(dst),
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// The JIT under threads. Worker threads execute the same compiled `step`
// functions over the thread-safe `Shared` runtime — the JIT counterpart of the
// interpreter's `Shared::run_threads`. The runtime still owns continuation
// semantics per thread; only the pure/effect ops run as native code. Respects
// the LLVM/Miri boundary: `livetype-core` never links LLVM — this crate owns
// the threads and the compiled code and calls *into* core's `Shared`.
// ---------------------------------------------------------------------------

use std::sync::{Arc, Mutex};

/// A version-cached JIT code store, shared by the worker threads of one run. It
/// compiles the world lazily and *recompiles when the world's epoch advances* —
/// which is what lets a program being live-edited on the JIT threads pick up new
/// function versions. It hands out only resolved code addresses (`usize`, so
/// `Send`); the engines that back them are leaked so they outlive every caller
/// (a research-prototype tradeoff: one engine leaks per edit generation — a real
/// system would reclaim them once no frame pins an old version). Recompiles are
/// serialized under the lock, so LLVM's compilation globals are never raced;
/// executing already-compiled code stays fully concurrent.
pub struct JitCode {
    inner: Mutex<JitCodeInner>,
}
struct JitCodeInner {
    epoch: Option<u64>,
    addrs: Arc<HashMap<(DefId, Version), usize>>,
}

impl JitCode {
    pub fn new() -> JitCode {
        JitCode {
            inner: Mutex::new(JitCodeInner {
                epoch: None,
                addrs: Arc::new(HashMap::new()),
            }),
        }
    }

    /// The current address map, recompiling first if the world has changed since
    /// the last compile (or if nothing is compiled yet). Takes a bare [`World`]
    /// so both the concurrent tier (via `Shared::with_world`) and the
    /// single-threaded tiered runtime can share one cache.
    pub fn addrs(&self, world: &World) -> Result<Arc<HashMap<(DefId, Version), usize>>, JitError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.epoch != Some(world.epoch) {
            // Leak the context so the compiled code outlives this call; extract
            // the addresses (the only thing callers need), then leak the
            // `Compiled` too so its engine stays alive.
            let ctx: &'static Context = Box::leak(Box::new(Context::create()));
            let compiled = compile(ctx, world)?;
            let addrs = compiled.addr_map();
            Box::leak(Box::new(compiled));
            inner.addrs = Arc::new(addrs);
            inner.epoch = Some(world.epoch);
        }
        Ok(Arc::clone(&inner.addrs))
    }
}

impl Default for JitCode {
    fn default() -> Self {
        JitCode::new()
    }
}

/// Run each `(function, args)` as an actor on its own OS thread, executing
/// JIT-compiled `step` functions over one shared runtime. Code is compiled lazily
/// and cached by world epoch (see [`JitCode`]); worker threads share the resolved
/// addresses and recompile-on-demand when a live edit advances the world, so a
/// program can be edited *while it runs on the JIT threads*. Returns each actor's
/// [`Outcome`] in order.
pub fn run_jit_threads(
    shared: &Arc<Shared>,
    actors: Vec<(DefId, Vec<Value>)>,
) -> Result<Vec<Outcome>, JitError> {
    let code = Arc::new(JitCode::new());
    // Mailboxes up front so a Send can't race its recipient into existence.
    for tid in 0..actors.len() {
        shared.ensure_mailbox(tid);
    }
    let handles: Vec<_> = actors
        .into_iter()
        .enumerate()
        .map(|(tid, (func, args))| {
            let shared = Arc::clone(shared);
            let code = Arc::clone(&code);
            std::thread::spawn(move || concurrent_actor(&shared, &code, tid, func, args))
        })
        .collect();
    Ok(handles.into_iter().map(|h| h.join().unwrap()).collect())
}

/// One JIT worker: spawn the actor, register as a mutator, drive it to a stop.
fn concurrent_actor(
    shared: &Arc<Shared>,
    code: &Arc<JitCode>,
    tid: usize,
    func: DefId,
    args: Vec<Value>,
) -> Outcome {
    struct Active<'a>(&'a Shared, usize);
    impl Drop for Active<'_> {
        fn drop(&mut self) {
            self.0.unregister(self.1);
        }
    }
    shared.register();
    let _active = Active(shared, tid);

    let mut actor = match shared.with_world(|w| JitActor::spawn_in(w, tid as ActorId, func, args)) {
        Ok(a) => a,
        Err(c) => return Outcome::Paused(c),
    };
    if let Err(e) = concurrent_drive(shared, code, tid, &mut actor) {
        // A driver-level fault (e.g. an edit introduced a version with no
        // compiled code) surfaces as a clear trap, never a silent stall.
        return Outcome::Paused(Condition::RuntimeTypeError {
            function: 0,
            pc: 0,
            message: e.0,
        });
    }
    match actor.status {
        ActorStatus::Complete(v) => Outcome::Complete(v),
        ActorStatus::Paused(c) => Outcome::Paused(c),
        ActorStatus::Runnable => Outcome::Paused(Condition::RuntimeTypeError {
            function: 0,
            pc: 0,
            message: "actor left runnable".into(),
        }),
    }
}

/// Drive one JIT actor over `Shared` to completion or a pause. Mirrors [`drive`]
/// but reads the world under the shared read lock and hits a GC safepoint each
/// step (publishing its native frame roots) so a stop-the-world collection can
/// pause it. `Yield` never stops here — worker threads run to completion.
fn concurrent_drive(
    shared: &Arc<Shared>,
    code: &Arc<JitCode>,
    tid: usize,
    actor: &mut JitActor,
) -> Result<(), JitError> {
    let mut addrs = shared.with_world(|w| code.addrs(w))?;
    while matches!(actor.status, ActorStatus::Runnable) {
        if shared.gc_pending() {
            shared.safepoint_roots(tid, actor.roots());
        }
        let key = {
            let f = actor.frames.last().unwrap();
            (f.func_id, f.version)
        };
        // A live edit may have introduced this version since we last compiled;
        // recompile-on-demand (cached by epoch) picks it up.
        if !addrs.contains_key(&key) {
            addrs = shared.with_world(|w| code.addrs(w))?;
        }
        let Some(&addr) = addrs.get(&key) else {
            return Err(JitError(format!(
                "no compiled step for {}@{}",
                key.0, key.1.0
            )));
        };
        let step = step_at(addr);
        let frame = actor.frames.last_mut().unwrap();
        let mut raw = RawFrame {
            func_id: frame.func_id as i64,
            version: frame.version.0 as i64,
            pc: frame.pc as i64,
            n_regs: frame.regs.len() as i64,
            regs: frame.regs.as_mut_ptr(),
            scratch: RawSlot::EMPTY,
            return_reg: frame.return_to.map_or(-1, |r| r as i64),
        };
        let (outcome, pending) = {
            let mut host = JitHost::concurrent(shared);
            let out = unsafe { step(&mut raw, &mut host as *mut JitHost) };
            (out, host.take_pending())
        };
        frame.pc = raw.pc as usize;

        match outcome {
            OUT_RETURN => {
                let result = raw.scratch.to_value();
                let (fid, ver, ret_pc) = {
                    let f = actor.frames.last().unwrap();
                    (f.func_id, f.version, f.pc)
                };
                let result_ty = shared.with_world(|w| match &w.functions[&(fid, ver)] {
                    FunctionState::Ready(f) => Some(f.result.clone()),
                    _ => None,
                });
                if let Some(ty) = result_ty {
                    if !shared.value_ok(&result, &ty) {
                        actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                            function: fid,
                            pc: ret_pc,
                            message: format!(
                                "return value: expected {ty:?}, found a value of another type"
                            ),
                        });
                        continue;
                    }
                }
                let done = actor.frames.pop().unwrap();
                match done.return_to {
                    Some(reg) => actor.frames.last_mut().unwrap().regs[reg] = raw.scratch,
                    None => actor.status = ActorStatus::Complete(result),
                }
            }
            OUT_CALL => handle_call_concurrent(shared, actor)?,
            OUT_CONDITION => {
                actor.status = ActorStatus::Paused(
                    pending.expect("CONDITION outcome without a stashed condition"),
                );
            }
            OUT_TYPE_ERROR => {
                let (fid, ver, trap_pc) = {
                    let f = actor.frames.last().unwrap();
                    (f.func_id, f.version, f.pc)
                };
                let instruction = shared.with_world(|w| match &w.functions[&(fid, ver)] {
                    FunctionState::Ready(f) => Some(f.code[trap_pc].clone()),
                    _ => None,
                });
                let Some(instruction) = instruction else {
                    return Err(JitError("type-error frame pins non-ready code".into()));
                };
                actor.status = ActorStatus::Paused(operand_type_error(fid, trap_pc, &instruction));
            }
            OUT_YIELD => {} // run to completion
            other => return Err(JitError(format!("unknown step outcome {other}"))),
        }
    }
    Ok(())
}

/// A `Call` hand-back on the concurrent tier — mirrors [`handle_call`], reading
/// the world under the shared read lock.
fn handle_call_concurrent(shared: &Arc<Shared>, actor: &mut JitActor) -> Result<(), JitError> {
    let (caller_id, caller_ver, call_pc) = {
        let f = actor.frames.last().unwrap();
        (f.func_id, f.version, f.pc)
    };
    let call = shared.with_world(|w| match &w.functions[&(caller_id, caller_ver)] {
        FunctionState::Ready(caller) => Ok(caller.code[call_pc].clone()),
        _ => Err(JitError("caller frame pins non-ready code".into())),
    })?;
    let Instruction::Call { dst, function: callee, args } = call else {
        return Err(JitError("CALL outcome at a non-call pc".into()));
    };

    // Resolve the callee's current version + params under one guard.
    let resolved = shared.with_world(|w| match w.current_functions.get(&callee) {
        None => Err(vec!["unknown function".to_string()]),
        Some(v) => match &w.functions[&(callee, *v)] {
            FunctionState::Ready(code) => Ok((*v, code.params.clone(), code.registers)),
            FunctionState::Broken { diagnostics, .. } => Err(diagnostics.clone()),
        },
    });
    let (version, params, registers) = match resolved {
        Ok(x) => x,
        Err(diagnostics) => {
            actor.status = ActorStatus::Paused(Condition::BrokenFunction {
                function: callee,
                diagnostics,
            });
            return Ok(());
        }
    };

    let arg_slots: Vec<RawSlot> = {
        let caller_frame = actor.frames.last().unwrap();
        args.iter().map(|r| caller_frame.regs[*r]).collect()
    };
    for (slot, expected) in arg_slots.iter().zip(&params) {
        if !shared.value_ok(&slot.to_value(), expected) {
            actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                function: callee,
                pc: call_pc,
                message: "call argument: expected a value of another type".into(),
            });
            return Ok(());
        }
    }
    let mut regs = vec![RawSlot::EMPTY; registers];
    for (slot, value) in arg_slots.into_iter().enumerate() {
        regs[slot] = value;
    }
    actor.frames.last_mut().unwrap().pc = call_pc + 1;
    actor.frames.push(JitFrame {
        func_id: callee,
        version,
        pc: 0,
        regs: regs.into_boxed_slice(),
        return_to: Some(dst),
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Auto-tiering: start interpreted, promote hot functions to JIT.
//
// One runtime, two engines, chosen automatically. A function begins as an
// interpreted frame; a per-(func, version) call counter promotes it once it
// crosses a threshold, and thereafter its frames run as compiled `step`
// functions. A single actor's stack freely mixes interpreted and JIT frames,
// marshalling `Value <-> RawSlot` at the call/return boundaries — so a hot
// callee runs native while its cold caller keeps interpreting, with no
// re-implementation of either engine (the interpreter's `step_instruction` and
// the JIT's compiled `step` are reused verbatim).
// ---------------------------------------------------------------------------

use std::collections::HashSet;

/// A frame in a tiered actor: either interpreted (register `Value`s) or JIT
/// (flat `RawSlot`s). Both carry the caller register their result returns to.
enum TierFrame {
    Interp(Frame),
    Jit(JitFrame),
}

/// A single-threaded runtime that auto-promotes hot functions from the
/// interpreter to the JIT.
pub struct Tiered {
    rt: Runtime,
    code: JitCode,
    counts: HashMap<(DefId, Version), u64>,
    hot: HashSet<(DefId, Version)>,
    threshold: u64,
    stack: Vec<TierFrame>,
    result: Option<Value>,
    paused: Option<Condition>,
}

impl Tiered {
    /// Wrap a set-up runtime; `threshold` call-count promotes a function to JIT.
    pub fn new(rt: Runtime, threshold: u64) -> Tiered {
        Tiered {
            rt,
            code: JitCode::new(),
            counts: HashMap::new(),
            hot: HashSet::new(),
            threshold,
            stack: Vec::new(),
            result: None,
            paused: None,
        }
    }

    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }
    /// How many function versions have been promoted to JIT so far.
    pub fn promoted(&self) -> usize {
        self.hot.len()
    }
    /// Was `(func, version)` promoted to the JIT?
    pub fn is_hot(&self, func: DefId, version: Version) -> bool {
        self.hot.contains(&(func, version))
    }

    /// Run `func(args)` to completion or a trap, promoting hot callees mid-run.
    pub fn run(&mut self, func: DefId, args: Vec<Value>) -> Outcome {
        let (version, regcount) = match self.rt.world.current_functions.get(&func) {
            Some(v) => match &self.rt.world.functions[&(func, *v)] {
                FunctionState::Ready(f) => (*v, f.registers),
                FunctionState::Broken { diagnostics, .. } => {
                    return Outcome::Paused(Condition::BrokenFunction {
                        function: func,
                        diagnostics: diagnostics.clone(),
                    });
                }
            },
            None => {
                return Outcome::Paused(Condition::BrokenFunction {
                    function: func,
                    diagnostics: vec!["unknown function".into()],
                });
            }
        };
        let mut registers = vec![None; regcount];
        for (i, v) in args.into_iter().enumerate() {
            registers[i] = Some(v);
        }
        self.stack.push(TierFrame::Interp(Frame {
            function: (func, version),
            pc: 0,
            registers,
            return_to: None,
        }));

        loop {
            if let Some(v) = self.result.take() {
                return Outcome::Complete(v);
            }
            if let Some(c) = self.paused.take() {
                return Outcome::Paused(c);
            }
            match self.stack.last() {
                Some(TierFrame::Jit(_)) => self.jit_step(),
                Some(TierFrame::Interp(_)) => self.interp_step(),
                None => return Outcome::Complete(Value::Unit),
            }
        }
    }

    /// Push a callee frame, promoting to JIT if it is (now) hot. `registers` is
    /// the full seeded register array; for a JIT frame it is marshalled to
    /// `RawSlot`s. A failed compile degrades gracefully to interpreting.
    fn push_callee(
        &mut self,
        callee: DefId,
        version: Version,
        registers: Vec<Option<Value>>,
        return_to: Option<usize>,
    ) {
        let key = (callee, version);
        *self.counts.entry(key).or_insert(0) += 1;
        if self.counts[&key] >= self.threshold {
            self.hot.insert(key);
        }
        if self.hot.contains(&key) {
            if let Ok(addrs) = self.code.addrs(&self.rt.world) {
                if addrs.contains_key(&key) {
                    let regs = registers
                        .iter()
                        .map(|o| o.as_ref().map_or(RawSlot::EMPTY, RawSlot::from_value))
                        .collect();
                    self.stack.push(TierFrame::Jit(JitFrame {
                        func_id: callee,
                        version,
                        pc: 0,
                        regs,
                        return_to,
                    }));
                    return;
                }
            }
        }
        self.stack.push(TierFrame::Interp(Frame {
            function: key,
            pc: 0,
            registers,
            return_to,
        }));
    }

    /// Pop the returning frame and deliver `value` to the caller, converting to
    /// the caller's representation; if the stack empties, the actor completes.
    fn deliver_return(&mut self, value: Value) {
        let done = self.stack.pop().unwrap();
        let return_to = match &done {
            TierFrame::Interp(f) => f.return_to,
            TierFrame::Jit(f) => f.return_to,
        };
        match self.stack.last_mut() {
            None => self.result = Some(value),
            Some(TierFrame::Interp(f)) => {
                if let Some(r) = return_to {
                    f.registers[r] = Some(value);
                }
            }
            Some(TierFrame::Jit(f)) => {
                if let Some(r) = return_to {
                    f.regs[r] = RawSlot::from_value(&value);
                }
            }
        }
    }

    fn interp_step(&mut self) {
        let instr = {
            let (key, pc) = match self.stack.last().unwrap() {
                TierFrame::Interp(f) => (f.function, f.pc),
                _ => unreachable!(),
            };
            match &self.rt.world.functions[&key] {
                FunctionState::Ready(f) => f.code[pc].clone(),
                _ => {
                    self.paused = Some(Condition::RuntimeTypeError {
                        function: key.0,
                        pc,
                        message: "interp frame pins non-ready code".into(),
                    });
                    return;
                }
            }
        };
        let mut m = TieredMachine { t: self };
        if let Err(condition) = step_instruction(&mut m, &instr) {
            self.paused = Some(condition);
        }
    }

    fn jit_step(&mut self) {
        let key = match self.stack.last().unwrap() {
            TierFrame::Jit(f) => (f.func_id, f.version),
            _ => unreachable!(),
        };
        let addrs = match self.code.addrs(&self.rt.world) {
            Ok(a) => a,
            Err(e) => {
                self.paused = Some(Condition::RuntimeTypeError { function: key.0, pc: 0, message: e.0 });
                return;
            }
        };
        let Some(&addr) = addrs.get(&key) else {
            self.paused = Some(Condition::RuntimeTypeError {
                function: key.0,
                pc: 0,
                message: "no compiled step".into(),
            });
            return;
        };
        let step = step_at(addr);
        let (outcome, pending, scratch) = {
            let frame = match self.stack.last_mut().unwrap() {
                TierFrame::Jit(f) => f,
                _ => unreachable!(),
            };
            let mut raw = RawFrame {
                func_id: frame.func_id as i64,
                version: frame.version.0 as i64,
                pc: frame.pc as i64,
                n_regs: frame.regs.len() as i64,
                regs: frame.regs.as_mut_ptr(),
                scratch: RawSlot::EMPTY,
                return_reg: frame.return_to.map_or(-1, |r| r as i64),
            };
            let mut host = JitHost::single(&mut self.rt);
            let out = unsafe { step(&mut raw, &mut host as *mut JitHost) };
            frame.pc = raw.pc as usize;
            (out, host.take_pending(), raw.scratch)
        };

        match outcome {
            OUT_RETURN => {
                let result = scratch.to_value();
                let (fid, ver, ret_pc) = match self.stack.last().unwrap() {
                    TierFrame::Jit(f) => (f.func_id, f.version, f.pc),
                    _ => unreachable!(),
                };
                if let FunctionState::Ready(f) = &self.rt.world.functions[&(fid, ver)] {
                    let ty = f.result.clone();
                    if !self.rt.value_ok(&result, &ty) {
                        self.paused = Some(Condition::RuntimeTypeError {
                            function: fid,
                            pc: ret_pc,
                            message: format!("return value: expected {ty:?}, found a value of another type"),
                        });
                        return;
                    }
                }
                self.deliver_return(result);
            }
            OUT_CALL => self.jit_handle_call(),
            OUT_YIELD => {} // run to completion
            OUT_CONDITION => self.paused = pending,
            OUT_TYPE_ERROR => {
                let (fid, ver, trap_pc) = match self.stack.last().unwrap() {
                    TierFrame::Jit(f) => (f.func_id, f.version, f.pc),
                    _ => unreachable!(),
                };
                let instruction = match &self.rt.world.functions[&(fid, ver)] {
                    FunctionState::Ready(f) => f.code[trap_pc].clone(),
                    _ => {
                        self.paused = Some(Condition::RuntimeTypeError { function: fid, pc: trap_pc, message: "non-ready".into() });
                        return;
                    }
                };
                self.paused = Some(operand_type_error(fid, trap_pc, &instruction));
            }
            other => {
                self.paused = Some(Condition::RuntimeTypeError {
                    function: key.0,
                    pc: 0,
                    message: format!("unknown step outcome {other}"),
                });
            }
        }
    }

    /// A `Call` hand-back from a JIT frame: gather args from its slots, decide
    /// the callee's tier, and push the matching frame.
    fn jit_handle_call(&mut self) {
        let (caller_key, call_pc) = match self.stack.last().unwrap() {
            TierFrame::Jit(f) => ((f.func_id, f.version), f.pc),
            _ => unreachable!(),
        };
        let Instruction::Call { dst, function: callee, args } =
            (match &self.rt.world.functions[&caller_key] {
                FunctionState::Ready(f) => f.code[call_pc].clone(),
                _ => unreachable!(),
            })
        else {
            self.paused = Some(Condition::RuntimeTypeError { function: caller_key.0, pc: call_pc, message: "CALL at non-call pc".into() });
            return;
        };
        let (version, params, regcount) = match self.rt.world.current_functions.get(&callee) {
            None => {
                self.paused = Some(Condition::BrokenFunction { function: callee, diagnostics: vec!["unknown function".into()] });
                return;
            }
            Some(v) => match &self.rt.world.functions[&(callee, *v)] {
                FunctionState::Ready(f) => (*v, f.params.clone(), f.registers),
                FunctionState::Broken { diagnostics, .. } => {
                    self.paused = Some(Condition::BrokenFunction { function: callee, diagnostics: diagnostics.clone() });
                    return;
                }
            },
        };
        // Gather + type-check args from the caller's JIT slots.
        let arg_vals: Vec<Value> = {
            let frame = match self.stack.last().unwrap() {
                TierFrame::Jit(f) => f,
                _ => unreachable!(),
            };
            args.iter().map(|r| frame.regs[*r].to_value()).collect()
        };
        for (v, expected) in arg_vals.iter().zip(&params) {
            if !self.rt.value_ok(v, expected) {
                self.paused = Some(Condition::RuntimeTypeError {
                    function: callee,
                    pc: call_pc,
                    message: "call argument: expected a value of another type".into(),
                });
                return;
            }
        }
        // Advance the caller past the Call, then push the callee.
        if let TierFrame::Jit(f) = self.stack.last_mut().unwrap() {
            f.pc = call_pc + 1;
        }
        let mut registers = vec![None; regcount];
        for (i, v) in arg_vals.into_iter().enumerate() {
            registers[i] = Some(v);
        }
        self.push_callee(callee, version, registers, Some(dst));
    }
}

/// The [`Machine`] for a tiered actor's *interpreted* frames — reads/writes the
/// top interpreted frame, and routes `Call`/`Return` through the tiering policy
/// (`push_callee`/`deliver_return`), so a call from interpreted code can land in
/// a JIT frame and vice versa.
struct TieredMachine<'a> {
    t: &'a mut Tiered,
}

impl TieredMachine<'_> {
    fn top(&self) -> &Frame {
        match self.t.stack.last().unwrap() {
            TierFrame::Interp(f) => f,
            _ => unreachable!("interp step over a non-interp frame"),
        }
    }
    fn top_mut(&mut self) -> &mut Frame {
        match self.t.stack.last_mut().unwrap() {
            TierFrame::Interp(f) => f,
            _ => unreachable!(),
        }
    }
}

impl Machine for TieredMachine<'_> {
    fn world(&self) -> &World {
        &self.t.rt.world
    }
    fn heap(&self) -> &Heap {
        &self.t.rt.heap
    }
    fn current(&self) -> (DefId, Version) {
        self.top().function
    }
    fn pc(&self) -> usize {
        self.top().pc
    }
    fn reg(&self, i: usize) -> Option<Value> {
        self.top().registers.get(i).cloned().flatten()
    }
    fn set_reg(&mut self, dst: usize, value: Value) {
        self.top_mut().registers[dst] = Some(value);
    }
    fn set_pc(&mut self, pc: usize) {
        self.top_mut().pc = pc;
    }
    fn advance(&mut self) {
        self.top_mut().pc += 1;
    }
    fn emit(&mut self, value: Value) {
        self.t.rt.jit_emit(value);
    }
    fn global(&self, id: DefId) -> GlobalRead {
        match self.t.rt.globals.get(&id) {
            Some(v) => GlobalRead::Value(v.clone()),
            None => GlobalRead::Missing,
        }
    }
    fn call_foreign(&mut self, id: ForeignFnId, args: &[Value]) -> ForeignCall {
        ForeignCall::Done(self.t.rt.call_foreign_raw(id, args))
    }
    fn push_call(&mut self, callee: DefId, version: Version, registers: Vec<Option<Value>>, return_reg: usize) {
        self.t.push_callee(callee, version, registers, Some(return_reg));
    }
    fn do_return(&mut self, value: Value) {
        self.t.deliver_return(value);
    }
    fn send(&mut self, _t: usize, _v: Value) -> Option<bool> {
        None
    }
    fn recv(&mut self) -> RecvResult {
        RecvResult::Unsupported
    }
}
