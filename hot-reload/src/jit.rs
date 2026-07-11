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

/// One typed register slot. For `TAG_REF` the payload is the [`ObjectId`] — a
/// GC root the collector reads directly out of the frame.
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
        }
    }

    pub fn to_value(self) -> Value {
        match self.tag {
            TAG_UNIT => Value::Unit,
            TAG_I64 => Value::I64(self.payload),
            TAG_BOOL => Value::Bool(self.payload != 0),
            TAG_REF => Value::Ref(self.payload as ObjectId),
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
// Runtime externs. Native code calls these for the ops that touch runtime
// state; each is a thin bridge to a shared `Runtime` helper, so the JIT and the
// interpreter cannot diverge on allocation, migration, or effects.
// ---------------------------------------------------------------------------

/// Returns 0 on success (writes `*out_objid`), 1 when construction trips the
/// soundness check (the condition is stashed in `rt.pending_condition`).
///
/// # Safety
/// `rt` is a live `*mut Runtime`, `fields` points to `n` `SuppliedField`s,
/// `out_objid` is a writable `*mut i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_new(
    rt: *mut Runtime,
    type_id: i64,
    fields: *const SuppliedField,
    n: i64,
    out_objid: *mut i64,
) -> i64 {
    let rt = unsafe { &mut *rt };
    let mut supplied = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        let f = unsafe { &*fields.offset(i) };
        supplied.push((f.field_id as FieldId, f.value.to_value()));
    }
    match rt.jit_new(type_id as DefId, &supplied) {
        Ok(id) => {
            unsafe { *out_objid = id as i64 };
            0
        }
        Err(condition) => {
            rt.pending_condition = Some(condition);
            1
        }
    }
}

/// Returns 0 on success (writes `*out`), 1 when a migration barrier trips (the
/// condition is stashed in `rt.pending_condition` for the driver).
///
/// # Safety
/// `rt` is a live `*mut Runtime`, `out` a writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_get_field(
    rt: *mut Runtime,
    objid: i64,
    field: i64,
    out: *mut RawSlot,
) -> i64 {
    let rt = unsafe { &mut *rt };
    match rt.jit_get_field(objid as ObjectId, field as FieldId) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            rt.pending_condition = Some(condition);
            1
        }
    }
}

/// # Safety
/// `rt` is a live `*mut Runtime`, `value` a readable `*const RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_emit(rt: *mut Runtime, value: *const RawSlot) {
    let rt = unsafe { &mut *rt };
    rt.jit_emit(unsafe { *value }.to_value());
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
pub struct Compiled<'ctx> {
    _engine: ExecutionEngine<'ctx>,
    _module: Module<'ctx>,
    addrs: HashMap<(DefId, Version), usize>,
}

type StepFn = unsafe extern "C" fn(*mut RawFrame, *mut Runtime) -> i64;

impl<'ctx> Compiled<'ctx> {
    fn step_of(&self, func: DefId, version: Version) -> Option<StepFn> {
        self.addrs
            .get(&(func, version))
            .map(|addr| unsafe { std::mem::transmute::<usize, StepFn>(*addr) })
    }
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
pub fn compile<'ctx>(ctx: &'ctx Context, rt: &Runtime) -> Result<Compiled<'ctx>, JitError> {
    let cg = Codegen {
        ctx,
        module: ctx.create_module("livetype"),
        builder: ctx.create_builder(),
    };
    cg.declare_externs();

    let ready: Vec<&Function> = rt
        .world
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
        let version = *rt.world.current_functions.get(&func).ok_or_else(|| {
            Condition::BrokenFunction {
                function: func,
                diagnostics: vec!["unknown function".into()],
            }
        })?;
        let FunctionState::Ready(code) = &rt.world.functions[&(func, version)] else {
            let FunctionState::Broken { diagnostics, .. } = &rt.world.functions[&(func, version)]
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
    let compiled = compile(&ctx, rt)?;

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
        // Reborrow `rt` fresh each iteration: intervening `&mut rt` uses below
        // (handle_call, pending_condition) would invalidate a pointer derived
        // once before the loop, which the optimizer is entitled to exploit.
        let rt_ptr: *mut Runtime = rt as *mut Runtime;
        let outcome = unsafe { step(&mut raw, rt_ptr) };
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
                let condition = rt
                    .pending_condition
                    .take()
                    .expect("CONDITION outcome without a stashed condition");
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
