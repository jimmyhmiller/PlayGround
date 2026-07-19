//! LLVM `step` backend — the compiler half of the native tier.
//!
//! Each Ready function version is JIT-compiled to a
//! `step(frame*, host*) -> outcome` function. Native code runs the pure ops
//! and the non-pausing runtime calls (through the externs defined in
//! `livetype-core`), then *hands back* to the engine for everything that owns
//! continuation semantics: pushing a frame (`Call`), ending one (`Return`),
//! pausing (`Condition`), or reaching a recurring safe point (`Yield`). LLVM
//! therefore accelerates execution without owning pause/resume — exactly the
//! design's split.
//!
//! This crate contains ONLY the compiler: the raw frame representation, the
//! runtime externs, and the executor all live in `livetype-core` (they are
//! LLVM-free). The seam is [`TierSource`]: the engine asks for a native step
//! address, and [`JitCode`] answers by compiling the current world on demand,
//! cached by world epoch.
//!
//! Two properties make this sound and simple:
//!   * **No SSA lives across a basic block.** Every register read loads from
//!     `frame->regs[i]`; every write stores back. Loops and branches need no
//!     phi nodes, and — the design's key claim — every live reference sits in a
//!     typed frame slot, so the GC has a complete precise root map.
//!   * **Recompile per epoch.** A live edit bumps the world epoch; the next
//!     query rebuilds the module/engine for the current world, so repaired
//!     code is simply the next compile.

use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::types::StructType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate};
use livetype_core::{
    DefId, Engine, Function, FunctionState, Instruction, OUT_CALL, OUT_CONDITION, OUT_RETURN,
    OUT_TYPE_ERROR, OUT_YIELD, RawSlot, TAG_BOOL, TAG_F64, TAG_I64, TAG_REF, TAG_STR, TierSource,
    Version, World, lt_array_len, lt_array_push, lt_call_foreign, lt_case_variant, lt_concat_str,
    lt_emit, lt_get_field, lt_index_get, lt_index_set, lt_load_global, lt_new, lt_new_array,
    lt_new_variant, lt_trap_div_zero,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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


impl<'ctx> Compiled<'ctx> {
    /// A `Send` copy of the compiled addresses, for handing to worker threads.
    /// The addresses are raw code pointers — valid to call from any thread as
    /// long as this `Compiled` (which owns the engine backing them) outlives the
    /// threads. The concurrent runner guarantees that by joining before drop.
    fn addr_map(&self) -> HashMap<(DefId, Version), usize> {
        self.addrs.clone()
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
        self.module.add_function(
            "lt_concat_str",
            i64t.fn_type(&[i64t.into(), i64t.into()], false),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_new_variant",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), i64t.into(), ptr.into(), i64t.into(), ptr.into()],
                false,
            ),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_case_variant",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), ptr.into(), i64t.into(), ptr.into()],
                false,
            ),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_trap_div_zero",
            void.fn_type(&[ptr.into()], false),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_new_array",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), ptr.into(), i64t.into(), ptr.into()],
                false,
            ),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_index_get",
            i64t.fn_type(&[ptr.into(), i64t.into(), i64t.into(), ptr.into()], false),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_index_set",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), i64t.into(), i64t.into(), i64t.into()],
                false,
            ),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_array_len",
            i64t.fn_type(&[ptr.into(), i64t.into(), ptr.into()], false),
            Some(Linkage::External),
        );
        self.module.add_function(
            "lt_array_push",
            i64t.fn_type(
                &[ptr.into(), i64t.into(), i64t.into(), i64t.into()],
                false,
            ),
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

    /// The shared constructor lowering for `New` (variant `None`) and
    /// `NewVariant` (variant `Some`): marshal supplied `(field_id, slot)`
    /// triples into the entry-block scratch as `SuppliedField[n]`, call the
    /// matching extern, branch on its status (a failed soundness check pauses),
    /// and store the fresh object id. Leaves the builder wired to fall through.
    #[allow(clippy::too_many_arguments)]
    fn emit_constructor(
        &self,
        step: FunctionValue<'ctx>,
        frame: PointerValue<'ctx>,
        rt: PointerValue<'ctx>,
        scratch: Option<PointerValue<'ctx>>,
        out_objid_slot: Option<PointerValue<'ctx>>,
        blocks: &[inkwell::basic_block::BasicBlock<'ctx>],
        pc: usize,
        dst: usize,
        type_id: DefId,
        variant: Option<u64>,
        fields: &[(livetype_core::FieldId, usize)],
    ) {
        let i64t = self.ctx.i64_type();
        let n = fields.len();
        let arr_ty = i64t.array_type((3 * n).max(1) as u32);
        let arr = scratch.expect("a constructor implies scratch space");
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
        let out_objid = out_objid_slot.expect("a constructor implies an out slot");
        let call = match variant {
            None => {
                let lt_new = self.module.get_function("lt_new").unwrap();
                self.builder
                    .build_call(
                        lt_new,
                        &[
                            rt.into(),
                            i64t.const_int(type_id, false).into(),
                            base.into(),
                            i64t.const_int(n as u64, false).into(),
                            out_objid.into(),
                        ],
                        "status",
                    )
                    .unwrap()
            }
            Some(variant) => {
                let lt_nv = self.module.get_function("lt_new_variant").unwrap();
                self.builder
                    .build_call(
                        lt_nv,
                        &[
                            rt.into(),
                            i64t.const_int(type_id, false).into(),
                            i64t.const_int(variant, false).into(),
                            base.into(),
                            i64t.const_int(n as u64, false).into(),
                            out_objid.into(),
                        ],
                        "status",
                    )
                    .unwrap()
            }
        };
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
        let objid = self
            .builder
            .build_load(i64t, out_objid, "objid")
            .unwrap()
            .into_int_value();
        self.store_slot(frame, dst, i64t.const_int(TAG_REF as u64, false), objid);
        if pc + 1 < blocks.len() {
            self.builder.build_unconditional_branch(blocks[pc + 1]).unwrap();
        }
    }

    /// Branch on an extern's status: 0 falls through, nonzero parks the pc and
    /// returns `OUT_CONDITION` (the extern stashed the condition in the host).
    fn status_or_pause(
        &self,
        step: FunctionValue<'ctx>,
        frame: PointerValue<'ctx>,
        pc: usize,
        status: IntValue<'ctx>,
        label: &str,
    ) {
        let i64t = self.ctx.i64_type();
        let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.{label}ok"));
        let bad = self.ctx.append_basic_block(step, &format!("pc{pc}.{label}bad"));
        let is_ok = self
            .builder
            .build_int_compare(IntPredicate::EQ, status, i64t.const_zero(), "isok")
            .unwrap();
        self.builder.build_conditional_branch(is_ok, ok, bad).unwrap();
        self.builder.position_at_end(bad);
        self.set_pc(frame, pc as u64);
        self.ret_outcome(OUT_CONDITION);
        self.builder.position_at_end(ok);
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
        // Stack scratch for extern-call marshaling (`lt_new` supplied-fields,
        // `lt_call_foreign` args), hoisted to the entry block and shared by
        // every site. An alloca inside a loop body is never reclaimed until the
        // function returns, so per-site allocas made an allocating loop grow
        // the native stack every iteration — 200k iterations overflowed it.
        let max_words = func
            .code
            .iter()
            .filter_map(|i| match i {
                // At least one word even for a zero-arg site: the extern call
                // still receives a (never-read) base pointer.
                Instruction::New { fields, .. } | Instruction::NewVariant { fields, .. } => {
                    Some((3 * fields.len()).max(1))
                }
                Instruction::CallForeign { args, .. } => Some((2 * args.len()).max(1)),
                Instruction::CaseVariant { arms, .. } => Some(arms.len().max(1)),
                Instruction::NewArray { items, .. } => Some((2 * items.len()).max(1)),
                _ => None,
            })
            .max();
        let scratch = max_words.map(|words| {
            self.builder
                .build_alloca(i64t.array_type(words as u32), "scratch")
                .unwrap()
        });
        let has_out = func.code.iter().any(|i| {
            matches!(
                i,
                Instruction::New { .. }
                    | Instruction::NewVariant { .. }
                    | Instruction::CaseVariant { .. }
                    | Instruction::NewArray { .. }
                    | Instruction::ArrayLen { .. }
            )
        });
        let out_objid_slot =
            has_out.then(|| self.builder.build_alloca(i64t, "out.objid").unwrap());
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
                Instruction::DivI64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_I64), (*right, TAG_I64)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    // Zero divisor: stash the shared division-by-zero condition
                    // and pause — identical to the interpreter's trap.
                    let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.divok"));
                    let zero_bb = self.ctx.append_basic_block(step, &format!("pc{pc}.divzero"));
                    let nonzero = self
                        .builder
                        .build_int_compare(IntPredicate::NE, b, i64t.const_zero(), "nonzero")
                        .unwrap();
                    self.builder.build_conditional_branch(nonzero, ok, zero_bb).unwrap();
                    self.builder.position_at_end(zero_bb);
                    let trap_fn = self.module.get_function("lt_trap_div_zero").unwrap();
                    self.builder.build_call(trap_fn, &[rt.into()], "").unwrap();
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CONDITION);
                    self.builder.position_at_end(ok);
                    let r = self.builder.build_int_signed_div(a, b, "div").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_I64 as u64, false), r);
                    fallthrough();
                }
                Instruction::AddF64 { dst, left, right }
                | Instruction::SubF64 { dst, left, right }
                | Instruction::MulF64 { dst, left, right }
                | Instruction::DivF64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_F64), (*right, TAG_F64)]);
                    let f64t = self.ctx.f64_type();
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let af = self.builder.build_bit_cast(a, f64t, "af").unwrap().into_float_value();
                    let bf = self.builder.build_bit_cast(b, f64t, "bf").unwrap().into_float_value();
                    let rf = match instruction {
                        Instruction::AddF64 { .. } => self.builder.build_float_add(af, bf, "fadd").unwrap(),
                        Instruction::SubF64 { .. } => self.builder.build_float_sub(af, bf, "fsub").unwrap(),
                        Instruction::MulF64 { .. } => self.builder.build_float_mul(af, bf, "fmul").unwrap(),
                        _ => self.builder.build_float_div(af, bf, "fdiv").unwrap(),
                    };
                    let r = self.builder.build_bit_cast(rf, i64t, "fbits").unwrap().into_int_value();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_F64 as u64, false), r);
                    fallthrough();
                }
                Instruction::LtF64 { dst, left, right }
                | Instruction::LeF64 { dst, left, right }
                | Instruction::EqF64 { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_F64), (*right, TAG_F64)]);
                    let f64t = self.ctx.f64_type();
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let af = self.builder.build_bit_cast(a, f64t, "af").unwrap().into_float_value();
                    let bf = self.builder.build_bit_cast(b, f64t, "bf").unwrap().into_float_value();
                    let pred = match instruction {
                        Instruction::LtF64 { .. } => inkwell::FloatPredicate::OLT,
                        Instruction::LeF64 { .. } => inkwell::FloatPredicate::OLE,
                        _ => inkwell::FloatPredicate::OEQ,
                    };
                    let cmp = self.builder.build_float_compare(pred, af, bf, "fcmp").unwrap();
                    let r = self.builder.build_int_z_extend(cmp, i64t, "fcmpz").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_BOOL as u64, false), r);
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
                Instruction::ConcatStr { dst, left, right } => {
                    self.guard_tags(step, frame, pc, &[(*left, TAG_STR), (*right, TAG_STR)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let lt_cs = self.module.get_function("lt_concat_str").unwrap();
                    let call = self
                        .builder
                        .build_call(lt_cs, &[a.into(), b.into()], "concat")
                        .unwrap();
                    let r = call_result(call).into_int_value();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_STR as u64, false), r);
                    fallthrough();
                }
                Instruction::EqStr { dst, left, right } => {
                    // Interning dedups, so string equality is id equality —
                    // fully native, no extern.
                    self.guard_tags(step, frame, pc, &[(*left, TAG_STR), (*right, TAG_STR)]);
                    let a = self.payload_of(frame, *left);
                    let b = self.payload_of(frame, *right);
                    let eq = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, a, b, "eqstr")
                        .unwrap();
                    let r = self.builder.build_int_z_extend(eq, i64t, "eqstrz").unwrap();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_BOOL as u64, false), r);
                    fallthrough();
                }
                Instruction::New {
                    dst,
                    type_id,
                    fields,
                } => {
                    self.emit_constructor(
                        step, frame, rt, scratch, out_objid_slot, &blocks, pc, *dst, *type_id,
                        None, fields,
                    );
                }
                Instruction::NewVariant {
                    dst,
                    type_id,
                    variant,
                    fields,
                } => {
                    self.emit_constructor(
                        step, frame, rt, scratch, out_objid_slot, &blocks, pc, *dst, *type_id,
                        Some(*variant), fields,
                    );
                }
                Instruction::CaseVariant { object, arms } => {
                    // The match barrier: guard the operand tag, hand the arm
                    // list (variant ids in the entry-block scratch) to
                    // `lt_case_variant` (migrate + variant lookup + unhandled-
                    // variant trap), then switch on the returned ARM INDEX.
                    self.guard_tags(step, frame, pc, &[(*object, TAG_REF)]);
                    let n = arms.len();
                    let arr_ty = i64t.array_type(n.max(1) as u32);
                    let arr = scratch.expect("CaseVariant implies scratch space");
                    for (k, (variant, _)) in arms.iter().enumerate() {
                        let elem = unsafe {
                            self.builder
                                .build_in_bounds_gep(
                                    arr_ty,
                                    arr,
                                    &[i64t.const_zero(), i64t.const_int(k as u64, false)],
                                    "arm",
                                )
                                .unwrap()
                        };
                        self.builder
                            .build_store(elem, i64t.const_int(*variant, false))
                            .unwrap();
                    }
                    let base = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                arr_ty,
                                arr,
                                &[i64t.const_zero(), i64t.const_zero()],
                                "arms.base",
                            )
                            .unwrap()
                    };
                    let out_index = out_objid_slot.expect("CaseVariant implies an out slot");
                    let objid = self.payload_of(frame, *object);
                    let lt_cv = self.module.get_function("lt_case_variant").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_cv,
                            &[
                                rt.into(),
                                objid.into(),
                                base.into(),
                                i64t.const_int(n as u64, false).into(),
                                out_index.into(),
                            ],
                            "status",
                        )
                        .unwrap();
                    let status = call_result(call).into_int_value();
                    let ok = self.ctx.append_basic_block(step, &format!("pc{pc}.caseok"));
                    let bad = self.ctx.append_basic_block(step, &format!("pc{pc}.casebad"));
                    let is_ok = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, status, i64t.const_zero(), "caseisok")
                        .unwrap();
                    self.builder.build_conditional_branch(is_ok, ok, bad).unwrap();
                    self.builder.position_at_end(bad);
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CONDITION);
                    self.builder.position_at_end(ok);
                    let index = self
                        .builder
                        .build_load(i64t, out_index, "arm.index")
                        .unwrap()
                        .into_int_value();
                    // The extern's index is always in-bounds; the default block
                    // is unreachable by construction.
                    let unreachable_bb =
                        self.ctx.append_basic_block(step, &format!("pc{pc}.casedead"));
                    let cases: Vec<_> = arms
                        .iter()
                        .enumerate()
                        .map(|(k, (_, target))| (i64t.const_int(k as u64, false), blocks[*target]))
                        .collect();
                    self.builder.build_switch(index, unreachable_bb, &cases).unwrap();
                    self.builder.position_at_end(unreachable_bb);
                    self.builder.build_unreachable().unwrap();
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
                    // Marshal argument slots into the shared entry-block scratch
                    // as `RawSlot[n]` (flat `i64[2n]`), then call the foreign
                    // extern, which writes the result into `dst`'s slot or
                    // stashes a trap condition.
                    let n = args.len();
                    let arr_ty = i64t.array_type(2 * n as u32);
                    let arr = scratch.expect("CallForeign implies scratch space");
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
                Instruction::NewArray { dst, elem, items } => {
                    // Marshal items into the entry-block scratch as RawSlot
                    // pairs; the element type crosses the C ABI as its interned
                    // TypeId (see core `types`).
                    let n = items.len();
                    let arr_ty = i64t.array_type((2 * n).max(1) as u32);
                    let arr = scratch.expect("NewArray implies scratch space");
                    for (k, reg) in items.iter().enumerate() {
                        let slot = self.slot_ptr(frame, *reg);
                        let tag = self.load_field(slot, 0, "it.tag");
                        let payload = self.load_field(slot, 1, "it.pay");
                        for (off, v) in [(0usize, tag), (1usize, payload)] {
                            let elem_ptr = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        arr_ty,
                                        arr,
                                        &[i64t.const_zero(), i64t.const_int((2 * k + off) as u64, false)],
                                        "item",
                                    )
                                    .unwrap()
                            };
                            self.builder.build_store(elem_ptr, v).unwrap();
                        }
                    }
                    let base = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                arr_ty,
                                arr,
                                &[i64t.const_zero(), i64t.const_zero()],
                                "items.base",
                            )
                            .unwrap()
                    };
                    let out = out_objid_slot.expect("NewArray implies an out slot");
                    let type_id = livetype_core::types::intern(elem);
                    let lt_na = self.module.get_function("lt_new_array").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_na,
                            &[
                                rt.into(),
                                i64t.const_int(type_id, false).into(),
                                base.into(),
                                i64t.const_int(n as u64, false).into(),
                                out.into(),
                            ],
                            "status",
                        )
                        .unwrap();
                    self.status_or_pause(step, frame, pc, call_result(call).into_int_value(), "newarr");
                    let objid = self.builder.build_load(i64t, out, "arrid").unwrap().into_int_value();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_REF as u64, false), objid);
                    fallthrough();
                }
                Instruction::IndexGet { dst, array, index } => {
                    self.guard_tags(step, frame, pc, &[(*array, TAG_REF), (*index, TAG_I64)]);
                    let arr = self.payload_of(frame, *array);
                    let idx = self.payload_of(frame, *index);
                    let out = self.slot_ptr(frame, *dst);
                    let lt_ig = self.module.get_function("lt_index_get").unwrap();
                    let call = self
                        .builder
                        .build_call(lt_ig, &[rt.into(), arr.into(), idx.into(), out.into()], "status")
                        .unwrap();
                    self.status_or_pause(step, frame, pc, call_result(call).into_int_value(), "ixget");
                    fallthrough();
                }
                Instruction::IndexSet { array, index, value } => {
                    self.guard_tags(step, frame, pc, &[(*array, TAG_REF), (*index, TAG_I64)]);
                    let arr = self.payload_of(frame, *array);
                    let idx = self.payload_of(frame, *index);
                    let vslot = self.slot_ptr(frame, *value);
                    let vtag = self.load_field(vslot, 0, "v.tag");
                    let vpay = self.load_field(vslot, 1, "v.pay");
                    let lt_is = self.module.get_function("lt_index_set").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_is,
                            &[rt.into(), arr.into(), idx.into(), vtag.into(), vpay.into()],
                            "status",
                        )
                        .unwrap();
                    self.status_or_pause(step, frame, pc, call_result(call).into_int_value(), "ixset");
                    fallthrough();
                }
                Instruction::ArrayLen { dst, array } => {
                    self.guard_tags(step, frame, pc, &[(*array, TAG_REF)]);
                    let arr = self.payload_of(frame, *array);
                    let out = out_objid_slot.expect("ArrayLen implies an out slot");
                    let lt_al = self.module.get_function("lt_array_len").unwrap();
                    let call = self
                        .builder
                        .build_call(lt_al, &[rt.into(), arr.into(), out.into()], "status")
                        .unwrap();
                    self.status_or_pause(step, frame, pc, call_result(call).into_int_value(), "arrlen");
                    let len = self.builder.build_load(i64t, out, "len").unwrap().into_int_value();
                    self.store_slot(frame, *dst, i64t.const_int(TAG_I64 as u64, false), len);
                    fallthrough();
                }
                Instruction::ArrayPush { array, value } => {
                    self.guard_tags(step, frame, pc, &[(*array, TAG_REF)]);
                    let arr = self.payload_of(frame, *array);
                    let vslot = self.slot_ptr(frame, *value);
                    let vtag = self.load_field(vslot, 0, "v.tag");
                    let vpay = self.load_field(vslot, 1, "v.pay");
                    let lt_ap = self.module.get_function("lt_array_push").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            lt_ap,
                            &[rt.into(), arr.into(), vtag.into(), vpay.into()],
                            "status",
                        )
                        .unwrap();
                    self.status_or_pause(step, frame, pc, call_result(call).into_int_value(), "arrpush");
                    fallthrough();
                }
                Instruction::IndirectCall { .. } => {
                    // Hand back like a direct Call: the engine reads the callee
                    // register, resolves the current version, and pushes the
                    // frame. pc parks here for broken-callee pause/resume.
                    self.set_pc(frame, pc as u64);
                    self.ret_outcome(OUT_CALL);
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
        ("lt_concat_str", lt_concat_str as *const () as usize),
        ("lt_new_variant", lt_new_variant as *const () as usize),
        ("lt_case_variant", lt_case_variant as *const () as usize),
        ("lt_trap_div_zero", lt_trap_div_zero as *const () as usize),
        ("lt_new_array", lt_new_array as *const () as usize),
        ("lt_index_get", lt_index_get as *const () as usize),
        ("lt_index_set", lt_index_set as *const () as usize),
        ("lt_array_len", lt_array_len as *const () as usize),
        ("lt_array_push", lt_array_push as *const () as usize),
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

/// [`JitCode`] IS the tier source: the engine asks for the compiled-address
/// map, and the answer is "whatever the current world compiles to", cached by
/// epoch (the engine additionally snapshots it per actor, so this lock is
/// entered only after an edit). A function the codegen skips has no address
/// and therefore stays interpreted — same loop, cold tier.
impl TierSource for JitCode {
    fn native_map(&self, world: &World) -> Option<livetype_core::NativeMap> {
        Some(
            self.addrs(world)
                .unwrap_or_else(|e| panic!("JIT compilation failed: {}", e.0)),
        )
    }
}

/// The standard full engine: auto-tiering over the LLVM backend. Functions
/// start interpreted and are promoted to native code after `threshold`
/// activations (0 = compile everything on first entry).
pub fn jit_engine(threshold: u64) -> Arc<Engine> {
    Engine::new(Arc::new(JitCode::new()), threshold)
}
