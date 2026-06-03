//! Batch code emission pass using register-alloc's `Allocation`.
//!
//! Walks the dynir SSA IR and emits AArch64 machine code, using
//! pre-computed physical register assignments from the batch allocator.
//! No streaming regalloc — all allocation decisions are final before
//! code emission begins.

use dynasm::buffer::{CodeBuffer, Label};
use dynir::ir::*;
use dynir::types::Type;

use regalloc::allocator::{Allocation, InsertedMove, MoveOperand, MovePosition};
use regalloc::ir::Function as RegallocFunction;
use regalloc::types::*;

use crate::backend::{
    Arm64Backend, LoweringBackend, MachineFpBinOp, MachineGpBinOp, MachineReg, MachineWordSize,
};
use crate::regalloc::{machine_fp, machine_gp};
use crate::regalloc_bridge::{AArch64Target, DynIRFunction, FP, GP, fp_preg, gp_preg};

use dynasm::arm64::reloc::Arm64;

// ── PReg → MachineReg conversion ──────────────────────────────────

fn preg_to_machine(preg: PReg) -> MachineReg {
    if preg.0 < 32 {
        machine_gp(preg.0 as u8)
    } else {
        machine_fp((preg.0 - 32) as u8)
    }
}

fn preg_to_gp_index(preg: PReg) -> u8 {
    debug_assert!(preg.0 < 32, "expected GP preg, got {:?}", preg);
    preg.0 as u8
}

fn preg_to_fp_index(preg: PReg) -> u8 {
    debug_assert!(preg.0 >= 32, "expected FP preg, got {:?}", preg);
    (preg.0 - 32) as u8
}

// ── Batch emitter ─────────────────────────────────────────────────

/// Tag scheme configuration for the batch emitter.
pub struct TagConfig {
    pub has_unboxed_float: bool,
    pub payload_bits: u8,
    pub tag_count: u32,
    /// Closure: (tag) → encoded tag pattern (with zero payload)
    pub encode_tagged: fn(u32, u64) -> u64,
}

/// Emit machine code for one function using a pre-computed `Allocation`.
///
/// This replaces the streaming `Lowerer` for functions compiled with
/// the batch register allocator.
pub struct BatchEmitter<'a> {
    func: &'a Function,
    adapted: &'a DynIRFunction<'a>,
    alloc: &'a Allocation,
    buf: CodeBuffer<Arm64>,
    /// Block labels for branch targets
    block_labels: Vec<Label>,
    /// Spill slot base offset from FP (negative)
    spill_base_offset: i32,
    /// Frame size (computed after emission)
    frame_size: u32,
    /// External function pointers
    externs: &'a [*const u8],
    /// Call table base address (for internal function calls)
    call_table_base: u64,
    /// Tag scheme configuration
    tags: TagConfig,
    /// Callee-saved registers used by this function (saved in prologue, restored before ret)
    callee_saved_used: Vec<PReg>,
    /// Base offset for the safepoint-specific callee-saved spill area (separate from
    /// the prologue/epilogue save area to avoid overwriting caller's register values).
    safepoint_callee_save_base: i32,
    /// Epilogue patch offsets (for frame size patching)
    epilogue_offsets: Vec<usize>,
    /// GC safepoint handler function pointer (if GC is enabled)
    safepoint_handler: Option<u64>,
    /// Inline cache array base address (for InvokeDynamic fast path)
    ic_base: u64,
    /// Slow-path invoke extern: fn(receiver: u64, symbol_id: u64, num_args: u64, ic_entry_ptr: u64) -> u64
    /// Returns the closure value. Updates the IC entry at ic_entry_ptr.
    slow_invoke: u64,
    /// Offset of the class field within Instance objects (for IC type check)
    instance_class_offset: i32,
    /// Collected safepoint records (precise stack map slots)
    safepoints: Vec<crate::SafepointRecord>,
    /// Number of safepoint records that precede this function's records in
    /// the flat global safepoint array passed to the JIT runtime. Inst::
    /// Safepoint emission uses `base + self.safepoints.len()` as the
    /// payload so the runtime's index into the flat array points to this
    /// function's record, not whatever record happens to be at the same
    /// per-function offset in another function.
    safepoint_index_base: usize,
}

impl<'a> BatchEmitter<'a> {
    pub fn new(
        func: &'a Function,
        adapted: &'a DynIRFunction<'a>,
        alloc: &'a Allocation,
        externs: &'a [*const u8],
        call_table_base: u64,
        tags: TagConfig,
        safepoint_handler: Option<u64>,
        ic_base: u64,
        slow_invoke: u64,
        instance_class_offset: i32,
        safepoint_index_base: usize,
    ) -> Self {
        let n_blocks = func.blocks.len();
        let mut buf = CodeBuffer::new();
        let block_labels: Vec<Label> = (0..n_blocks).map(|_| buf.create_label()).collect();

        // Frame layout (all positive offsets from FP):
        // [FP+0]: saved FP, [FP+8]: saved LR
        // [FP+16..]: callee-saved regs, spill slots, stack slots
        // Initial offset overridden in emit() after callee-saved count is known.
        let spill_base_offset = 16; // placeholder, updated in emit()

        BatchEmitter {
            func,
            adapted,
            alloc,
            buf,
            block_labels,
            spill_base_offset,
            frame_size: 0,
            externs,
            call_table_base,
            tags,
            callee_saved_used: Vec::new(),
            safepoint_callee_save_base: 0,
            epilogue_offsets: Vec::new(),
            safepoint_handler,
            ic_base,
            slow_invoke,
            instance_class_offset,
            safepoints: Vec::new(),
            safepoint_index_base,
        }
    }

    /// Determine which callee-saved GP registers are used by the allocation.
    fn used_callee_saved(&self) -> Vec<PReg> {
        let callee_saved: &[PReg] = &[
            PReg(19),
            PReg(20),
            PReg(21),
            PReg(22),
            PReg(23),
            PReg(24),
            PReg(25),
            PReg(26),
            PReg(27),
        ];
        let mut used = Vec::new();
        for &preg in callee_saved {
            // Check if any instruction operand or move uses this register
            let is_used =
                self.alloc.inst_allocs.values().any(|&p| p == preg)
                    || self.alloc.moves.iter().any(|m| {
                        m.from == MoveOperand::Reg(preg) || m.to == MoveOperand::Reg(preg)
                    });
            if is_used {
                used.push(preg);
            }
        }
        used
    }

    /// Emit the full function. Returns (CodeBuffer, frame_size, safepoint_records).
    pub fn emit(mut self) -> (CodeBuffer<Arm64>, u32, Vec<crate::SafepointRecord>) {
        // Determine callee-saved registers to preserve
        let callee_saved = self.used_callee_saved();
        self.callee_saved_used = callee_saved.clone();

        // Prologue (patch frame size later)
        let prologue_offset = Arm64Backend::emit_prologue(&mut self.buf);

        // Save callee-saved registers within the frame at positive offsets from FP.
        // Frame layout: [FP+0]=saved_FP, [FP+8]=saved_LR, [FP+16..]=callee-saved,
        // then spill slots, then stack slots. Negative offsets from FP are BELOW SP
        // and get overwritten by callee stack frames during function calls.
        let callee_save_base = 16i32; // first callee-save slot at FP+16
        for (i, &preg) in callee_saved.iter().enumerate() {
            let offset = callee_save_base + (i as i32) * 8;
            Arm64Backend::emit_store_gp(
                &mut self.buf,
                preg_to_machine(preg),
                machine_gp(29),
                offset,
                MachineWordSize::W64,
            );
        }

        // Frame layout (positive offsets from FP):
        //   [FP+0]: saved FP, [FP+8]: saved LR
        //   [FP+16..]: callee-saved saves (prologue/epilogue — caller's values, never overwritten)
        //   [FP+16+N*8..]: safepoint callee-saved area (GC spills current register values)
        //   [FP+16+2*N*8..]: spill slots, stack slots
        let callee_save_bytes = callee_saved.len() as u32 * 8;
        self.safepoint_callee_save_base = 16 + callee_save_bytes as i32;
        let safepoint_callee_save_bytes = callee_save_bytes; // same count of registers
        self.spill_base_offset = 16 + callee_save_bytes as i32 + safepoint_callee_save_bytes as i32;

        // Compute frame size: FP+LR + callee-saved saves + safepoint callee-saved + spill slots + stack slots
        // + InvokeDynamic scratch (16 bytes for receiver/closure save if any InvokeDynamic exists)
        let num_spills = self.alloc.num_spill_slots;
        let stack_slot_bytes: u32 = self.func.stack_slots.iter().map(|s| s.size).sum();
        let has_invoke_dynamic = self.func.blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|n| matches!(&n.inst, Inst::InvokeDynamic { .. }))
        });
        let invoke_scratch_bytes: u32 = if has_invoke_dynamic { 16 } else { 0 };
        let raw_frame = 16 /* FP+LR */ + callee_save_bytes + safepoint_callee_save_bytes
            + num_spills * 8 + stack_slot_bytes + invoke_scratch_bytes;
        self.frame_size = (raw_frame + 15) & !15; // 16-byte aligned

        // Emit blocks
        for (block_idx, block) in self.func.blocks.iter().enumerate() {
            self.emit_block(block_idx, block);
        }

        // Patch frame size
        Arm64Backend::emit_frame_size_patch(
            &mut self.buf,
            prologue_offset,
            &self.epilogue_offsets,
            self.frame_size as i32,
        );

        (self.buf, self.frame_size, self.safepoints)
    }

    fn emit_block(&mut self, block_idx: usize, block: &Block) {
        // Bind block label
        let label = self.block_labels[block_idx];
        self.buf.bind_label(label);

        // Emit each instruction
        let base = self.adapted.block_inst_base(block_idx);
        for (local_idx, inst_node) in block.insts.iter().enumerate() {
            let inst_id = InstId(base + local_idx as u32);
            self.emit_before_moves(inst_id);
            self.emit_inst(inst_id, inst_node, block_idx);
            self.emit_after_moves(inst_id);
        }

        // Emit block-edge moves BEFORE the terminator (they shuffle
        // values into the successor's block param registers)
        self.emit_edge_moves_from(block_idx);

        // Terminator
        let term_id = InstId(base + block.insts.len() as u32);
        self.emit_before_moves(term_id);
        self.emit_terminator(term_id, &block.terminator, block_idx);
        self.emit_after_moves(term_id);
    }

    // ── Move emission ─────────────────────────────────────────

    fn emit_before_moves(&mut self, inst: InstId) {
        let moves: Vec<_> = self
            .alloc
            .moves
            .iter()
            .filter(|m| m.at == MovePosition::Before(inst))
            .cloned()
            .collect();
        self.emit_parallel_moves(&moves);
    }

    fn emit_after_moves(&mut self, inst: InstId) {
        let moves: Vec<_> = self
            .alloc
            .moves
            .iter()
            .filter(|m| m.at == MovePosition::After(inst))
            .cloned()
            .collect();
        self.emit_parallel_moves(&moves);
    }

    fn emit_edge_moves_from(&mut self, from_block: usize) {
        let from = BlockId(from_block as u32);
        let moves: Vec<_> = self
            .alloc
            .moves
            .iter()
            .filter(|m| matches!(&m.at, MovePosition::BlockEdge { from: f, .. } if *f == from))
            .cloned()
            .collect();
        self.emit_parallel_moves(&moves);
    }

    /// Emit a set of parallel moves without clobbering.
    ///
    /// Uses a simple strategy: emit moves whose destination isn't a source
    /// of another move first, then handle remaining (cycles) via X28 as scratch.
    fn emit_parallel_moves(&mut self, moves: &[InsertedMove]) {
        if moves.len() <= 1 {
            for m in moves {
                self.emit_move(m);
            }
            return;
        }

        // Build set of sources
        let sources: Vec<_> = moves.iter().map(|m| m.from).collect();
        let mut emitted = vec![false; moves.len()];
        let mut progress = true;

        // Emit moves whose destination isn't a source of another pending move
        while progress {
            progress = false;
            for i in 0..moves.len() {
                if emitted[i] {
                    continue;
                }
                let dst_is_source = sources
                    .iter()
                    .enumerate()
                    .any(|(j, src)| !emitted[j] && i != j && *src == moves[i].to);
                if !dst_is_source {
                    self.emit_move(&moves[i]);
                    emitted[i] = true;
                    progress = true;
                }
            }
        }

        // Remaining moves form cycles — break with scratch register X28
        for i in 0..moves.len() {
            if emitted[i] {
                continue;
            }
            // Save dst to scratch, emit move, restore from scratch
            // For simplicity, use X28 as scratch (it's reserved)
            let scratch = MoveOperand::Reg(gp_preg(28));
            // Save whatever is in dst to scratch
            self.emit_move(&InsertedMove {
                at: moves[i].at.clone(),
                from: moves[i].to,
                to: scratch,
                class: moves[i].class,
            });
            self.emit_move(&moves[i]);
            emitted[i] = true;
            // Now emit any move that sources from the original dst
            for j in 0..moves.len() {
                if emitted[j] {
                    continue;
                }
                if moves[j].from == moves[i].to {
                    // This move's source was clobbered — use scratch instead
                    self.emit_move(&InsertedMove {
                        at: moves[j].at.clone(),
                        from: scratch,
                        to: moves[j].to,
                        class: moves[j].class,
                    });
                    emitted[j] = true;
                }
            }
        }
    }

    fn emit_move(&mut self, m: &InsertedMove) {
        match (&m.from, &m.to) {
            (MoveOperand::Reg(src), MoveOperand::Reg(dst)) => {
                if m.class == GP {
                    Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        preg_to_machine(*dst),
                        preg_to_machine(*src),
                    );
                } else {
                    // FP move - use gp_move for now (same encoding on ARM64 for 64-bit)
                    Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        preg_to_machine(*dst),
                        preg_to_machine(*src),
                    );
                }
            }
            (MoveOperand::Reg(src), MoveOperand::SpillSlot(slot)) => {
                // Spill: reg → stack
                let offset = self.spill_offset(*slot);
                if m.class == GP {
                    Arm64Backend::emit_store_gp(
                        &mut self.buf,
                        preg_to_machine(*src),
                        machine_gp(29), // FP
                        offset,
                        MachineWordSize::W64,
                    );
                } else {
                    Arm64Backend::emit_store_fp(
                        &mut self.buf,
                        preg_to_machine(*src),
                        machine_gp(29),
                        offset,
                    );
                }
            }
            (MoveOperand::SpillSlot(slot), MoveOperand::Reg(dst)) => {
                // Reload: stack → reg
                let offset = self.spill_offset(*slot);
                if m.class == GP {
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        preg_to_machine(*dst),
                        machine_gp(29), // FP
                        offset,
                        MachineWordSize::W64,
                    );
                } else {
                    Arm64Backend::emit_load_fp(
                        &mut self.buf,
                        preg_to_machine(*dst),
                        machine_gp(29),
                        offset,
                    );
                }
            }
            (MoveOperand::SpillSlot(src_slot), MoveOperand::SpillSlot(dst_slot)) => {
                // Stack-to-stack: load to scratch X28, then store
                let src_off = self.spill_offset(*src_slot);
                let dst_off = self.spill_offset(*dst_slot);
                if m.class == GP {
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(28),
                        machine_gp(29),
                        src_off,
                        MachineWordSize::W64,
                    );
                    Arm64Backend::emit_store_gp(
                        &mut self.buf,
                        machine_gp(28),
                        machine_gp(29),
                        dst_off,
                        MachineWordSize::W64,
                    );
                } else {
                    Arm64Backend::emit_load_fp(
                        &mut self.buf,
                        preg_to_machine(fp_preg(0)),
                        machine_gp(29),
                        src_off,
                    );
                    Arm64Backend::emit_store_fp(
                        &mut self.buf,
                        preg_to_machine(fp_preg(0)),
                        machine_gp(29),
                        dst_off,
                    );
                }
            }
            (MoveOperand::Remat(imm), MoveOperand::Reg(dst)) => {
                // Rematerialize: emit MOV immediate → register
                Arm64Backend::emit_mov_imm(&mut self.buf, preg_to_machine(*dst), *imm);
            }
            (MoveOperand::Remat(imm), MoveOperand::SpillSlot(slot)) => {
                // Remat to stack: materialize into scratch, then store
                let offset = self.spill_offset(*slot);
                Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(28), *imm);
                Arm64Backend::emit_store_gp(
                    &mut self.buf,
                    machine_gp(28),
                    machine_gp(29),
                    offset,
                    MachineWordSize::W64,
                );
            }
            (_, MoveOperand::Remat(_)) => {
                // Can't move INTO a remat — this shouldn't happen
                panic!("cannot move into a Remat operand");
            }
        }
    }

    fn restore_callee_saved(&mut self) {
        let callee_save_base = 16i32;
        for (i, &preg) in self.callee_saved_used.iter().enumerate() {
            let offset = callee_save_base + (i as i32) * 8;
            Arm64Backend::emit_load_gp(
                &mut self.buf,
                preg_to_machine(preg),
                machine_gp(29),
                offset,
                MachineWordSize::W64,
            );
        }
    }

    fn spill_offset(&self, slot: SpillSlot) -> i32 {
        // Spill slots are at positive offsets from FP, after callee-saved saves
        self.spill_base_offset + (slot.0 as i32) * 8
    }

    /// Push a SafepointRecord for the return address right after a Call/
    /// CallIndirect/InvokeDynamic. Used by the FP-chain walker if GC fires
    /// inside the callee — we need to know the caller's live roots so the
    /// collector can update them when forwarding moving heap pointers.
    ///
    /// The regalloc has already spilled live values to slots (via
    /// `SafepointAction::SpillAndRecord` from `DynIRFunction::safepoint_action`)
    /// because every call is a safepoint. `alloc.stackmaps[inst_id]`
    /// enumerates the slots; we record exactly those, plus `is_gc_root`
    /// stack slots.
    fn record_call_return_safepoint(&mut self, inst_id: InstId) {
        if self.safepoint_handler.is_none() {
            return;
        }
        let return_offset = self.buf.current_offset();
        let root_slots = self.collect_live_root_slots(inst_id);
        self.safepoints.push(crate::SafepointRecord {
            code_offset: return_offset,
            return_offset,
            root_slots,
        });
    }

    /// Precise root-slot collection for a safepoint instruction.
    ///
    /// The regalloc has already arranged (via `SafepointAction::SpillAndRecord`
    /// in `DynIRFunction::safepoint_action`) for every live vreg at this
    /// safepoint to be in a spill slot. `alloc.stackmaps[inst_id]` enumerates
    /// each live vreg's slot. We add the function's `is_gc_root` stack
    /// slots — those are the user-declared GC roots (e.g. local
    /// variables defined via `def_var` in dynlang).
    ///
    /// Crucially: we do *not* iterate all `num_spill_slots` (which would
    /// include dead/unallocated slots), nor all stack slots (which would
    /// include non-GC scratch slots), nor all callee-saved register
    /// banks (which would include preserved caller state). Only what the
    /// regalloc says is actually live, plus declared GC roots.
    fn collect_live_root_slots(&self, inst_id: InstId) -> Vec<i32> {
        let mut root_slots: Vec<i32> = Vec::new();
        if let Some(entries) = self.alloc.stackmaps.get(&inst_id) {
            for entry in entries {
                if let MoveOperand::SpillSlot(slot) = entry.location {
                    root_slots.push(self.spill_offset(slot));
                }
            }
        }
        // is_gc_root stack slots (declared GC roots, e.g. dynlang `def_var`).
        let spill_bytes = self.alloc.num_spill_slots * 8;
        let mut ss_offset = self.spill_base_offset + spill_bytes as i32;
        for ss in &self.func.stack_slots {
            if ss.is_gc_root {
                root_slots.push(ss_offset);
            }
            ss_offset += ss.size as i32;
        }
        root_slots.sort_unstable();
        root_slots.dedup();
        root_slots
    }

    // ── Instruction emission ──────────────────────────────────

    fn emit_inst(&mut self, inst_id: InstId, node: &InstNode, _block_idx: usize) {
        // Operand 0 is always the def (result) if present.
        // Operands 1+ are uses, in the order defined by compute_inst_operands.
        let get_def =
            |alloc: &Allocation| -> PReg { alloc.get(inst_id, 0).expect("missing def allocation") };
        let get_use = |alloc: &Allocation, idx: usize| -> PReg {
            alloc.get(inst_id, idx).expect("missing use allocation")
        };

        // Skip constant definitions that are rematerialized — they'll be
        // re-emitted as MOV immediates before each use by Remat moves.
        if let Some(val) = node.value {
            let vreg = VReg(val.index() as u32);
            if let Some(MoveOperand::Remat(_)) = self.alloc.vreg_homes.get(&vreg) {
                return;
            }
        }

        match &node.inst {
            Inst::Iconst(_ty, imm) => {
                let dst = get_def(&self.alloc);
                Arm64Backend::emit_mov_imm(&mut self.buf, preg_to_machine(dst), *imm as u64);
            }

            Inst::GcLiteral(_) => {
                // The batch register allocator path doesn't support GcLiteral
                // yet — frontends that need a moving GC use the greedy /
                // linear-scan path through `JitModule::extend`.
                unimplemented!("GcLiteral not yet supported in batch_lower");
            }

            Inst::F64Const(f) => {
                let dst = get_def(&self.alloc);
                // Load float bits into GP scratch, then move to FP reg
                let bits = f.to_bits();
                Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(28), bits);
                Arm64Backend::emit_gp_to_fp_move(
                    &mut self.buf,
                    preg_to_machine(dst),
                    machine_gp(28),
                );
            }

            Inst::Add(_, _)
            | Inst::Sub(_, _)
            | Inst::Mul(_, _)
            | Inst::SDiv(_, _)
            | Inst::UDiv(_, _)
            | Inst::And(_, _)
            | Inst::Or(_, _)
            | Inst::Xor(_, _)
            | Inst::Shl(_, _)
            | Inst::LShr(_, _)
            | Inst::AShr(_, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                let ty = node
                    .value
                    .map(|v| self.func.value_type(v))
                    .unwrap_or(Type::I64);
                let size = type_to_word_size(ty);
                let op = match &node.inst {
                    Inst::Add(_, _) => MachineGpBinOp::Add,
                    Inst::Sub(_, _) => MachineGpBinOp::Sub,
                    Inst::Mul(_, _) => MachineGpBinOp::Mul,
                    Inst::SDiv(_, _) => MachineGpBinOp::SDiv,
                    Inst::UDiv(_, _) => MachineGpBinOp::UDiv,
                    Inst::And(_, _) => MachineGpBinOp::And,
                    Inst::Or(_, _) => MachineGpBinOp::Or,
                    Inst::Xor(_, _) => MachineGpBinOp::Xor,
                    Inst::Shl(_, _) => MachineGpBinOp::Shl,
                    Inst::LShr(_, _) => MachineGpBinOp::LShr,
                    Inst::AShr(_, _) => MachineGpBinOp::AShr,
                    _ => unreachable!(),
                };
                Arm64Backend::emit_gp_binop(
                    &mut self.buf,
                    op,
                    preg_to_machine(dst),
                    preg_to_machine(lhs),
                    preg_to_machine(rhs),
                    size,
                );
            }

            Inst::FAdd(_, _) | Inst::FSub(_, _) | Inst::FMul(_, _) | Inst::FDiv(_, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                let op = match &node.inst {
                    Inst::FAdd(_, _) => MachineFpBinOp::Add,
                    Inst::FSub(_, _) => MachineFpBinOp::Sub,
                    Inst::FMul(_, _) => MachineFpBinOp::Mul,
                    Inst::FDiv(_, _) => MachineFpBinOp::Div,
                    _ => unreachable!(),
                };
                Arm64Backend::emit_fp_binop(
                    &mut self.buf,
                    op,
                    preg_to_machine(dst),
                    preg_to_machine(lhs),
                    preg_to_machine(rhs),
                );
            }

            Inst::Neg(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let ty = node
                    .value
                    .map(|v| self.func.value_type(v))
                    .unwrap_or(Type::I64);
                Arm64Backend::emit_gp_neg(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                    type_to_word_size(ty),
                );
            }
            Inst::Not(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let ty = node
                    .value
                    .map(|v| self.func.value_type(v))
                    .unwrap_or(Type::I64);
                Arm64Backend::emit_gp_not(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                    type_to_word_size(ty),
                );
            }
            Inst::FNeg(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                Arm64Backend::emit_fp_neg(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                );
            }

            Inst::Icmp(op, _, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                Arm64Backend::emit_icmp_set(
                    &mut self.buf,
                    *op,
                    preg_to_machine(dst),
                    preg_to_machine(lhs),
                    preg_to_machine(rhs),
                    MachineWordSize::W64,
                );
            }

            Inst::Fcmp(op, _, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                Arm64Backend::emit_fcmp_set(
                    &mut self.buf,
                    *op,
                    preg_to_machine(dst),
                    preg_to_machine(lhs),
                    preg_to_machine(rhs),
                );
            }

            Inst::Load(ty, _, offset) => {
                let dst = get_def(&self.alloc);
                let base = get_use(&self.alloc, 1);
                if *ty == Type::F64 {
                    Arm64Backend::emit_load_fp(
                        &mut self.buf,
                        preg_to_machine(dst),
                        preg_to_machine(base),
                        *offset,
                    );
                } else {
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        preg_to_machine(dst),
                        preg_to_machine(base),
                        *offset,
                        type_to_word_size(*ty),
                    );
                }
            }

            Inst::Store(_, _, offset) => {
                // Store has no def. Operands: [val, addr]
                let val = get_use(&self.alloc, 0);
                let addr = get_use(&self.alloc, 1);
                // Determine if float store by checking value type
                Arm64Backend::emit_store_gp(
                    &mut self.buf,
                    preg_to_machine(val),
                    preg_to_machine(addr),
                    *offset,
                    MachineWordSize::W64,
                );
            }

            Inst::StackAddr(slot) => {
                let dst = get_def(&self.alloc);
                // Compute address: dst = FP + slot_offset
                let slot_offset = self.stack_slot_offset(*slot);
                // Use ADD/SUB depending on sign
                if slot_offset >= 0 {
                    Arm64Backend::emit_gp_binop(
                        &mut self.buf,
                        MachineGpBinOp::Add,
                        preg_to_machine(dst),
                        machine_gp(29),
                        machine_gp(29), // will be overridden by immediate
                        MachineWordSize::W64,
                    );
                    // Actually, we need add-immediate. Use mov_imm + add for now.
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        preg_to_machine(dst),
                        slot_offset as u64,
                    );
                    Arm64Backend::emit_gp_binop(
                        &mut self.buf,
                        MachineGpBinOp::Add,
                        preg_to_machine(dst),
                        machine_gp(29),
                        preg_to_machine(dst),
                        MachineWordSize::W64,
                    );
                } else {
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        preg_to_machine(dst),
                        (-slot_offset) as u64,
                    );
                    Arm64Backend::emit_gp_binop(
                        &mut self.buf,
                        MachineGpBinOp::Sub,
                        preg_to_machine(dst),
                        machine_gp(29),
                        preg_to_machine(dst),
                        MachineWordSize::W64,
                    );
                }
            }

            Inst::Bitcast(src_v, _) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let dst_ty = node
                    .value
                    .map(|v| self.func.value_type(v))
                    .unwrap_or(Type::I64);
                let src_ty = self.func.value_type(*src_v);
                match (src_ty == Type::F64, dst_ty == Type::F64) {
                    (false, true) => Arm64Backend::emit_gp_to_fp_move(
                        &mut self.buf,
                        preg_to_machine(dst),
                        preg_to_machine(src),
                    ),
                    (true, false) => Arm64Backend::emit_fp_to_gp_move(
                        &mut self.buf,
                        preg_to_machine(dst),
                        preg_to_machine(src),
                    ),
                    // Same register class (GP→GP for I64↔GcPtr↔Ptr, or FP→FP) — just a move.
                    _ => Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        preg_to_machine(dst),
                        preg_to_machine(src),
                    ),
                }
            }

            Inst::Sext(_, _) | Inst::Zext(_, _) | Inst::Trunc(_, _) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                // For now, just mov (most conversions between i8/i32/i64 are trivial on arm64)
                Arm64Backend::emit_gp_move(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                );
            }

            Inst::Select(_, _, _) => {
                let dst = get_def(&self.alloc);
                let cond = get_use(&self.alloc, 1);
                let if_true = get_use(&self.alloc, 2);
                let if_false = get_use(&self.alloc, 3);
                Arm64Backend::emit_gp_select(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(cond),
                    preg_to_machine(if_true),
                    preg_to_machine(if_false),
                    MachineWordSize::W64,
                );
            }

            Inst::Payload(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                <Arm64Backend as LoweringBackend>::emit_extract_payload(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                    self.tags.has_unboxed_float,
                    self.tags.payload_bits,
                );
            }

            Inst::IsTag(_, tag) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let expected = if self.tags.has_unboxed_float {
                    (self.tags.encode_tagged)(*tag, 0) >> self.tags.payload_bits
                } else {
                    *tag as u64
                };
                <Arm64Backend as LoweringBackend>::emit_is_tag(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                    self.tags.has_unboxed_float,
                    self.tags.payload_bits,
                    self.tags.tag_count as u64 - 1,
                    expected,
                );
            }

            Inst::MakeTagged(tag, _) => {
                let dst = get_def(&self.alloc);
                let payload = get_use(&self.alloc, 1);
                <Arm64Backend as LoweringBackend>::emit_make_tagged(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(payload),
                    self.tags.has_unboxed_float,
                    self.tags.payload_bits,
                    (self.tags.encode_tagged)(*tag, 0),
                    *tag as u64,
                );
            }

            Inst::TagOf(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                <Arm64Backend as LoweringBackend>::emit_tag_of(
                    &mut self.buf,
                    preg_to_machine(dst),
                    preg_to_machine(src),
                    self.tags.has_unboxed_float,
                    self.tags.payload_bits,
                    self.tags.tag_count as u64 - 1,
                );
            }

            Inst::Call(fref, args) => {
                self.emit_call(inst_id, *fref, args, node.value);
                self.record_call_return_safepoint(inst_id);
            }

            Inst::CallIndirect(_callee_val, args, _ret_ty) => {
                self.emit_call_indirect(inst_id, args, node.value);
                self.record_call_return_safepoint(inst_id);
            }

            Inst::IntToFloat(_) | Inst::FloatToInt(_) => {
                todo!("Int/Float conversions in batch lowerer");
            }

            Inst::OverflowCheck(_, _, _) | Inst::Guard(_, _, _) => {
                todo!("Overflow/Guard in batch lowerer");
            }

            Inst::InvokeDynamic {
                receiver,
                symbol,
                args,
                cache_id,
            } => {
                let def_preg = self
                    .alloc
                    .get(inst_id, 0)
                    .expect("missing InvokeDynamic def");
                let recv_preg = self.alloc.get(inst_id, 1).expect("missing receiver alloc");

                if self.ic_base == 0 || self.slow_invoke == 0 {
                    // No inline cache wired up — fall back to invoke_fast + invoke_func_ptr.
                    // Use the Before/After moves that the register allocator already set up
                    // (receiver is in X0 from FixedReg constraint, args in X1+).
                    // The invoke_fast extern: (receiver, name_id, num_args) → closure
                    // We need to call it, then invoke_func_ptr, then call_indirect.
                    // BUT: we can't safely save values between calls here without
                    // the allocator's help. So use a spill slot on the stack.

                    // Scratch stack area: use frame space past all spill slots + stack slots.
                    // Two scratch slots for saving receiver and closure across calls.
                    let stack_slots_bytes: i32 =
                        self.func.stack_slots.iter().map(|s| s.size as i32).sum();
                    let scratch_base = self.spill_base_offset
                        + (self.alloc.num_spill_slots as i32) * 8
                        + stack_slots_bytes;
                    let recv_save = scratch_base; // FP + scratch_base
                    let closure_save = scratch_base + 8; // FP + scratch_base + 8

                    // Save receiver to stack (it will be clobbered by calls)
                    Arm64Backend::emit_store_gp(
                        &mut self.buf,
                        preg_to_machine(recv_preg),
                        machine_gp(29),
                        recv_save,
                        MachineWordSize::W64,
                    );

                    // Call invoke_fast(receiver, symbol_id, num_args)
                    // receiver is already in recv_preg. The Before moves put it in X0 (FixedReg).
                    // But we also need to set X1 and X2.
                    Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        machine_gp(0),
                        preg_to_machine(recv_preg),
                    );
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(1),
                        symbol.as_u32() as u64,
                    );
                    Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(2), args.len() as u64);
                    if let Some(idx) = self
                        .func
                        .extern_funcs
                        .iter()
                        .position(|ef| ef.name == "lox_invoke_fast")
                    {
                        Arm64Backend::emit_mov_imm(
                            &mut self.buf,
                            machine_gp(27),
                            self.call_table_base,
                        );
                        Arm64Backend::emit_load_gp(
                            &mut self.buf,
                            machine_gp(28),
                            machine_gp(27),
                            (idx * 8) as i32,
                            MachineWordSize::W64,
                        );
                        Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
                        // Post-BLR safepoint record: GC could fire inside
                        // lox_invoke_fast (it may allocate when promoting
                        // a method to a bound closure).
                        self.record_call_return_safepoint(inst_id);
                    }
                    // X0 = closure. Save it.
                    Arm64Backend::emit_store_gp(
                        &mut self.buf,
                        machine_gp(0),
                        machine_gp(29),
                        closure_save,
                        MachineWordSize::W64,
                    );

                    // Call invoke_func_ptr(closure) — X0 still has closure
                    if let Some(idx) = self
                        .func
                        .extern_funcs
                        .iter()
                        .position(|ef| ef.name == "lox_invoke_func_ptr")
                    {
                        Arm64Backend::emit_mov_imm(
                            &mut self.buf,
                            machine_gp(27),
                            self.call_table_base,
                        );
                        Arm64Backend::emit_load_gp(
                            &mut self.buf,
                            machine_gp(28),
                            machine_gp(27),
                            (idx * 8) as i32,
                            MachineWordSize::W64,
                        );
                        Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
                        // Pure load on closure object — shouldn't allocate
                        // — but conservatively record so the FP-chain walker
                        // never lands on an unrecorded return address.
                        self.record_call_return_safepoint(inst_id);
                    }
                    // X0 = func_ptr
                    Arm64Backend::emit_gp_move(&mut self.buf, machine_gp(28), machine_gp(0));
                    // Reload closure → X0
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(0),
                        machine_gp(29),
                        closure_save,
                        MachineWordSize::W64,
                    );
                    // Reload receiver → X1
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(1),
                        machine_gp(29),
                        recv_save,
                        MachineWordSize::W64,
                    );
                    // User args → X2+
                    for (i, &_arg) in args.iter().enumerate() {
                        let arg_preg = self.alloc.get(inst_id, 2 + i).unwrap_or(gp_preg(0));
                        Arm64Backend::emit_gp_move(
                            &mut self.buf,
                            machine_gp((i + 2) as u8),
                            preg_to_machine(arg_preg),
                        );
                    }
                    // Call the resolved func_ptr
                    Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
                    // Post-BLR safepoint record: the user closure body can
                    // freely allocate; GC may fire deep inside.
                    self.record_call_return_safepoint(inst_id);
                    // Result in X0
                    if def_preg != gp_preg(0) {
                        Arm64Backend::emit_gp_move(
                            &mut self.buf,
                            preg_to_machine(def_preg),
                            machine_gp(0),
                        );
                    }
                } else {
                    // Inline-cached dynamic dispatch.
                    //
                    // Fast path (~7 instructions):
                    //   1. Load receiver's class from instance header
                    //   2. Compare against cached class_id in IC entry
                    //   3. On match: load cached closure + func_ptr, call directly
                    //
                    // Slow path:
                    //   Call extern to do full lookup, update IC, then call result

                    // IC entry is 24 bytes: [cached_class_id: u64, cached_value: u64, cached_func_ptr: u64]
                    let ic_offset = (*cache_id as i64) * 24;

                    let hit_label = self.buf.create_label();
                    let miss_label = self.buf.create_label();
                    let done_label = self.buf.create_label();

                    // ── Fast path: check inline cache ──

                    // Load receiver's class: extract pointer, load class field
                    // receiver is NaN-boxed, extract payload (lower 48 bits)
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(28),
                        0x0000_FFFF_FFFF_FFFF,
                    );
                    Arm64Backend::emit_gp_binop(
                        &mut self.buf,
                        MachineGpBinOp::And,
                        machine_gp(28),
                        preg_to_machine(recv_preg),
                        machine_gp(28),
                        MachineWordSize::W64,
                    );
                    // X28 = raw pointer to instance object
                    // Load class field (a NaN-boxed value at known offset)
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(0),
                        machine_gp(28),
                        self.instance_class_offset,
                        MachineWordSize::W64,
                    );
                    // X0 = instance's class (NaN-boxed)

                    // Load IC entry's cached_class_id
                    Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(28), self.ic_base);
                    if ic_offset != 0 {
                        Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(1), ic_offset as u64);
                        Arm64Backend::emit_gp_binop(
                            &mut self.buf,
                            MachineGpBinOp::Add,
                            machine_gp(28),
                            machine_gp(28),
                            machine_gp(1),
                            MachineWordSize::W64,
                        );
                    }
                    // X28 = pointer to IC entry
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(1),
                        machine_gp(28),
                        0,
                        MachineWordSize::W64,
                    );
                    // X1 = cached_class_id

                    // Compare class with cached
                    Arm64Backend::emit_icmp_set(
                        &mut self.buf,
                        dynir::ir::CmpOp::Eq,
                        machine_gp(2),
                        machine_gp(0),
                        machine_gp(1),
                        MachineWordSize::W64,
                    );
                    Arm64Backend::emit_cbnz_to_label(&mut self.buf, machine_gp(2), hit_label);
                    Arm64Backend::emit_branch_to_label(&mut self.buf, miss_label);

                    // ── Cache hit: load closure + func_ptr, call directly ──
                    self.buf.bind_label(hit_label);
                    // Load cached_value (closure) at IC+8
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(0),
                        machine_gp(28),
                        8,
                        MachineWordSize::W64,
                    );
                    // Load cached_func_ptr at IC+16
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(28),
                        machine_gp(28),
                        16,
                        MachineWordSize::W64,
                    );
                    // Set up call args: X0=closure (already), X1=receiver, X2..=args
                    // Move receiver to X1
                    Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        machine_gp(1),
                        preg_to_machine(recv_preg),
                    );
                    // Move user args to X2+
                    for (i, &arg) in args.iter().enumerate() {
                        let arg_preg = self.alloc.get(inst_id, 2 + i).unwrap_or(gp_preg(0));
                        if arg_preg != gp_preg((i + 2) as u8) {
                            Arm64Backend::emit_gp_move(
                                &mut self.buf,
                                machine_gp((i + 2) as u8),
                                preg_to_machine(arg_preg),
                            );
                        }
                    }
                    // Call cached func_ptr
                    Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
                    // Post-BLR safepoint record: cached method body can
                    // allocate freely; GC may fire deep inside.
                    self.record_call_return_safepoint(inst_id);
                    // Result in X0
                    Arm64Backend::emit_branch_to_label(&mut self.buf, done_label);

                    // ── Cache miss: call slow path extern ──
                    self.buf.bind_label(miss_label);
                    // slow_invoke(receiver, symbol_id, num_args, ic_entry_ptr)
                    // X0 = receiver (already preg_to_machine(recv_preg))
                    Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        machine_gp(0),
                        preg_to_machine(recv_preg),
                    );
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(1),
                        symbol.as_u32() as u64,
                    );
                    Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(2), args.len() as u64);
                    // IC entry ptr = ic_base + cache_id * 24
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(3),
                        self.ic_base.wrapping_add(ic_offset as u64),
                    );
                    // Call slow lookup
                    Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(28), self.slow_invoke);
                    Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
                    // Post-BLR safepoint record: slow_invoke may allocate
                    // (e.g. promoting a method to a bound closure when filling
                    // the IC entry).
                    self.record_call_return_safepoint(inst_id);
                    // slow_invoke returned func_ptr in X0, closure is in IC entry (already updated)
                    // Reload closure from IC entry
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(28),
                        self.ic_base.wrapping_add(ic_offset as u64),
                    );
                    let func_ptr_reg = machine_gp(3);
                    Arm64Backend::emit_gp_move(&mut self.buf, func_ptr_reg, machine_gp(0));
                    // Load closure
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        machine_gp(0),
                        machine_gp(28),
                        8,
                        MachineWordSize::W64,
                    );
                    // Set up args: X0=closure, X1=receiver, X2..=user args
                    Arm64Backend::emit_gp_move(
                        &mut self.buf,
                        machine_gp(1),
                        preg_to_machine(recv_preg),
                    );
                    for (i, &arg) in args.iter().enumerate() {
                        let arg_preg = self.alloc.get(inst_id, 2 + i).unwrap_or(gp_preg(0));
                        if arg_preg != gp_preg((i + 2) as u8) {
                            Arm64Backend::emit_gp_move(
                                &mut self.buf,
                                machine_gp((i + 2) as u8),
                                preg_to_machine(arg_preg),
                            );
                        }
                    }
                    // Call the resolved func_ptr
                    Arm64Backend::emit_gp_move(&mut self.buf, machine_gp(28), func_ptr_reg);
                    Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
                    // Post-BLR safepoint record: user closure body can
                    // allocate freely; GC may fire deep inside.
                    self.record_call_return_safepoint(inst_id);

                    // ── Done ──
                    self.buf.bind_label(done_label);
                    // Result is in X0 → move to def register
                    if def_preg != gp_preg(0) {
                        Arm64Backend::emit_gp_move(
                            &mut self.buf,
                            preg_to_machine(def_preg),
                            machine_gp(0),
                        );
                    }
                } // end else (IC enabled)
                // Each inner BLR inside this InvokeDynamic body has its
                // own post-call SafepointRecord above (one per BLR), so
                // the FP-chain walker finds a record at *whichever*
                // saved_lr it encounters. No outer record is needed
                // here — the saved_lr never lands at this point.
            }

            Inst::PushPrompt(_, _)
            | Inst::PopPrompt(_)
            | Inst::PushHandler(_)
            | Inst::PopHandler
            | Inst::CaptureSlice(_, _)
            | Inst::CloneSlice(_) => {
                todo!("Prompt/Slice/Handler operations in batch lowerer");
            }

            Inst::Safepoint(_live) => {
                if let Some(handler) = self.safepoint_handler {
                    // Precise root slots: the regalloc has already spilled
                    // every live vreg to a slot for us (via
                    // `SafepointAction::SpillAndRecord`) and recorded the
                    // mapping in `alloc.stackmaps[inst_id]`. We just emit
                    // the spill-slot frame offsets — no spilling of
                    // callee-saved register banks, no walking of all spill
                    // slots, no walking of all stack slots. Only what's
                    // actually live.
                    let root_slots = self.collect_live_root_slots(inst_id);

                    // The runtime indexes into the *flat* safepoint array
                    // (`JitModule::all_safepoints()`), so emit a global
                    // index, not the per-function one.
                    let safepoint_index = self.safepoint_index_base + self.safepoints.len();
                    let code_offset = self.buf.current_offset();

                    // Call handler: X0 = FP, X1 = safepoint_index
                    Arm64Backend::emit_gp_move(&mut self.buf, machine_gp(0), machine_gp(29));
                    Arm64Backend::emit_mov_imm(
                        &mut self.buf,
                        machine_gp(1),
                        safepoint_index as u64,
                    );
                    Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(28), handler);
                    Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));

                    let return_offset = self.buf.current_offset();
                    self.safepoints.push(crate::SafepointRecord {
                        code_offset,
                        return_offset,
                        root_slots,
                    });
                }
            }
        }
    }

    fn emit_call(
        &mut self,
        _inst_id: InstId,
        fref: FuncRef,
        _args: &[Value],
        _result: Option<Value>,
    ) {
        // Arguments are already in their fixed registers thanks to the allocation.
        // Load function pointer from call table and BLR.
        let func_idx = fref.index();
        // Use X27 for the call table base, X28 for the function pointer.
        // X27 is added to call_clobbers so the allocator won't assign live
        // values to it across calls.
        Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(27), self.call_table_base);
        let offset = (func_idx * 8) as i32;
        Arm64Backend::emit_load_gp(
            &mut self.buf,
            machine_gp(28),
            machine_gp(27),
            offset,
            MachineWordSize::W64,
        );
        Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
        // Result is in X0 (the allocation already assigned it there via FixedReg constraint)
    }

    fn emit_call_indirect(&mut self, inst_id: InstId, args: &[Value], result: Option<Value>) {
        // Callee pointer is in operand 1 (after def). The allocation put it in some register.
        // We need to move it to X28 for BLR.
        let callee_preg = self
            .alloc
            .get(inst_id, 1)
            .expect("missing callee allocation");
        if callee_preg != gp_preg(28) {
            Arm64Backend::emit_gp_move(&mut self.buf, machine_gp(28), preg_to_machine(callee_preg));
        }
        Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
    }

    // ── Terminator emission ───────────────────────────────────

    fn emit_terminator(&mut self, inst_id: InstId, term: &Terminator, block_idx: usize) {
        match term {
            Terminator::Ret(_val) => {
                // Restore callee-saved registers
                self.restore_callee_saved();
                // Return value is in X0 (fixed constraint in operands).
                // Set X1 = 0 (JitOutcomeKind::ReturnValue) for the caller's outcome dispatch.
                Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(1), 0);
                Arm64Backend::emit_epilogue(&mut self.buf, &mut self.epilogue_offsets);
            }
            Terminator::RetVoid => {
                self.restore_callee_saved();
                Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(1), 1); // ReturnVoid
                Arm64Backend::emit_epilogue(&mut self.buf, &mut self.epilogue_offsets);
            }
            Terminator::Jump(target, _args) => {
                // Block args are handled by InsertedMoves (BlockEdge)
                let label = self.block_labels[target.index()];
                Arm64Backend::emit_branch_to_label(&mut self.buf, label);
            }
            Terminator::BrIf {
                cond,
                then_block,
                else_block,
                ..
            } => {
                // Cond is operand 0 of the terminator
                let cond_preg = self.alloc.get(inst_id, 0).expect("missing cond allocation");
                let then_label = self.block_labels[then_block.index()];
                let else_label = self.block_labels[else_block.index()];
                // CBZ = branch if zero → else, fall through to then
                // CBNZ = branch if nonzero → then
                Arm64Backend::emit_cbnz_to_label(
                    &mut self.buf,
                    preg_to_machine(cond_preg),
                    then_label,
                );
                Arm64Backend::emit_branch_to_label(&mut self.buf, else_label);
            }
            _ => {
                todo!(
                    "Terminator {:?} in batch lowerer",
                    std::mem::discriminant(term)
                );
            }
        }
    }

    fn stack_slot_offset(&self, slot: StackSlot) -> i32 {
        // Stack slots come after spill slots (positive offsets from FP)
        let spill_bytes = self.alloc.num_spill_slots * 8;
        let mut offset = self.spill_base_offset + spill_bytes as i32;
        for i in 0..slot.index() {
            offset += self.func.stack_slots[i].size as i32;
        }
        offset
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn type_to_word_size(ty: Type) -> MachineWordSize {
    match ty {
        Type::I8 | Type::I32 => MachineWordSize::W32,
        _ => MachineWordSize::W64,
    }
}

// ── Public entry point ────────────────────────────────────────────

/// Compile a single dynir function using the batch register allocator.
///
/// Compile a single dynir function using the batch register allocator.
///
/// Returns (machine_code, frame_size, safepoint_records).
pub fn compile_function_batch(
    func: &Function,
    externs: &[*const u8],
    call_table_base: u64,
    tags: TagConfig,
    safepoint_handler: Option<u64>,
    safepoint_index_base: usize,
) -> Result<(Vec<u8>, u32, Vec<crate::SafepointRecord>), String> {
    use regalloc::allocator::RegisterAllocator as _;
    use regalloc::linear_scan::LinearScanAllocator;

    // Phase 1: Adapt the IR for the allocator
    let adapted = DynIRFunction::new(func);
    let target = AArch64Target;

    // Phase 2: Run register allocation
    let mut allocator = LinearScanAllocator;
    let mut allocation = allocator
        .allocate(&adapted, &target)
        .map_err(|e| format!("Register allocation failed: {:?}", e))?;

    if std::env::var("BATCH_DEBUG").is_ok() {
        eprintln!("=== Allocation for {} ===", func.name);
        eprintln!(
            "  {} moves, {} spill slots",
            allocation.moves.len(),
            allocation.num_spill_slots
        );
        for (i, m) in allocation.moves.iter().enumerate() {
            eprintln!("  move[{i}]: {:?}", m);
        }
        // Print operands and their allocations for each instruction
        for block_idx in 0..func.blocks.len() {
            let base = adapted.block_inst_base(block_idx);
            let block = &func.blocks[block_idx];
            eprintln!("  block[{block_idx}]:");
            for (li, node) in block.insts.iter().enumerate() {
                let iid = InstId(base + li as u32);
                let ops: Vec<_> = adapted.inst_operands(iid).collect();
                let allocs: Vec<_> = ops
                    .iter()
                    .enumerate()
                    .map(|(oi, op)| {
                        let a = allocation
                            .get(iid, oi)
                            .map(|p| format!("{:?}", p))
                            .unwrap_or("NONE".into());
                        format!("{:?}:{:?}={}", op.kind, op.constraint, a)
                    })
                    .collect();
                let inst_name = match &node.inst {
                    Inst::Iconst(_, v) => format!("Iconst({})", v),
                    Inst::Call(fref, args) => format!(
                        "Call(f{}, args={:?})",
                        fref.index(),
                        args.iter().map(|v| v.index()).collect::<Vec<_>>()
                    ),
                    Inst::CallIndirect(c, args, _) => format!(
                        "CallIndirect(c={}, args={:?})",
                        c.index(),
                        args.iter().map(|v| v.index()).collect::<Vec<_>>()
                    ),
                    other => format!("{:?}", std::mem::discriminant(other)),
                };
                eprintln!(
                    "    inst({}) {} val={:?} → [{}]",
                    iid.0,
                    inst_name,
                    node.value.map(|v| v.index()),
                    allocs.join(", ")
                );
            }
            let tid = InstId(base + block.insts.len() as u32);
            let tops: Vec<_> = adapted.inst_operands(tid).collect();
            let tallocs: Vec<_> = tops
                .iter()
                .enumerate()
                .map(|(oi, op)| {
                    let a = allocation
                        .get(tid, oi)
                        .map(|p| format!("{:?}", p))
                        .unwrap_or("NONE".into());
                    format!("{:?}:{:?}={}", op.kind, op.constraint, a)
                })
                .collect();
            eprintln!(
                "    term({}) {:?} → [{}]",
                tid.0,
                std::mem::discriminant(&block.terminator),
                tallocs.join(", ")
            );
        }
    }

    // Phase 2.5: Inject moves for entry block params (function arguments).
    // The caller puts args in X0, X1, ... but the allocator may have assigned
    // the param vregs to different home registers. Insert Before(first_inst) moves.
    {
        let first_inst = InstId(0);
        let num_args = func.sig.params.len();
        for (i, (val, _ty)) in func.blocks[0].params.iter().take(num_args).enumerate() {
            let vreg = VReg(val.index() as u32);
            let arg_preg = gp_preg(i as u8);

            if let Some(home) = allocation.vreg_homes.get(&vreg) {
                let from = MoveOperand::Reg(arg_preg);
                if from != *home {
                    if std::env::var("BATCH_DEBUG").is_ok() {
                        eprintln!(
                            "  entry_param[{i}]: val={} {:?} → {:?}",
                            val.index(),
                            from,
                            home
                        );
                    }
                    allocation.moves.push(InsertedMove {
                        at: MovePosition::Before(first_inst),
                        from,
                        to: *home,
                        class: GP,
                    });
                }
            }
        }
    }

    // Phase 3: Emit machine code
    let emitter = BatchEmitter::new(
        func,
        &adapted,
        &allocation,
        externs,
        call_table_base,
        tags,
        safepoint_handler,
        0,
        0,
        0, // IC not yet wired — no InvokeDynamic in IR yet
        safepoint_index_base,
    );
    let (mut buf, frame_size, safepoints) = emitter.emit();

    // Finalize (resolve branch labels)
    buf.finalize();
    let code = buf.code().to_vec();

    Ok((code, frame_size, safepoints))
}
