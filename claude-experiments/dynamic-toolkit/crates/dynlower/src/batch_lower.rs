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
use crate::regalloc_bridge::{AArch64Target, DynIRFunction, GP, FP, gp_preg, fp_preg};

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
    /// Epilogue patch offsets (for frame size patching)
    epilogue_offsets: Vec<usize>,
}

impl<'a> BatchEmitter<'a> {
    pub fn new(
        func: &'a Function,
        adapted: &'a DynIRFunction<'a>,
        alloc: &'a Allocation,
        externs: &'a [*const u8],
        call_table_base: u64,
        tags: TagConfig,
    ) -> Self {
        let n_blocks = func.blocks.len();
        let mut buf = CodeBuffer::new();
        let block_labels: Vec<Label> = (0..n_blocks).map(|_| buf.create_label()).collect();

        // Spill slots start below the frame pointer.
        // Layout: [FP, LR] at [FP+0], then spill slots at [FP-8], [FP-16], ...
        // Plus space for callee-saved registers we need to preserve.
        let spill_base_offset = -16; // first spill at FP-16 (FP-8 reserved)

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
            epilogue_offsets: Vec::new(),
        }
    }

    /// Determine which callee-saved GP registers are used by the allocation.
    fn used_callee_saved(&self) -> Vec<PReg> {
        let callee_saved: &[PReg] = &[
            PReg(19), PReg(20), PReg(21), PReg(22), PReg(23),
            PReg(24), PReg(25), PReg(26), PReg(27),
        ];
        let mut used = Vec::new();
        for &preg in callee_saved {
            // Check if any instruction operand or move uses this register
            let is_used = self.alloc.inst_allocs.values().any(|&p| p == preg)
                || self.alloc.moves.iter().any(|m| {
                    m.from == MoveOperand::Reg(preg) || m.to == MoveOperand::Reg(preg)
                });
            if is_used {
                used.push(preg);
            }
        }
        used
    }

    /// Emit the full function. Returns the CodeBuffer ready for finalization.
    pub fn emit(mut self) -> (CodeBuffer<Arm64>, u32) {
        // Determine callee-saved registers to preserve
        let callee_saved = self.used_callee_saved();
        self.callee_saved_used = callee_saved.clone();

        // Prologue (patch frame size later)
        let prologue_offset = Arm64Backend::emit_prologue(&mut self.buf);

        // Save callee-saved registers to frame (at FP - 16, FP - 24, ...)
        let callee_save_base = -16i32; // first callee-save slot at FP-16
        for (i, &preg) in callee_saved.iter().enumerate() {
            let offset = callee_save_base - (i as i32) * 8;
            Arm64Backend::emit_store_gp(
                &mut self.buf, preg_to_machine(preg), machine_gp(29), offset,
                MachineWordSize::W64,
            );
        }

        // Adjust spill base to account for callee-saved saves
        let callee_save_bytes = callee_saved.len() as u32 * 8;
        self.spill_base_offset = -16 - callee_save_bytes as i32;

        // Compute frame size: FP+LR + callee-saved saves + spill slots + stack slots
        let num_spills = self.alloc.num_spill_slots;
        let stack_slot_bytes: u32 = self.func.stack_slots.iter().map(|s| s.size).sum();
        let raw_frame = 16 /* FP+LR */ + callee_save_bytes + num_spills * 8 + stack_slot_bytes;
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

        (self.buf, self.frame_size)
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
        let moves: Vec<_> = self.alloc.moves.iter()
            .filter(|m| m.at == MovePosition::Before(inst))
            .cloned()
            .collect();
        self.emit_parallel_moves(&moves);
    }

    fn emit_after_moves(&mut self, inst: InstId) {
        let moves: Vec<_> = self.alloc.moves.iter()
            .filter(|m| m.at == MovePosition::After(inst))
            .cloned()
            .collect();
        self.emit_parallel_moves(&moves);
    }

    fn emit_edge_moves_from(&mut self, from_block: usize) {
        let from = BlockId(from_block as u32);
        let moves: Vec<_> = self.alloc.moves.iter()
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
            for m in moves { self.emit_move(m); }
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
                if emitted[i] { continue; }
                let dst_is_source = sources.iter().enumerate().any(|(j, src)| {
                    !emitted[j] && i != j && *src == moves[i].to
                });
                if !dst_is_source {
                    self.emit_move(&moves[i]);
                    emitted[i] = true;
                    progress = true;
                }
            }
        }

        // Remaining moves form cycles — break with scratch register X28
        for i in 0..moves.len() {
            if emitted[i] { continue; }
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
                if emitted[j] { continue; }
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
            (MoveOperand::SpillSlot(_), MoveOperand::SpillSlot(_)) => {
                panic!("stack-to-stack moves should not be generated");
            }
        }
    }

    fn restore_callee_saved(&mut self) {
        let callee_save_base = -16i32;
        for (i, &preg) in self.callee_saved_used.iter().enumerate() {
            let offset = callee_save_base - (i as i32) * 8;
            Arm64Backend::emit_load_gp(
                &mut self.buf, preg_to_machine(preg), machine_gp(29), offset,
                MachineWordSize::W64,
            );
        }
    }

    fn spill_offset(&self, slot: SpillSlot) -> i32 {
        // Spill slots are at FP - 16 - slot_index * 8
        self.spill_base_offset - (slot.0 as i32) * 8
    }

    // ── Instruction emission ──────────────────────────────────

    fn emit_inst(&mut self, inst_id: InstId, node: &InstNode, _block_idx: usize) {
        // Operand 0 is always the def (result) if present.
        // Operands 1+ are uses, in the order defined by compute_inst_operands.
        let get_def = |alloc: &Allocation| -> PReg {
            alloc.get(inst_id, 0).expect("missing def allocation")
        };
        let get_use = |alloc: &Allocation, idx: usize| -> PReg {
            alloc.get(inst_id, idx).expect("missing use allocation")
        };

        match &node.inst {
            Inst::Iconst(ty, imm) => {
                let dst = get_def(&self.alloc);
                Arm64Backend::emit_mov_imm(&mut self.buf, preg_to_machine(dst), *imm as u64);
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

            Inst::Add(_, _) | Inst::Sub(_, _) | Inst::Mul(_, _) |
            Inst::SDiv(_, _) | Inst::UDiv(_, _) |
            Inst::And(_, _) | Inst::Or(_, _) | Inst::Xor(_, _) |
            Inst::Shl(_, _) | Inst::LShr(_, _) | Inst::AShr(_, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                let ty = node.value.map(|v| self.func.value_type(v)).unwrap_or(Type::I64);
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
                    &mut self.buf, op,
                    preg_to_machine(dst), preg_to_machine(lhs), preg_to_machine(rhs),
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
                    &mut self.buf, op,
                    preg_to_machine(dst), preg_to_machine(lhs), preg_to_machine(rhs),
                );
            }

            Inst::Neg(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let ty = node.value.map(|v| self.func.value_type(v)).unwrap_or(Type::I64);
                Arm64Backend::emit_gp_neg(&mut self.buf, preg_to_machine(dst), preg_to_machine(src), type_to_word_size(ty));
            }
            Inst::Not(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let ty = node.value.map(|v| self.func.value_type(v)).unwrap_or(Type::I64);
                Arm64Backend::emit_gp_not(&mut self.buf, preg_to_machine(dst), preg_to_machine(src), type_to_word_size(ty));
            }
            Inst::FNeg(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                Arm64Backend::emit_fp_neg(&mut self.buf, preg_to_machine(dst), preg_to_machine(src));
            }

            Inst::Icmp(op, _, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                Arm64Backend::emit_icmp_set(
                    &mut self.buf, *op,
                    preg_to_machine(dst), preg_to_machine(lhs), preg_to_machine(rhs),
                    MachineWordSize::W64,
                );
            }

            Inst::Fcmp(op, _, _) => {
                let dst = get_def(&self.alloc);
                let lhs = get_use(&self.alloc, 1);
                let rhs = get_use(&self.alloc, 2);
                Arm64Backend::emit_fcmp_set(
                    &mut self.buf, *op,
                    preg_to_machine(dst), preg_to_machine(lhs), preg_to_machine(rhs),
                );
            }

            Inst::Load(ty, _, offset) => {
                let dst = get_def(&self.alloc);
                let base = get_use(&self.alloc, 1);
                if *ty == Type::F64 {
                    Arm64Backend::emit_load_fp(
                        &mut self.buf,
                        preg_to_machine(dst), preg_to_machine(base), *offset,
                    );
                } else {
                    Arm64Backend::emit_load_gp(
                        &mut self.buf,
                        preg_to_machine(dst), preg_to_machine(base), *offset,
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
                    preg_to_machine(val), preg_to_machine(addr), *offset,
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
                        &mut self.buf, MachineGpBinOp::Add,
                        preg_to_machine(dst), machine_gp(29),
                        machine_gp(29), // will be overridden by immediate
                        MachineWordSize::W64,
                    );
                    // Actually, we need add-immediate. Use mov_imm + add for now.
                    Arm64Backend::emit_mov_imm(&mut self.buf, preg_to_machine(dst), slot_offset as u64);
                    Arm64Backend::emit_gp_binop(
                        &mut self.buf, MachineGpBinOp::Add,
                        preg_to_machine(dst), machine_gp(29), preg_to_machine(dst),
                        MachineWordSize::W64,
                    );
                } else {
                    Arm64Backend::emit_mov_imm(&mut self.buf, preg_to_machine(dst), (-slot_offset) as u64);
                    Arm64Backend::emit_gp_binop(
                        &mut self.buf, MachineGpBinOp::Sub,
                        preg_to_machine(dst), machine_gp(29), preg_to_machine(dst),
                        MachineWordSize::W64,
                    );
                }
            }

            Inst::Bitcast(_, _) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                let dst_ty = node.value.map(|v| self.func.value_type(v)).unwrap_or(Type::I64);
                if dst_ty == Type::F64 {
                    // I64 → F64
                    Arm64Backend::emit_gp_to_fp_move(&mut self.buf, preg_to_machine(dst), preg_to_machine(src));
                } else {
                    // F64 → I64
                    Arm64Backend::emit_fp_to_gp_move(&mut self.buf, preg_to_machine(dst), preg_to_machine(src));
                }
            }

            Inst::Sext(_, _) | Inst::Zext(_, _) | Inst::Trunc(_, _) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                // For now, just mov (most conversions between i8/i32/i64 are trivial on arm64)
                Arm64Backend::emit_gp_move(&mut self.buf, preg_to_machine(dst), preg_to_machine(src));
            }

            Inst::Select(_, _, _) => {
                let dst = get_def(&self.alloc);
                let cond = get_use(&self.alloc, 1);
                let if_true = get_use(&self.alloc, 2);
                let if_false = get_use(&self.alloc, 3);
                Arm64Backend::emit_gp_select(
                    &mut self.buf,
                    preg_to_machine(dst), preg_to_machine(cond),
                    preg_to_machine(if_true), preg_to_machine(if_false),
                    MachineWordSize::W64,
                );
            }

            Inst::Payload(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                <Arm64Backend as LoweringBackend>::emit_extract_payload(
                    &mut self.buf,
                    preg_to_machine(dst), preg_to_machine(src),
                    self.tags.has_unboxed_float, self.tags.payload_bits,
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
                    preg_to_machine(dst), preg_to_machine(src),
                    self.tags.has_unboxed_float, self.tags.payload_bits,
                    self.tags.tag_count as u64 - 1, expected,
                );
            }

            Inst::MakeTagged(tag, _) => {
                let dst = get_def(&self.alloc);
                let payload = get_use(&self.alloc, 1);
                <Arm64Backend as LoweringBackend>::emit_make_tagged(
                    &mut self.buf,
                    preg_to_machine(dst), preg_to_machine(payload),
                    self.tags.has_unboxed_float, self.tags.payload_bits,
                    (self.tags.encode_tagged)(*tag, 0), *tag as u64,
                );
            }

            Inst::TagOf(_) => {
                let dst = get_def(&self.alloc);
                let src = get_use(&self.alloc, 1);
                <Arm64Backend as LoweringBackend>::emit_tag_of(
                    &mut self.buf,
                    preg_to_machine(dst), preg_to_machine(src),
                    self.tags.has_unboxed_float, self.tags.payload_bits,
                    self.tags.tag_count as u64 - 1,
                );
            }

            Inst::Call(fref, args) => {
                self.emit_call(inst_id, *fref, args, node.value);
            }

            Inst::CallIndirect(callee_val, args, _ret_ty) => {
                self.emit_call_indirect(inst_id, args, node.value);
            }

            Inst::IntToFloat(_) | Inst::FloatToInt(_) => {
                todo!("Int/Float conversions in batch lowerer");
            }

            Inst::OverflowCheck(_, _, _) | Inst::Guard(_, _, _) => {
                todo!("Overflow/Guard in batch lowerer");
            }

            Inst::PushPrompt(_, _) | Inst::PopPrompt(_) |
            Inst::CaptureSlice(_, _) | Inst::CloneSlice(_) => {
                todo!("Prompt/Slice operations in batch lowerer");
            }

            Inst::Safepoint(_) => {
                // TODO: emit safepoint
            }
        }
    }

    fn emit_call(&mut self, _inst_id: InstId, fref: FuncRef, _args: &[Value], _result: Option<Value>) {
        // Arguments are already in their fixed registers thanks to the allocation.
        // Load function pointer from call table and BLR.
        let func_idx = fref.index();
        Arm64Backend::emit_mov_imm(&mut self.buf, machine_gp(27), self.call_table_base);
        let offset = (func_idx * 8) as i32;
        Arm64Backend::emit_load_gp(
            &mut self.buf, machine_gp(28), machine_gp(27), offset, MachineWordSize::W64,
        );
        Arm64Backend::emit_call_reg(&mut self.buf, machine_gp(28));
        // Result is in X0 (the allocation already assigned it there via FixedReg constraint)
    }

    fn emit_call_indirect(&mut self, inst_id: InstId, args: &[Value], result: Option<Value>) {
        // Callee pointer is in operand 1 (after def). The allocation put it in some register.
        // We need to move it to X28 for BLR.
        let callee_preg = self.alloc.get(inst_id, 1).expect("missing callee allocation");
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
            Terminator::BrIf { cond, then_block, else_block, .. } => {
                // Cond is operand 0 of the terminator
                let cond_preg = self.alloc.get(inst_id, 0).expect("missing cond allocation");
                let then_label = self.block_labels[then_block.index()];
                let else_label = self.block_labels[else_block.index()];
                // CBZ = branch if zero → else, fall through to then
                // CBNZ = branch if nonzero → then
                Arm64Backend::emit_cbnz_to_label(&mut self.buf, preg_to_machine(cond_preg), then_label);
                Arm64Backend::emit_branch_to_label(&mut self.buf, else_label);
            }
            _ => {
                todo!("Terminator {:?} in batch lowerer", std::mem::discriminant(term));
            }
        }
    }

    fn stack_slot_offset(&self, slot: StackSlot) -> i32 {
        // Stack slots come after spill slots
        let spill_bytes = self.alloc.num_spill_slots * 8;
        let mut offset = self.spill_base_offset - spill_bytes as i32;
        for i in 0..slot.index() {
            offset -= self.func.stack_slots[i].size as i32;
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
/// Returns the emitted machine code bytes and frame size.
/// Compile a single dynir function using the batch register allocator.
///
/// Returns the emitted machine code bytes and frame size.
pub fn compile_function_batch(
    func: &Function,
    externs: &[*const u8],
    call_table_base: u64,
    tags: TagConfig,
) -> Result<(Vec<u8>, u32), String> {
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
        eprintln!("  {} moves, {} spill slots", allocation.moves.len(), allocation.num_spill_slots);
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
                let allocs: Vec<_> = ops.iter().enumerate()
                    .map(|(oi, op)| {
                        let a = allocation.get(iid, oi).map(|p| format!("{:?}", p)).unwrap_or("NONE".into());
                        format!("{:?}:{:?}={}", op.kind, op.constraint, a)
                    }).collect();
                let inst_name = match &node.inst {
                    Inst::Iconst(_, v) => format!("Iconst({})", v),
                    Inst::Call(fref, args) => format!("Call(f{}, args={:?})", fref.index(), args.iter().map(|v| v.index()).collect::<Vec<_>>()),
                    Inst::CallIndirect(c, args, _) => format!("CallIndirect(c={}, args={:?})", c.index(), args.iter().map(|v| v.index()).collect::<Vec<_>>()),
                    other => format!("{:?}", std::mem::discriminant(other)),
                };
                eprintln!("    inst({}) {} val={:?} → [{}]", iid.0, inst_name, node.value.map(|v| v.index()), allocs.join(", "));
            }
            let tid = InstId(base + block.insts.len() as u32);
            let tops: Vec<_> = adapted.inst_operands(tid).collect();
            let tallocs: Vec<_> = tops.iter().enumerate()
                .map(|(oi, op)| {
                    let a = allocation.get(tid, oi).map(|p| format!("{:?}", p)).unwrap_or("NONE".into());
                    format!("{:?}:{:?}={}", op.kind, op.constraint, a)
                }).collect();
            eprintln!("    term({}) {:?} → [{}]", tid.0, std::mem::discriminant(&block.terminator), tallocs.join(", "));
        }
    }

    // Phase 2.5: Inject moves for entry block params (function arguments).
    // The caller puts args in X0, X1, ... but the allocator may have assigned
    // the param vregs to different registers. Insert Before(first_inst) moves.
    {
        let first_inst = InstId(0);
        for (i, (val, _ty)) in func.blocks[0].params.iter().enumerate() {
            let vreg = VReg(val.index() as u32);
            let arg_preg = gp_preg(i as u8);

            if let Some(&spill_slot) = allocation.spill_slots.get(&vreg) {
                // Param was spilled
                allocation.moves.push(InsertedMove {
                    at: MovePosition::Before(first_inst),
                    from: MoveOperand::Reg(arg_preg),
                    to: MoveOperand::SpillSlot(spill_slot),
                    class: GP,
                });
            } else {
                // Find the home register for this vreg — look at where it's used
                // as a Use operand. The allocated preg there is its home.
                let mut home = None;
                'outer: for block in adapted.blocks() {
                    for inst in adapted.block_insts(block) {
                        for (op_idx, op) in adapted.inst_operands(inst).enumerate() {
                            if let Reg::Virtual(v) = op.reg {
                                if v == vreg {
                                    if let Some(preg) = allocation.get(inst, op_idx) {
                                        home = Some(preg);
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(home_preg) = home {
                    if home_preg != arg_preg {
                        allocation.moves.push(InsertedMove {
                            at: MovePosition::Before(first_inst),
                            from: MoveOperand::Reg(arg_preg),
                            to: MoveOperand::Reg(home_preg),
                            class: GP,
                        });
                    }
                }
            }
        }
    }

    // Phase 3: Emit machine code
    let emitter = BatchEmitter::new(func, &adapted, &allocation, externs, call_table_base, tags);
    let (mut buf, frame_size) = emitter.emit();

    // Finalize (resolve branch labels)
    buf.finalize();
    let code = buf.code().to_vec();

    Ok((code, frame_size))
}
