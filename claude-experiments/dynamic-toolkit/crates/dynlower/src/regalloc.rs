use crate::backend::{LoweringBackend, MachineReg, MachineRegClass};
use dynasm::buffer::CodeBuffer;
use dynexec::FrameLayout;
use dynir::ir::{Function, Inst, Value};
use dynir::opt;
use dynir::types::Type;

// ─── Value Location ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueLoc {
    GpReg(u8),
    FpReg(u8),
    Spill(i32), // offset from FP (negative)
    Unassigned,
}

#[derive(Debug, Clone, Copy)]
pub struct ValueInfo {
    pub loc: ValueLoc,
    pub spill_slot: Option<i32>, // if spilled, the offset from FP
    pub remaining_uses: u32,
    pub ty: Type,
}

// ─── Constants ─────────────────────────────────────────────────────

pub const NUM_GP: usize = 28; // X0-X27
pub const NUM_FP: usize = 32; // D0-D31

// Default AArch64 register sets. Backends can override these through
// LoweringBackend; keep the constants for tests and non-backend helpers.
pub const ALLOCATABLE_GP: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// Allocatable FP regs: caller-saved only.
pub const ALLOCATABLE_FP: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

// Caller-saved GP: X0-X15 (X16-X18 are special)
pub const CALLER_SAVED_GP: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// Caller-saved FP: D0-D7, D16-D31
pub const CALLER_SAVED_FP: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

// ─── Helpers ───────────────────────────────────────────────────────

pub fn machine_gp(index: u8) -> MachineReg {
    MachineReg {
        class: MachineRegClass::Gp,
        index,
    }
}

pub fn machine_fp(index: u8) -> MachineReg {
    MachineReg {
        class: MachineRegClass::Fp,
        index,
    }
}

pub fn is_float_type(ty: Type) -> bool {
    ty == Type::F64
}

// ─── Register Allocator Trait ──────────────────────────────────────

pub trait RegisterAllocator: Sized {
    /// Create a new allocator for a function with `num_values` SSA values.
    fn new(num_values: usize) -> Self;

    /// Pre-lowering analysis. Called after use counts and types are set.
    /// Greedy: no-op. Linear scan: compute live intervals and assign registers.
    fn prepare<B: LoweringBackend>(&mut self, _func: &Function) {}

    // Value queries
    fn value_type(&self, val: Value) -> Type;
    fn value_loc(&self, val: Value) -> ValueLoc;
    fn value_spill_slot(&self, val: Value) -> Option<i32>;
    fn remaining_uses(&self, val: Value) -> u32;

    // Register allocation
    fn alloc_gp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8;
    fn alloc_fp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8;
    fn ensure_in_gp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8;
    fn ensure_in_fp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8;

    // Tracking
    fn assign_gp(&mut self, val: Value, r: u8);
    fn assign_fp(&mut self, val: Value, r: u8);
    fn set_spill_slot(&mut self, val: Value, offset: i32);
    fn set_value_loc(&mut self, val: Value, loc: ValueLoc);
    fn set_type(&mut self, val: Value, ty: Type);
    fn dec_use(&mut self, val: Value);
    fn inc_use(&mut self, val: Value);

    /// Mark a GP register as occupied by a value without updating the value's
    /// canonical location.  Used during call‑arg setup so that subsequent
    /// allocations don't clobber destination registers that already hold
    /// the correct value.
    fn mark_gp_occupied(&mut self, r: u8, val: Value);
    fn clear_gp_occupied(&mut self, r: u8, val: Value);

    // Spilling
    fn spill_caller_saved<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    );
    fn spill_all_live<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    );

    // Block transitions
    fn clear_regs(&mut self);

    /// Set up register state at block entry.
    /// Greedy: clear regs, load block params from canonical slots.
    /// Linear scan: set register state from pre-computed assignments.
    fn enter_block<B: LoweringBackend>(
        &mut self,
        block_idx: usize,
        func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    );

    /// Emit moves for block arguments at a branch to `target`.
    /// Greedy: store to canonical frame slots.
    /// Linear scan: emit parallel register-to-register moves.
    fn emit_block_args<B: LoweringBackend>(
        &mut self,
        target_idx: usize,
        args: &[Value],
        func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    );

    /// After a call returns, caller-saved registers are clobbered.
    /// Clear their occupants and revert values to spill slots.
    /// No code is emitted — values were already spilled before the call.
    fn clobber_caller_saved<B: LoweringBackend>(&mut self);

    /// Place the value currently in the C-ABI return register
    /// (machine GP register 0, or FP register 0 for floats) into
    /// the location this regalloc has assigned to param `param_idx`
    /// of `target_block`.
    ///
    /// Used by Invoke's normal/exception continuation paths: the
    /// JIT calling convention always delivers the call's return
    /// value (or — for the exception path — the thrown value
    /// preserved in x0 across pop_suspended_frame) in machine
    /// register 0. But the SSA block param may have been assigned
    /// a different register (LinearScan), an explicit spill slot
    /// (LinearScan), or only the canonical frame slot (Greedy).
    /// Each regalloc knows its own conventions; this method
    /// emits the appropriate store/move.
    ///
    /// Default impl: store to canonical block-param slot. Works for
    /// Greedy (which loads block params from canonical slots on
    /// block entry). LinearScan overrides to honor register/spill
    /// assignments — without that override, an Invoke's return
    /// value silently fails to reach its consumer.
    fn place_call_return_in_block_param<B: LoweringBackend>(
        &mut self,
        target_idx: usize,
        param_idx: usize,
        ret_ty: Type,
        _func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        let slot = frame.slot_access(frame.block_param_slots(target_idx)[param_idx]);
        if is_float_type(ret_ty) {
            B::emit_store_fp_to_frame(buf, machine_fp(0), slot);
        } else {
            B::emit_store_gp_to_frame(buf, machine_gp(0), slot);
        }
    }

    // State save/restore (for BrIf)
    type SavedState: Clone;
    fn save_state(&self) -> Self::SavedState;
    fn restore_state(&mut self, state: Self::SavedState);

    // Register occupant inspection
    fn gp_occupant(&self, r: usize) -> Option<Value>;
    fn fp_occupant(&self, r: usize) -> Option<Value>;

    /// Number of tracked values.
    fn num_values(&self) -> usize;

    /// Get a snapshot of (spill_slot, type) for each value, for frame reification.
    fn value_info_snapshot(&self) -> Vec<(Option<i32>, Type)>;
}

// ─── Greedy Register State ─────────────────────────────────────────
//
// NOT a publicly-exposed allocator. `LinearScanAllocator` is the only
// allocator frontends should use. `GreedyRegState`'s methods are
// reused internally by `LinearScanAllocator` for the dynamic
// (non-prepared) lowering paths — register-allocation fallbacks
// during the lower pass when no pre-assigned home exists.

pub(crate) struct GreedyRegState {
    pub gp_occupant: [Option<Value>; NUM_GP],
    pub fp_occupant: [Option<Value>; NUM_FP],
    pub values: Vec<ValueInfo>,
    gp_evict_cursor: usize,
    fp_evict_cursor: usize,
}

#[derive(Clone)]
pub struct GreedySavedState {
    pub values: Vec<(ValueLoc, Option<i32>)>,
    pub gp_occupant: [Option<Value>; NUM_GP],
    pub fp_occupant: [Option<Value>; NUM_FP],
}

impl GreedyRegState {
    pub fn new(num_values: usize) -> Self {
        GreedyRegState {
            gp_occupant: [None; NUM_GP],
            fp_occupant: [None; NUM_FP],
            values: (0..num_values)
                .map(|_| ValueInfo {
                    loc: ValueLoc::Unassigned,
                    spill_slot: None,
                    remaining_uses: 0,
                    ty: Type::I64,
                })
                .collect(),
            gp_evict_cursor: 0,
            fp_evict_cursor: 0,
        }
    }

    pub fn alloc_gp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8 {
        // Try to find a free allocatable GP reg
        for &r in B::allocatable_gp() {
            if self.gp_occupant[r as usize].is_none() {
                return r;
            }
        }
        // Evict: round-robin through allocatable regs
        let start = self.gp_evict_cursor;
        let regs = B::allocatable_gp();
        for i in 0..regs.len() {
            let idx = (start + i) % regs.len();
            let r = regs[idx];
            if let Some(val) = self.gp_occupant[r as usize] {
                self.spill_gp_reg::<B>(buf, frame, r, val);
                self.gp_evict_cursor = (idx + 1) % regs.len();
                return r;
            }
        }
        panic!("no GP register available");
    }

    pub fn alloc_fp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8 {
        for &r in B::allocatable_fp() {
            if self.fp_occupant[r as usize].is_none() {
                return r;
            }
        }
        let start = self.fp_evict_cursor;
        let regs = B::allocatable_fp();
        for i in 0..regs.len() {
            let idx = (start + i) % regs.len();
            let r = regs[idx];
            if let Some(val) = self.fp_occupant[r as usize] {
                self.spill_fp_reg::<B>(buf, frame, r, val);
                self.fp_evict_cursor = (idx + 1) % regs.len();
                return r;
            }
        }
        panic!("no FP register available");
    }

    pub fn spill_gp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        r: u8,
        val: Value,
    ) {
        if self.values[val.index()].remaining_uses == 0 {
            self.gp_occupant[r as usize] = None;
            self.values[val.index()].loc = ValueLoc::Unassigned;
            return;
        }
        let offset = match self.values[val.index()].spill_slot {
            Some(off) => off,
            None => {
                let off = frame.alloc_root_slot();
                self.values[val.index()].spill_slot = Some(off);
                off
            }
        };
        B::emit_store_gp_to_frame(buf, machine_gp(r), frame.slot_access(offset));
        self.values[val.index()].loc = ValueLoc::Spill(offset);
        self.gp_occupant[r as usize] = None;
    }

    pub fn spill_fp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        r: u8,
        val: Value,
    ) {
        if self.values[val.index()].remaining_uses == 0 {
            self.fp_occupant[r as usize] = None;
            self.values[val.index()].loc = ValueLoc::Unassigned;
            return;
        }
        let offset = match self.values[val.index()].spill_slot {
            Some(off) => off,
            None => {
                let off = frame.alloc_root_slot();
                self.values[val.index()].spill_slot = Some(off);
                off
            }
        };
        B::emit_store_fp_to_frame(buf, machine_fp(r), frame.slot_access(offset));
        self.values[val.index()].loc = ValueLoc::Spill(offset);
        self.fp_occupant[r as usize] = None;
    }

    pub fn assign_gp(&mut self, val: Value, r: u8) {
        self.gp_occupant[r as usize] = Some(val);
        self.values[val.index()].loc = ValueLoc::GpReg(r);
    }

    pub fn assign_fp(&mut self, val: Value, r: u8) {
        self.fp_occupant[r as usize] = Some(val);
        self.values[val.index()].loc = ValueLoc::FpReg(r);
    }

    /// Ensure a value is in a GP register, returning the reg index.
    pub fn ensure_in_gp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8 {
        match self.values[val.index()].loc {
            ValueLoc::GpReg(r) => r,
            ValueLoc::Spill(offset) => {
                let r = self.alloc_gp::<B>(buf, frame);
                B::emit_load_gp_from_frame(buf, machine_gp(r), frame.slot_access(offset));
                // Record the spill slot so the value can be restored after calls
                if self.values[val.index()].spill_slot.is_none() {
                    self.values[val.index()].spill_slot = Some(offset);
                }
                self.assign_gp(val, r);
                r
            }
            ValueLoc::FpReg(_) => panic!("expected GP value, got FP"),
            ValueLoc::Unassigned => panic!("value v{} is unassigned", val.index()),
        }
    }

    /// Ensure a value is in an FP register, returning the reg index.
    pub fn ensure_in_fp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8 {
        match self.values[val.index()].loc {
            ValueLoc::FpReg(r) => r,
            ValueLoc::Spill(offset) => {
                let r = self.alloc_fp::<B>(buf, frame);
                B::emit_load_fp_from_frame(buf, machine_fp(r), frame.slot_access(offset));
                self.assign_fp(val, r);
                r
            }
            ValueLoc::GpReg(_) => panic!("expected FP value, got GP"),
            ValueLoc::Unassigned => panic!("value v{} is unassigned", val.index()),
        }
    }

    /// Decrement use count and free register if dead.
    pub fn dec_use(&mut self, val: Value) {
        let info = &mut self.values[val.index()];
        info.remaining_uses = info.remaining_uses.saturating_sub(1);
        if info.remaining_uses == 0 {
            match info.loc {
                ValueLoc::GpReg(r) => {
                    self.gp_occupant[r as usize] = None;
                }
                ValueLoc::FpReg(r) => {
                    self.fp_occupant[r as usize] = None;
                }
                _ => {}
            }
        }
    }

    /// Spill all caller-saved registers that have live values.
    pub fn spill_caller_saved<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        for &r in B::caller_saved_gp() {
            if let Some(val) = self.gp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_gp_reg::<B>(buf, frame, r, val);
                    // Mark the value as in its spill slot (register will be clobbered)
                    if let Some(slot) = self.values[val.index()].spill_slot {
                        self.values[val.index()].loc = ValueLoc::Spill(slot);
                    }
                    self.gp_occupant[r as usize] = None;
                } else {
                    self.gp_occupant[r as usize] = None;
                    self.values[val.index()].loc = ValueLoc::Unassigned;
                }
            }
        }
        for &r in B::caller_saved_fp() {
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_fp_reg::<B>(buf, frame, r, val);
                    if let Some(slot) = self.values[val.index()].spill_slot {
                        self.values[val.index()].loc = ValueLoc::Spill(slot);
                    }
                    self.fp_occupant[r as usize] = None;
                } else {
                    self.fp_occupant[r as usize] = None;
                    self.values[val.index()].loc = ValueLoc::Unassigned;
                }
            }
        }
    }

    /// Spill ALL live register-resident values (used before multi-pred block transitions).
    pub fn spill_all_live<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        for r in 0..NUM_GP as u8 {
            if let Some(val) = self.gp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_gp_reg::<B>(buf, frame, r, val);
                } else {
                    self.gp_occupant[r as usize] = None;
                }
            }
        }
        for r in 0..NUM_FP as u8 {
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_fp_reg::<B>(buf, frame, r, val);
                } else {
                    self.fp_occupant[r as usize] = None;
                }
            }
        }
    }

    /// Free all register mappings. Values in registers revert to their spill
    /// slot if one exists, otherwise become Unassigned.
    pub fn clear_regs(&mut self) {
        for info in &mut self.values {
            match info.loc {
                ValueLoc::GpReg(_) | ValueLoc::FpReg(_) => {
                    if let Some(off) = info.spill_slot {
                        info.loc = ValueLoc::Spill(off);
                    } else {
                        info.loc = ValueLoc::Unassigned;
                    }
                }
                _ => {}
            }
        }
        self.gp_occupant = [None; NUM_GP];
        self.fp_occupant = [None; NUM_FP];
    }
}

// ─── RegisterAllocator impl for GreedyRegState ────────────────────

impl RegisterAllocator for GreedyRegState {
    fn new(num_values: usize) -> Self {
        GreedyRegState::new(num_values)
    }

    fn value_type(&self, val: Value) -> Type {
        self.values[val.index()].ty
    }

    fn value_loc(&self, val: Value) -> ValueLoc {
        self.values[val.index()].loc
    }

    fn value_spill_slot(&self, val: Value) -> Option<i32> {
        self.values[val.index()].spill_slot
    }

    fn remaining_uses(&self, val: Value) -> u32 {
        self.values[val.index()].remaining_uses
    }

    fn alloc_gp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8 {
        GreedyRegState::alloc_gp::<B>(self, buf, frame)
    }

    fn alloc_fp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8 {
        GreedyRegState::alloc_fp::<B>(self, buf, frame)
    }

    fn ensure_in_gp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8 {
        GreedyRegState::ensure_in_gp_reg::<B>(self, buf, frame, val)
    }

    fn ensure_in_fp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8 {
        GreedyRegState::ensure_in_fp_reg::<B>(self, buf, frame, val)
    }

    fn assign_gp(&mut self, val: Value, r: u8) {
        GreedyRegState::assign_gp(self, val, r)
    }

    fn assign_fp(&mut self, val: Value, r: u8) {
        GreedyRegState::assign_fp(self, val, r)
    }

    fn mark_gp_occupied(&mut self, r: u8, val: Value) {
        self.gp_occupant[r as usize] = Some(val);
    }

    fn clear_gp_occupied(&mut self, r: u8, val: Value) {
        if self.gp_occupant[r as usize] == Some(val) {
            self.gp_occupant[r as usize] = None;
        }
    }

    fn set_spill_slot(&mut self, val: Value, offset: i32) {
        self.values[val.index()].spill_slot = Some(offset);
    }

    fn set_value_loc(&mut self, val: Value, loc: ValueLoc) {
        self.values[val.index()].loc = loc;
    }

    fn set_type(&mut self, val: Value, ty: Type) {
        self.values[val.index()].ty = ty;
    }

    fn dec_use(&mut self, val: Value) {
        GreedyRegState::dec_use(self, val)
    }

    fn inc_use(&mut self, val: Value) {
        self.values[val.index()].remaining_uses += 1;
    }

    fn spill_caller_saved<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        GreedyRegState::spill_caller_saved::<B>(self, buf, frame)
    }

    fn spill_all_live<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        GreedyRegState::spill_all_live::<B>(self, buf, frame)
    }

    fn clear_regs(&mut self) {
        GreedyRegState::clear_regs(self)
    }

    fn enter_block<B: LoweringBackend>(
        &mut self,
        block_idx: usize,
        func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        if block_idx == 0 {
            return; // Entry block params handled by Lowerer (calling convention).
        }
        self.clear_regs();
        let block = &func.blocks[block_idx];
        let param_slots: Vec<i32> = frame.block_param_slots(block_idx).to_vec();
        for (i, &(val, ty)) in block.params.iter().enumerate() {
            let slot = frame.slot_access(param_slots[i]);
            if is_float_type(ty) {
                let r = self.alloc_fp::<B>(buf, frame);
                B::emit_load_fp_from_frame(buf, machine_fp(r), slot);
                self.assign_fp(val, r);
            } else {
                let r = self.alloc_gp::<B>(buf, frame);
                B::emit_load_gp_from_frame(buf, machine_gp(r), slot);
                self.assign_gp(val, r);
            }
        }
    }

    fn emit_block_args<B: LoweringBackend>(
        &mut self,
        target_idx: usize,
        args: &[Value],
        _func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        if args.is_empty() {
            return;
        }
        // Collect slot accesses upfront to avoid borrow conflicts.
        let slot_accesses: Vec<_> = frame
            .block_param_slots(target_idx)
            .iter()
            .map(|&off| frame.slot_access(off))
            .collect();
        for (i, &arg) in args.iter().enumerate() {
            let ty = self.values[arg.index()].ty;
            if is_float_type(ty) {
                let src = self.ensure_in_fp_reg::<B>(buf, frame, arg);
                B::emit_store_fp_to_frame(buf, machine_fp(src), slot_accesses[i]);
            } else {
                let src = self.ensure_in_gp_reg::<B>(buf, frame, arg);
                B::emit_store_gp_to_frame(buf, machine_gp(src), slot_accesses[i]);
            }
        }
        for &arg in args {
            self.dec_use(arg);
        }
    }

    fn clobber_caller_saved<B: LoweringBackend>(&mut self) {
        for &r in B::caller_saved_gp() {
            if let Some(val) = self.gp_occupant[r as usize] {
                if let Some(slot) = self.values[val.index()].spill_slot {
                    self.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.gp_occupant[r as usize] = None;
            }
        }
        for &r in B::caller_saved_fp() {
            if let Some(val) = self.fp_occupant[r as usize] {
                if let Some(slot) = self.values[val.index()].spill_slot {
                    self.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.fp_occupant[r as usize] = None;
            }
        }
    }

    type SavedState = GreedySavedState;

    fn save_state(&self) -> GreedySavedState {
        GreedySavedState {
            values: self.values.iter().map(|v| (v.loc, v.spill_slot)).collect(),
            gp_occupant: self.gp_occupant,
            fp_occupant: self.fp_occupant,
        }
    }

    fn restore_state(&mut self, state: GreedySavedState) {
        for (i, (loc, spill)) in state.values.into_iter().enumerate() {
            self.values[i].loc = loc;
            self.values[i].spill_slot = spill;
        }
        self.gp_occupant = state.gp_occupant;
        self.fp_occupant = state.fp_occupant;
    }

    fn gp_occupant(&self, r: usize) -> Option<Value> {
        self.gp_occupant[r]
    }

    fn fp_occupant(&self, r: usize) -> Option<Value> {
        self.fp_occupant[r]
    }

    fn num_values(&self) -> usize {
        self.values.len()
    }

    fn value_info_snapshot(&self) -> Vec<(Option<i32>, Type)> {
        self.values.iter().map(|v| (v.spill_slot, v.ty)).collect()
    }
}

// ─── Linear Scan Register Allocator ──────────────────────────────

/// A live interval for a single SSA value.
#[derive(Debug, Clone)]
struct LiveInterval {
    value: Value,
    #[allow(dead_code)]
    ty: Type,
    start: u32,         // inclusive
    end: u32,           // inclusive
    reg: Option<u8>,    // assigned physical register (after allocation)
    spill: Option<i32>, // spill slot offset (if spilled)
    is_float: bool,
}

/// Pre-computed assignment for a block parameter at a specific edge.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PhiMove {
    src: Value,             // Value in predecessor
    dst_reg: Option<u8>,    // Target register for the block param
    dst_spill: Option<i32>, // Target spill slot if param is spilled
    is_float: bool,
}

/// A single move in a parallel copy resolution.
#[derive(Debug, Clone, Copy)]
struct ParallelMove {
    src: MoveLocation,
    dst: MoveLocation,
    is_float: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveLocation {
    GpReg(u8),
    FpReg(u8),
    Spill(i32),
}

pub struct LinearScanAllocator {
    /// Per-value runtime state — tracks current location during lowering.
    values: Vec<ValueInfo>,
    /// GP register occupancy (mirrors GreedyRegState for compat).
    gp_occupant: [Option<Value>; NUM_GP],
    /// FP register occupancy.
    fp_occupant: [Option<Value>; NUM_FP],
    /// Pre-computed register assignment per value (from linear scan).
    /// None = spilled.
    assignments: Vec<Option<u8>>,
    /// Pre-computed spill slot per value (from linear scan). Assigned during prepare.
    assigned_spills: Vec<Option<i32>>,
    /// Position map: (block_idx, inst_idx) -> linear position.
    /// block_positions[block_idx] = position of block entry.
    block_positions: Vec<u32>,
    /// For each block with params, the assigned registers for each param.
    block_param_regs: Vec<Vec<Option<u8>>>,
    /// Current lowering position (updated as we process instructions).
    #[allow(dead_code)]
    current_pos: u32,
    /// Number of spill slots allocated during prepare.
    next_spill_offset: i32,
    /// Eviction cursor for fallback allocation.
    gp_evict_cursor: usize,
    fp_evict_cursor: usize,
    /// Whether prepare() has been called.
    prepared: bool,
}

impl LinearScanAllocator {
    /// Compute live intervals for all values in the function.
    fn compute_live_intervals(func: &Function) -> Vec<LiveInterval> {
        let rpo = opt::compute_rpo(func);
        let preds = opt::compute_predecessors(func);

        // Assign linear positions: each block entry gets a position,
        // each instruction gets a position, terminator gets a position.
        let mut block_start: Vec<u32> = vec![0; func.blocks.len()];
        let mut block_end: Vec<u32> = vec![0; func.blocks.len()];
        let mut pos: u32 = 0;
        for &bi in &rpo {
            block_start[bi] = pos;
            pos += 1; // block entry
            pos += func.blocks[bi].insts.len() as u32; // instructions
            pos += 1; // terminator
            block_end[bi] = pos - 1;
        }

        let num_values = func.value_types.len();
        let mut interval_start = vec![u32::MAX; num_values];
        let mut interval_end = vec![0u32; num_values];

        // Define: block params are defined at block entry position.
        for &bi in &rpo {
            let entry_pos = block_start[bi];
            for (_, (val, _ty)) in func.blocks[bi].params.iter().enumerate() {
                let vi = val.index();
                interval_start[vi] = interval_start[vi].min(entry_pos);
                interval_end[vi] = interval_end[vi].max(entry_pos);
            }
        }

        // Define: instruction results are defined at their instruction position.
        for &bi in &rpo {
            let base = block_start[bi] + 1; // skip block entry
            for (ii, node) in func.blocks[bi].insts.iter().enumerate() {
                if let Some(val) = node.value {
                    let p = base + ii as u32;
                    let vi = val.index();
                    interval_start[vi] = interval_start[vi].min(p);
                    interval_end[vi] = interval_end[vi].max(p);
                }
            }
        }

        // Uses: instruction operands extend interval to their use position.
        for &bi in &rpo {
            let base = block_start[bi] + 1;
            for (ii, node) in func.blocks[bi].insts.iter().enumerate() {
                let p = base + ii as u32;
                node.inst.for_each_value(|v| {
                    let vi = v.index();
                    interval_end[vi] = interval_end[vi].max(p);
                });
            }
            // Terminator uses
            let term_pos = block_end[bi];
            func.blocks[bi].terminator.for_each_value(|v| {
                let vi = v.index();
                interval_end[vi] = interval_end[vi].max(term_pos);
            });
        }

        // Block args: values passed as args to successor blocks must be live
        // at the end of the current block. Also, block params must be live
        // through their entire block (at minimum).
        for &bi in &rpo {
            func.blocks[bi]
                .terminator
                .for_each_successor_args(|target, args| {
                    for &arg in args {
                        // Arg value must be live up to end of this block
                        let vi = arg.index();
                        interval_end[vi] = interval_end[vi].max(block_end[bi]);
                    }
                    // Target block params must be live from block start
                    for (pi, (pval, _)) in func.blocks[target.index()].params.iter().enumerate() {
                        let _ = pi;
                        let vi = pval.index();
                        // Param is live from block start to at least block start
                        interval_start[vi] = interval_start[vi].min(block_start[target.index()]);
                    }
                });
        }

        // Extend intervals for values live across block boundaries:
        // If a value is used in a block but defined in a dominator,
        // extend it through all intermediate blocks.
        // Simple approach: iterate blocks in reverse RPO, and for each use,
        // extend the interval back through predecessors until we reach the def.
        let mut changed = true;
        while changed {
            changed = false;
            for &bi in rpo.iter().rev() {
                for (pi, (pval, _)) in func.blocks[bi].params.iter().enumerate() {
                    let _ = pi;
                    // Block params get their value from predecessors.
                    // The arg value in each predecessor must be live to end of pred.
                    // (Already handled above via for_each_successor_args)
                    let _ = pval;
                }
                // For each value used in this block that's defined before this block,
                // extend it to cover from its current start through this block.
                let block_s = block_start[bi];
                let block_e = block_end[bi];
                for node in &func.blocks[bi].insts {
                    node.inst.for_each_value(|v| {
                        let vi = v.index();
                        if interval_start[vi] < block_s && interval_end[vi] >= block_s {
                            // Value defined before this block and used here.
                            // Make sure it's live through the whole block.
                            if interval_end[vi] < block_e {
                                interval_end[vi] = block_e;
                                changed = true;
                            }
                        }
                    });
                }
                func.blocks[bi].terminator.for_each_value(|v| {
                    let vi = v.index();
                    if interval_start[vi] < block_s && interval_end[vi] >= block_s {
                        if interval_end[vi] < block_e {
                            interval_end[vi] = block_e;
                            changed = true;
                        }
                    }
                });
                // Propagate liveness through predecessors:
                // If a value is live at the start of this block, it must be
                // live at the end of all predecessors.
                for val_idx in 0..num_values {
                    if interval_start[val_idx] < block_s && interval_end[val_idx] >= block_s {
                        for &pred in &preds[bi] {
                            let pred_end = block_end[pred];
                            if interval_end[val_idx] < pred_end
                                && interval_start[val_idx] <= block_start[pred]
                            {
                                // Shouldn't happen: value is defined before pred
                                // but somehow not live through it. Extend.
                            }
                        }
                    }
                }
            }
        }

        // Build intervals for values that are actually defined.
        let mut intervals = Vec::new();
        for vi in 0..num_values {
            if interval_start[vi] != u32::MAX {
                intervals.push(LiveInterval {
                    value: Value::from_index(vi),
                    ty: func.value_types[vi],
                    start: interval_start[vi],
                    end: interval_end[vi],
                    reg: None,
                    spill: None,
                    is_float: func.value_types[vi] == Type::F64,
                });
            }
        }

        intervals
    }

    /// Run the linear scan allocation algorithm.
    fn allocate<B: LoweringBackend>(
        intervals: &mut [LiveInterval],
        func: &Function,
    ) -> (Vec<u32>, i32) {
        // Sort by start position.
        intervals.sort_by_key(|iv| iv.start);

        // Compute block positions for later use.
        let rpo = opt::compute_rpo(func);
        let mut block_positions = vec![0u32; func.blocks.len()];
        let mut pos: u32 = 0;
        for &bi in &rpo {
            block_positions[bi] = pos;
            pos += 1 + func.blocks[bi].insts.len() as u32 + 1;
        }

        // Find call positions (values live across calls need spill slots).
        let mut call_positions: Vec<u32> = Vec::new();
        for &bi in &rpo {
            let base = block_positions[bi] + 1;
            for (ii, node) in func.blocks[bi].insts.iter().enumerate() {
                match &node.inst {
                    Inst::Call(..) | Inst::CallIndirect(..) => {
                        call_positions.push(base + ii as u32);
                    }
                    _ => {}
                }
            }
        }

        // Active intervals sorted by end position.
        let mut active_gp: Vec<usize> = Vec::new(); // indices into intervals
        let mut active_fp: Vec<usize> = Vec::new();
        let mut free_gp: Vec<u8> = B::allocatable_gp().to_vec();
        let mut free_fp: Vec<u8> = B::allocatable_fp().to_vec();
        let mut next_spill: i32 = -8; // first spill slot

        for i in 0..intervals.len() {
            let start = intervals[i].start;

            // Expire old intervals.
            active_gp.retain(|&ai| {
                if intervals[ai].end < start {
                    if let Some(r) = intervals[ai].reg {
                        free_gp.push(r);
                    }
                    false
                } else {
                    true
                }
            });
            active_fp.retain(|&ai| {
                if intervals[ai].end < start {
                    if let Some(r) = intervals[ai].reg {
                        free_fp.push(r);
                    }
                    false
                } else {
                    true
                }
            });

            // Check if this interval crosses a call. If so, it MUST be spilled
            // around calls since all our allocatable registers are caller-saved.
            let crosses_call = call_positions
                .iter()
                .any(|&cp| cp > start && cp <= intervals[i].end);

            if intervals[i].is_float {
                if let Some(r) = free_fp.pop() {
                    intervals[i].reg = Some(r);
                    active_fp.push(i);
                    active_fp.sort_by_key(|&ai| intervals[ai].end);
                    // If crosses call, also assign a spill slot for save/restore.
                    if crosses_call {
                        intervals[i].spill = Some(next_spill);
                        next_spill -= 8;
                    }
                } else {
                    // Spill: pick active with furthest end.
                    if let Some(&spill_idx) = active_fp.last() {
                        if intervals[spill_idx].end > intervals[i].end {
                            // Spill the longer-lived interval.
                            let r = intervals[spill_idx].reg.unwrap();
                            intervals[spill_idx].reg = None;
                            if intervals[spill_idx].spill.is_none() {
                                intervals[spill_idx].spill = Some(next_spill);
                                next_spill -= 8;
                            }
                            active_fp.pop();
                            intervals[i].reg = Some(r);
                            active_fp.push(i);
                            active_fp.sort_by_key(|&ai| intervals[ai].end);
                        } else {
                            // Spill current.
                            intervals[i].spill = Some(next_spill);
                            next_spill -= 8;
                        }
                    } else {
                        intervals[i].spill = Some(next_spill);
                        next_spill -= 8;
                    }
                }
            } else {
                // GP allocation
                if let Some(r) = free_gp.pop() {
                    intervals[i].reg = Some(r);
                    active_gp.push(i);
                    active_gp.sort_by_key(|&ai| intervals[ai].end);
                    if crosses_call {
                        intervals[i].spill = Some(next_spill);
                        next_spill -= 8;
                    }
                } else {
                    if let Some(&spill_idx) = active_gp.last() {
                        if intervals[spill_idx].end > intervals[i].end {
                            let r = intervals[spill_idx].reg.unwrap();
                            intervals[spill_idx].reg = None;
                            if intervals[spill_idx].spill.is_none() {
                                intervals[spill_idx].spill = Some(next_spill);
                                next_spill -= 8;
                            }
                            active_gp.pop();
                            intervals[i].reg = Some(r);
                            active_gp.push(i);
                            active_gp.sort_by_key(|&ai| intervals[ai].end);
                        } else {
                            intervals[i].spill = Some(next_spill);
                            next_spill -= 8;
                        }
                    } else {
                        intervals[i].spill = Some(next_spill);
                        next_spill -= 8;
                    }
                }
            }
        }

        (block_positions, next_spill)
    }
}

/// Resolve a set of parallel moves, emitting them in a safe order.
/// Uses `temp_reg` (typically X27/D27) to break cycles.
fn resolve_parallel_moves<B: LoweringBackend>(
    moves: &[ParallelMove],
    buf: &mut CodeBuffer<B::Arch>,
    frame: &impl FrameLayout,
) {
    if moves.is_empty() {
        return;
    }

    // Filter out self-moves.
    let moves: Vec<&ParallelMove> = moves.iter().filter(|m| m.src != m.dst).collect();
    if moves.is_empty() {
        return;
    }

    // Build a dependency map: dst -> src.
    // A move is blocked if its dst is someone else's src.
    let mut pending: Vec<bool> = vec![true; moves.len()];
    let mut emitted = 0;
    let total = moves.len();

    // Simple iterative resolution: emit moves whose dst is not a src of
    // any pending move. Repeat until done. If stuck, break cycle with temp.
    while emitted < total {
        let mut progress = false;
        for i in 0..moves.len() {
            if !pending[i] {
                continue;
            }
            let dst = moves[i].dst;
            // Check if dst is used as src by any other pending move.
            let blocked = moves
                .iter()
                .enumerate()
                .any(|(j, m)| j != i && pending[j] && m.src == dst);
            if !blocked {
                emit_single_move::<B>(moves[i], buf, frame);
                pending[i] = false;
                emitted += 1;
                progress = true;
            }
        }
        if !progress {
            // Cycle detected. Break it using temp register (X27 for GP, index 27).
            // Find a pending move, save its dst to temp, then emit it.
            for i in 0..moves.len() {
                if pending[i] {
                    let temp = if moves[i].is_float {
                        MoveLocation::FpReg(27)
                    } else {
                        MoveLocation::GpReg(27)
                    };
                    // Save: move dst -> temp (preserve the value being overwritten)
                    emit_single_move::<B>(
                        &ParallelMove {
                            src: moves[i].dst,
                            dst: temp,
                            is_float: moves[i].is_float,
                        },
                        buf,
                        frame,
                    );
                    // Emit the blocked move: src -> dst
                    emit_single_move::<B>(moves[i], buf, frame);
                    pending[i] = false;
                    emitted += 1;
                    // Now update all remaining moves that use the overwritten dst as src
                    // to use temp instead. We need to find them and emit them.
                    // Actually, the original value is in temp now. Any pending move
                    // that had src == moves[i].dst should now read from temp.
                    // Since we can't modify the moves slice, just emit the rest
                    // that depend on this value using temp.
                    for j in 0..moves.len() {
                        if pending[j] && moves[j].src == moves[i].dst {
                            emit_single_move::<B>(
                                &ParallelMove {
                                    src: temp,
                                    dst: moves[j].dst,
                                    is_float: moves[j].is_float,
                                },
                                buf,
                                frame,
                            );
                            pending[j] = false;
                            emitted += 1;
                        }
                    }
                    break;
                }
            }
        }
    }
}

fn emit_single_move<B: LoweringBackend>(
    m: &ParallelMove,
    buf: &mut CodeBuffer<B::Arch>,
    frame: &impl FrameLayout,
) {
    match (m.src, m.dst) {
        (MoveLocation::GpReg(src), MoveLocation::GpReg(dst)) => {
            if src != dst {
                B::emit_gp_move(buf, machine_gp(dst), machine_gp(src));
            }
        }
        (MoveLocation::FpReg(src), MoveLocation::FpReg(dst)) => {
            if src != dst {
                // No direct FP-FP move; go through GP temp register X27.
                let temp = machine_gp(27);
                B::emit_fp_to_gp_move(buf, temp, machine_fp(src));
                B::emit_gp_to_fp_move(buf, machine_fp(dst), temp);
            }
        }
        (MoveLocation::Spill(off), MoveLocation::GpReg(dst)) => {
            B::emit_load_gp_from_frame(buf, machine_gp(dst), frame.slot_access(off));
        }
        (MoveLocation::GpReg(src), MoveLocation::Spill(off)) => {
            B::emit_store_gp_to_frame(buf, machine_gp(src), frame.slot_access(off));
        }
        (MoveLocation::Spill(off), MoveLocation::FpReg(dst)) => {
            B::emit_load_fp_from_frame(buf, machine_fp(dst), frame.slot_access(off));
        }
        (MoveLocation::FpReg(src), MoveLocation::Spill(off)) => {
            B::emit_store_fp_to_frame(buf, machine_fp(src), frame.slot_access(off));
        }
        (MoveLocation::Spill(src_off), MoveLocation::Spill(dst_off)) => {
            if src_off != dst_off {
                // Spill-to-spill: use temp GP register.
                let temp = machine_gp(27);
                B::emit_load_gp_from_frame(buf, temp, frame.slot_access(src_off));
                B::emit_store_gp_to_frame(buf, temp, frame.slot_access(dst_off));
            }
        }
        _ => {
            // Cross-class moves (GP <-> FP) through temp.
            // Shouldn't happen in well-typed IR.
            panic!("unsupported cross-class move in parallel copy");
        }
    }
}

impl RegisterAllocator for LinearScanAllocator {
    fn new(num_values: usize) -> Self {
        LinearScanAllocator {
            values: (0..num_values)
                .map(|_| ValueInfo {
                    loc: ValueLoc::Unassigned,
                    spill_slot: None,
                    remaining_uses: 0,
                    ty: Type::I64,
                })
                .collect(),
            gp_occupant: [None; NUM_GP],
            fp_occupant: [None; NUM_FP],
            assignments: vec![None; num_values],
            assigned_spills: vec![None; num_values],
            block_positions: Vec::new(),
            block_param_regs: Vec::new(),
            current_pos: 0,
            next_spill_offset: 0,
            gp_evict_cursor: 0,
            fp_evict_cursor: 0,
            prepared: false,
        }
    }

    fn prepare<B: LoweringBackend>(&mut self, func: &Function) {
        let mut intervals = Self::compute_live_intervals(func);
        let (block_positions, next_spill) = Self::allocate::<B>(&mut intervals, func);
        self.block_positions = block_positions;
        self.next_spill_offset = next_spill;

        // Store assignments into per-value arrays.
        for iv in &intervals {
            let vi = iv.value.index();
            self.assignments[vi] = iv.reg;
            // Linear-scan spill ids are allocation decisions, not concrete
            // frame-layout offsets. The streaming lowerer allocates frame
            // slots lazily and block params without registers use their
            // canonical block-param slots.
            self.assigned_spills[vi] = None;
        }

        // Pre-compute block param register assignments.
        self.block_param_regs = func
            .blocks
            .iter()
            .map(|block| {
                block
                    .params
                    .iter()
                    .map(|(val, _ty)| self.assignments[val.index()])
                    .collect()
            })
            .collect();

        self.prepared = true;
    }

    fn value_type(&self, val: Value) -> Type {
        self.values[val.index()].ty
    }

    fn value_loc(&self, val: Value) -> ValueLoc {
        self.values[val.index()].loc
    }

    fn value_spill_slot(&self, val: Value) -> Option<i32> {
        self.values[val.index()].spill_slot
    }

    fn remaining_uses(&self, val: Value) -> u32 {
        self.values[val.index()].remaining_uses
    }

    fn alloc_gp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8 {
        // If prepared, the caller is lowering an instruction whose result has
        // a pre-assigned register. But the lowerer calls alloc_gp generically.
        // We fall back to greedy allocation for robustness.
        let regs = B::allocatable_gp();
        for &r in regs {
            if self.gp_occupant[r as usize].is_none() {
                return r;
            }
        }
        // Evict round-robin.
        let start = self.gp_evict_cursor;
        for i in 0..regs.len() {
            let idx = (start + i) % regs.len();
            let r = regs[idx];
            if let Some(val) = self.gp_occupant[r as usize] {
                // Spill the evicted value.
                if self.values[val.index()].remaining_uses > 0 {
                    let offset = match self.values[val.index()].spill_slot {
                        Some(off) => off,
                        None => {
                            let off = frame.alloc_root_slot();
                            self.values[val.index()].spill_slot = Some(off);
                            off
                        }
                    };
                    B::emit_store_gp_to_frame(buf, machine_gp(r), frame.slot_access(offset));
                    self.values[val.index()].loc = ValueLoc::Spill(offset);
                }
                self.gp_occupant[r as usize] = None;
                self.gp_evict_cursor = (idx + 1) % regs.len();
                return r;
            }
        }
        panic!("no GP register available");
    }

    fn alloc_fp<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) -> u8 {
        let regs = B::allocatable_fp();
        for &r in regs {
            if self.fp_occupant[r as usize].is_none() {
                return r;
            }
        }
        let start = self.fp_evict_cursor;
        for i in 0..regs.len() {
            let idx = (start + i) % regs.len();
            let r = regs[idx];
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    let offset = match self.values[val.index()].spill_slot {
                        Some(off) => off,
                        None => {
                            let off = frame.alloc_root_slot();
                            self.values[val.index()].spill_slot = Some(off);
                            off
                        }
                    };
                    B::emit_store_fp_to_frame(buf, machine_fp(r), frame.slot_access(offset));
                    self.values[val.index()].loc = ValueLoc::Spill(offset);
                }
                self.fp_occupant[r as usize] = None;
                self.fp_evict_cursor = (idx + 1) % regs.len();
                return r;
            }
        }
        panic!("no FP register available");
    }

    fn ensure_in_gp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8 {
        match self.values[val.index()].loc {
            ValueLoc::GpReg(r) => r,
            ValueLoc::Spill(offset) => {
                // Check if this value has a pre-assigned register.
                let target_reg = self.assignments.get(val.index()).copied().flatten();
                let r = if let Some(tr) = target_reg {
                    // Try to use the pre-assigned register.
                    if self.gp_occupant[tr as usize].is_none() {
                        tr
                    } else if self.gp_occupant[tr as usize] == Some(val) {
                        tr // already there
                    } else {
                        // Pre-assigned reg is occupied, use any free.
                        self.alloc_gp::<B>(buf, frame)
                    }
                } else {
                    self.alloc_gp::<B>(buf, frame)
                };
                B::emit_load_gp_from_frame(buf, machine_gp(r), frame.slot_access(offset));
                if self.values[val.index()].spill_slot.is_none() {
                    self.values[val.index()].spill_slot = Some(offset);
                }
                self.gp_occupant[r as usize] = Some(val);
                self.values[val.index()].loc = ValueLoc::GpReg(r);
                r
            }
            ValueLoc::FpReg(_) => panic!("expected GP value, got FP"),
            ValueLoc::Unassigned => panic!("value v{} is unassigned", val.index()),
        }
    }

    fn ensure_in_fp_reg<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
        val: Value,
    ) -> u8 {
        match self.values[val.index()].loc {
            ValueLoc::FpReg(r) => r,
            ValueLoc::Spill(offset) => {
                let r = self.alloc_fp::<B>(buf, frame);
                B::emit_load_fp_from_frame(buf, machine_fp(r), frame.slot_access(offset));
                self.fp_occupant[r as usize] = Some(val);
                self.values[val.index()].loc = ValueLoc::FpReg(r);
                r
            }
            ValueLoc::GpReg(_) => panic!("expected FP value, got GP"),
            ValueLoc::Unassigned => panic!("value v{} is unassigned", val.index()),
        }
    }

    fn assign_gp(&mut self, val: Value, r: u8) {
        self.gp_occupant[r as usize] = Some(val);
        self.values[val.index()].loc = ValueLoc::GpReg(r);
    }

    fn assign_fp(&mut self, val: Value, r: u8) {
        self.fp_occupant[r as usize] = Some(val);
        self.values[val.index()].loc = ValueLoc::FpReg(r);
    }

    fn mark_gp_occupied(&mut self, r: u8, val: Value) {
        self.gp_occupant[r as usize] = Some(val);
    }

    fn clear_gp_occupied(&mut self, r: u8, val: Value) {
        if self.gp_occupant[r as usize] == Some(val) {
            self.gp_occupant[r as usize] = None;
        }
    }

    fn set_spill_slot(&mut self, val: Value, offset: i32) {
        self.values[val.index()].spill_slot = Some(offset);
    }

    fn set_value_loc(&mut self, val: Value, loc: ValueLoc) {
        self.values[val.index()].loc = loc;
    }

    fn set_type(&mut self, val: Value, ty: Type) {
        self.values[val.index()].ty = ty;
    }

    fn dec_use(&mut self, val: Value) {
        let info = &mut self.values[val.index()];
        info.remaining_uses = info.remaining_uses.saturating_sub(1);
        if info.remaining_uses == 0 {
            match info.loc {
                ValueLoc::GpReg(r) => {
                    self.gp_occupant[r as usize] = None;
                }
                ValueLoc::FpReg(r) => {
                    self.fp_occupant[r as usize] = None;
                }
                _ => {}
            }
        }
    }

    fn inc_use(&mut self, val: Value) {
        self.values[val.index()].remaining_uses += 1;
    }

    fn spill_caller_saved<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        // Same as greedy: spill all values in caller-saved registers.
        for &r in B::caller_saved_gp() {
            if let Some(val) = self.gp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    let offset = match self.values[val.index()].spill_slot {
                        Some(off) => off,
                        None => {
                            let off = frame.alloc_root_slot();
                            self.values[val.index()].spill_slot = Some(off);
                            off
                        }
                    };
                    B::emit_store_gp_to_frame(buf, machine_gp(r), frame.slot_access(offset));
                    self.values[val.index()].loc = ValueLoc::Spill(offset);
                    self.gp_occupant[r as usize] = None;
                } else {
                    self.gp_occupant[r as usize] = None;
                    self.values[val.index()].loc = ValueLoc::Unassigned;
                }
            }
        }
        for &r in B::caller_saved_fp() {
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    let offset = match self.values[val.index()].spill_slot {
                        Some(off) => off,
                        None => {
                            let off = frame.alloc_root_slot();
                            self.values[val.index()].spill_slot = Some(off);
                            off
                        }
                    };
                    B::emit_store_fp_to_frame(buf, machine_fp(r), frame.slot_access(offset));
                    self.values[val.index()].loc = ValueLoc::Spill(offset);
                    self.fp_occupant[r as usize] = None;
                } else {
                    self.fp_occupant[r as usize] = None;
                    self.values[val.index()].loc = ValueLoc::Unassigned;
                }
            }
        }
    }

    fn spill_all_live<B: LoweringBackend>(
        &mut self,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        for r in 0..NUM_GP as u8 {
            if let Some(val) = self.gp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    let offset = match self.values[val.index()].spill_slot {
                        Some(off) => off,
                        None => {
                            let off = frame.alloc_root_slot();
                            self.values[val.index()].spill_slot = Some(off);
                            off
                        }
                    };
                    B::emit_store_gp_to_frame(buf, machine_gp(r), frame.slot_access(offset));
                    self.values[val.index()].loc = ValueLoc::Spill(offset);
                } else {
                    self.values[val.index()].loc = ValueLoc::Unassigned;
                }
                self.gp_occupant[r as usize] = None;
            }
        }
        for r in 0..NUM_FP as u8 {
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    let offset = match self.values[val.index()].spill_slot {
                        Some(off) => off,
                        None => {
                            let off = frame.alloc_root_slot();
                            self.values[val.index()].spill_slot = Some(off);
                            off
                        }
                    };
                    B::emit_store_fp_to_frame(buf, machine_fp(r), frame.slot_access(offset));
                    self.values[val.index()].loc = ValueLoc::Spill(offset);
                } else {
                    self.values[val.index()].loc = ValueLoc::Unassigned;
                }
                self.fp_occupant[r as usize] = None;
            }
        }
    }

    fn clear_regs(&mut self) {
        for info in &mut self.values {
            match info.loc {
                ValueLoc::GpReg(_) | ValueLoc::FpReg(_) => {
                    if let Some(off) = info.spill_slot {
                        info.loc = ValueLoc::Spill(off);
                    } else {
                        info.loc = ValueLoc::Unassigned;
                    }
                }
                _ => {}
            }
        }
        self.gp_occupant = [None; NUM_GP];
        self.fp_occupant = [None; NUM_FP];
    }

    fn enter_block<B: LoweringBackend>(
        &mut self,
        block_idx: usize,
        func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        if block_idx == 0 {
            return;
        }

        if !self.prepared {
            // Fall back to greedy behavior if not prepared.
            self.clear_regs();
            let param_slots: Vec<i32> = frame.block_param_slots(block_idx).to_vec();
            for (i, &(val, ty)) in func.blocks[block_idx].params.iter().enumerate() {
                let slot = frame.slot_access(param_slots[i]);
                if is_float_type(ty) {
                    let r = self.alloc_fp::<B>(buf, frame);
                    B::emit_load_fp_from_frame(buf, machine_fp(r), slot);
                    self.assign_fp(val, r);
                } else {
                    let r = self.alloc_gp::<B>(buf, frame);
                    B::emit_load_gp_from_frame(buf, machine_gp(r), slot);
                    self.assign_gp(val, r);
                }
            }
            return;
        }

        // Linear scan: block params arrive in their pre-assigned registers
        // (placed there by emit_block_args at the predecessor). Set up
        // register state to match.
        self.clear_regs();
        let param_slots = frame.block_param_slots(block_idx).to_vec();
        for (param_idx, &(val, ty)) in func.blocks[block_idx].params.iter().enumerate() {
            let vi = val.index();
            if let Some(r) = self.assignments[vi] {
                if is_float_type(ty) {
                    self.fp_occupant[r as usize] = Some(val);
                    self.values[vi].loc = ValueLoc::FpReg(r);
                } else {
                    self.gp_occupant[r as usize] = Some(val);
                    self.values[vi].loc = ValueLoc::GpReg(r);
                }
            } else if let Some(off) = self.assigned_spills[vi] {
                self.values[vi].loc = ValueLoc::Spill(off);
                self.values[vi].spill_slot = Some(off);
            } else if let Some(&off) = param_slots.get(param_idx) {
                self.values[vi].loc = ValueLoc::Spill(off);
                self.values[vi].spill_slot = Some(off);
            }
        }

        // Also restore locations for values defined in dominating blocks
        // that are still live at this point. They should still be in their
        // spill slots (they were spilled before the branch).
        // The clear_regs above already reverted register values to their
        // spill slots, which is correct.
    }

    fn emit_block_args<B: LoweringBackend>(
        &mut self,
        target_idx: usize,
        args: &[Value],
        func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        if args.is_empty() {
            return;
        }

        if !self.prepared {
            // Fall back to greedy: store to canonical slots.
            let slot_accesses: Vec<_> = frame
                .block_param_slots(target_idx)
                .iter()
                .map(|&off| frame.slot_access(off))
                .collect();
            for (i, &arg) in args.iter().enumerate() {
                let ty = self.values[arg.index()].ty;
                if is_float_type(ty) {
                    let src = self.ensure_in_fp_reg::<B>(buf, frame, arg);
                    B::emit_store_fp_to_frame(buf, machine_fp(src), slot_accesses[i]);
                } else {
                    let src = self.ensure_in_gp_reg::<B>(buf, frame, arg);
                    B::emit_store_gp_to_frame(buf, machine_gp(src), slot_accesses[i]);
                }
            }
            for &arg in args {
                self.dec_use(arg);
            }
            return;
        }

        // Linear scan: emit parallel moves from current arg locations
        // to the target block params' assigned registers.
        let target_params = &func.blocks[target_idx].params;
        let mut moves: Vec<ParallelMove> = Vec::new();

        for (i, &arg) in args.iter().enumerate() {
            let (target_val, ty) = target_params[i];
            let tvi = target_val.index();
            let is_fp = is_float_type(ty);

            // Where is the arg value now?
            let src = match self.values[arg.index()].loc {
                ValueLoc::GpReg(r) => MoveLocation::GpReg(r),
                ValueLoc::FpReg(r) => MoveLocation::FpReg(r),
                ValueLoc::Spill(off) => MoveLocation::Spill(off),
                ValueLoc::Unassigned => {
                    panic!("emit_block_args: arg v{} is unassigned", arg.index());
                }
            };

            // Where does the target param want to be?
            let dst = if let Some(r) = self.assignments[tvi] {
                if is_fp {
                    MoveLocation::FpReg(r)
                } else {
                    MoveLocation::GpReg(r)
                }
            } else if let Some(off) = self.assigned_spills[tvi] {
                MoveLocation::Spill(off)
            } else {
                // No assignment — use canonical slot as fallback.
                let slots = frame.block_param_slots(target_idx);
                MoveLocation::Spill(slots[i])
            };

            moves.push(ParallelMove {
                src,
                dst,
                is_float: is_fp,
            });
        }

        resolve_parallel_moves::<B>(&moves, buf, frame);

        for &arg in args {
            self.dec_use(arg);
        }
    }

    fn clobber_caller_saved<B: LoweringBackend>(&mut self) {
        for &r in B::caller_saved_gp() {
            if let Some(val) = self.gp_occupant[r as usize] {
                if let Some(slot) = self.values[val.index()].spill_slot {
                    self.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.gp_occupant[r as usize] = None;
            }
        }
        for &r in B::caller_saved_fp() {
            if let Some(val) = self.fp_occupant[r as usize] {
                if let Some(slot) = self.values[val.index()].spill_slot {
                    self.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.fp_occupant[r as usize] = None;
            }
        }
    }

    fn place_call_return_in_block_param<B: LoweringBackend>(
        &mut self,
        target_idx: usize,
        param_idx: usize,
        ret_ty: Type,
        func: &Function,
        buf: &mut CodeBuffer<B::Arch>,
        frame: &mut impl FrameLayout,
    ) {
        // Linear scan: place x0/fp0 into the location pre-assigned
        // to the target block's param. Without this override, the
        // default impl (store-to-canonical-slot) would be invisible
        // to LinearScan's `enter_block` — which expects block-param
        // values in registers (when assigned) and never loads from
        // canonical slots.
        let (param_val, _) = func.blocks[target_idx].params[param_idx];
        let vi = param_val.index();
        let is_fp = is_float_type(ret_ty);

        if let Some(r) = self.assignments[vi] {
            // Assigned to register r: move from machine reg 0 → r.
            if is_fp {
                if r != 0 {
                    // No direct FP-FP move on ARM64; bounce through
                    // X27 (matches `resolve_parallel_moves`).
                    let temp = machine_gp(27);
                    B::emit_fp_to_gp_move(buf, temp, machine_fp(0));
                    B::emit_gp_to_fp_move(buf, machine_fp(r), temp);
                }
            } else if r != 0 {
                B::emit_gp_move(buf, machine_gp(r), machine_gp(0));
            }
        } else if let Some(off) = self.assigned_spills[vi] {
            let slot = frame.slot_access(off);
            if is_fp {
                B::emit_store_fp_to_frame(buf, machine_fp(0), slot);
            } else {
                B::emit_store_gp_to_frame(buf, machine_gp(0), slot);
            }
        } else {
            let slot =
                frame.slot_access(frame.block_param_slots(target_idx)[param_idx]);
            if is_fp {
                B::emit_store_fp_to_frame(buf, machine_fp(0), slot);
            } else {
                B::emit_store_gp_to_frame(buf, machine_gp(0), slot);
            }
        }
    }

    type SavedState = GreedySavedState;

    fn save_state(&self) -> GreedySavedState {
        GreedySavedState {
            values: self.values.iter().map(|v| (v.loc, v.spill_slot)).collect(),
            gp_occupant: self.gp_occupant,
            fp_occupant: self.fp_occupant,
        }
    }

    fn restore_state(&mut self, state: GreedySavedState) {
        for (i, (loc, spill)) in state.values.into_iter().enumerate() {
            self.values[i].loc = loc;
            self.values[i].spill_slot = spill;
        }
        self.gp_occupant = state.gp_occupant;
        self.fp_occupant = state.fp_occupant;
    }

    fn gp_occupant(&self, r: usize) -> Option<Value> {
        self.gp_occupant[r]
    }

    fn fp_occupant(&self, r: usize) -> Option<Value> {
        self.fp_occupant[r]
    }

    fn num_values(&self) -> usize {
        self.values.len()
    }

    fn value_info_snapshot(&self) -> Vec<(Option<i32>, Type)> {
        self.values.iter().map(|v| (v.spill_slot, v.ty)).collect()
    }
}
