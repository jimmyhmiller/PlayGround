use std::marker::PhantomData;

use dynexec::{
    DefaultExecutionConfig, ExecutionConfig, LayoutConfigDefaults, ValueLayout,
    validate_execution_config,
};
use dynasm::arm64::inst::*;
use dynasm::arm64::reg::*;
use dynasm::arm64::reloc::*;
use dynasm::buffer::{CodeBuffer, Label};
use dynasm::code_memory::{CodeMemory, PagedCodeMemory};
use dynir::ir::*;
use dynir::types::Type;

#[cfg(test)]
mod tests;

// ─── Public API ────────────────────────────────────────────────────

pub type DefaultJitConfig<L> = DefaultExecutionConfig<L>;

pub struct JitFunction {
    memory: PagedCodeMemory,
}

impl JitFunction {
    pub fn compile<L: LayoutConfigDefaults>(func: &Function, externs: &[*const u8]) -> Self {
        Self::compile_with_config::<DefaultJitConfig<L>>(func, externs)
    }

    pub fn compile_with_gc<L: LayoutConfigDefaults>(
        func: &Function,
        externs: &[*const u8],
        handler: extern "C" fn(*mut u8, usize),
    ) -> Self {
        Self::compile_with_config_and_gc::<DefaultJitConfig<L>>(func, externs, Some(handler as u64))
    }

    pub fn compile_with_config<Cfg: ExecutionConfig>(func: &Function, externs: &[*const u8]) -> Self {
        Self::compile_with_config_and_gc::<Cfg>(func, externs, None)
    }

    pub fn compile_with_config_and_gc<Cfg: ExecutionConfig>(
        func: &Function,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self {
        validate_execution_config::<Cfg>().unwrap_or_else(|err| {
            panic!("invalid dynlower config: {err}");
        });

        let mut lowerer =
            Lowerer::<Cfg::Layout>::new_inner(func, externs, None, safepoint_handler);
        lowerer.run();
        let code = lowerer.buf.into_code();

        let mut memory = PagedCodeMemory::new();
        memory.push(&code);
        memory.finalize();

        JitFunction { memory }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.memory.base_ptr()
    }
}

// ─── JitModule ─────────────────────────────────────────────────────

/// JIT-compiled module: multiple functions that can call each other.
///
/// Internal calls go through an indirect call table so all function
/// pointers are resolved after compilation.
pub struct JitModule {
    memory: PagedCodeMemory,
    /// One entry per `Module::func_table` slot. Extern entries hold the
    /// provided extern pointers; internal entries are filled in after
    /// compilation with pointers into `memory`.
    call_table: Vec<*const u8>,
}

impl JitModule {
    /// Compile a module with no GC safepoint handler.
    pub fn compile<L: LayoutConfigDefaults>(module: &Module, externs: &[*const u8]) -> Self {
        Self::compile_with_config::<DefaultJitConfig<L>>(module, externs)
    }

    /// Compile a module with a GC safepoint handler.
    ///
    /// At each `Inst::Safepoint`, the JIT will spill all live values and
    /// call `handler(frame_ptr, frame_size)`. The handler can scan the
    /// frame for GC pointers using `PtrPolicy::try_decode_ptr`.
    pub fn compile_with_gc<L: LayoutConfigDefaults>(
        module: &Module,
        externs: &[*const u8],
        handler: extern "C" fn(*mut u8, usize),
    ) -> Self {
        Self::compile_with_config_and_gc::<DefaultJitConfig<L>>(
            module,
            externs,
            Some(handler as u64),
        )
    }

    pub fn compile_with_config<Cfg: ExecutionConfig>(module: &Module, externs: &[*const u8]) -> Self {
        Self::compile_with_config_and_gc::<Cfg>(module, externs, None)
    }

    pub fn compile_with_config_and_gc<Cfg: ExecutionConfig>(
        module: &Module,
        externs: &[*const u8],
        safepoint_handler: Option<u64>,
    ) -> Self {
        validate_execution_config::<Cfg>().unwrap_or_else(|err| {
            panic!("invalid dynlower config: {err}");
        });

        // 1. Build call table: fill externs, leave internals as null
        let mut call_table: Vec<*const u8> = Vec::with_capacity(module.func_table.len());
        let mut extern_idx = 0usize;
        for def in &module.func_table {
            match def {
                FuncDef::Extern(_) => {
                    call_table.push(externs[extern_idx]);
                    extern_idx += 1;
                }
                FuncDef::Internal(_) => {
                    call_table.push(std::ptr::null());
                }
            }
        }

        // The Vec's heap pointer is stable across pushes (we pre-allocated)
        let call_table_base = call_table.as_ptr() as u64;

        // 2. Compile each internal function
        let mut memory = PagedCodeMemory::new();
        let mut entry_offsets: Vec<usize> = Vec::new();

        for func in &module.functions {
            let mut lowerer =
                Lowerer::<Cfg::Layout>::new_module(func, call_table_base, safepoint_handler);
            lowerer.run();
            let code = lowerer.buf.into_code();
            let offset = memory.push(&code);
            entry_offsets.push(offset);
        }

        memory.finalize();

        // 3. Patch internal function pointers into the call table
        let base = memory.base_ptr();
        for (ft_idx, def) in module.func_table.iter().enumerate() {
            if let FuncDef::Internal(func_idx) = def {
                let ptr = unsafe { base.add(entry_offsets[*func_idx]) };
                call_table[ft_idx] = ptr;
            }
        }

        JitModule { memory, call_table }
    }

    /// Call a function in the module by its `FuncRef`.
    pub fn call(&self, func_ref: FuncRef, args: &[u64]) -> u64 {
        let ptr = self.call_table[func_ref.index()];
        assert!(!ptr.is_null(), "call to unresolved function");
        unsafe { call_jit(ptr, args) }
    }
}

/// Call a JIT-compiled function with arbitrary 64-bit argument slots.
///
/// The JIT's entry convention is:
/// - slots 0..15 arrive in X0..X15
/// - remaining slots are passed on the stack at the incoming SP
///
/// This matches the lowering path used for JIT-to-JIT calls.
#[cfg(target_arch = "aarch64")]
pub unsafe fn call_jit(ptr: *const u8, args: &[u64]) -> u64 {
    let padded_len = args.len().max(16);
    let mut padded = vec![0u64; padded_len];
    padded[..args.len()].copy_from_slice(args);
    let overflow = args.len().saturating_sub(16);
    let overflow_bytes = align_up(overflow * 8, 16);
    let overflow_count = overflow;
    let overflow_src = unsafe { padded.as_ptr().add(16) };
    let result: u64;
    unsafe {
        core::arch::asm!(
            "sub sp, sp, x20",
            "mov x9, sp",
            "cbz x21, 2f",
            "1:",
            "ldr x10, [x22], #8",
            "str x10, [x9], #8",
            "sub x21, x21, #1",
            "cbnz x21, 1b",
            "2:",
            "ldp x0, x1, [x17]",
            "ldp x2, x3, [x17, #16]",
            "ldp x4, x5, [x17, #32]",
            "ldp x6, x7, [x17, #48]",
            "ldp x8, x9, [x17, #64]",
            "ldp x10, x11, [x17, #80]",
            "ldp x12, x13, [x17, #96]",
            "ldp x14, x15, [x17, #112]",
            "blr x16",
            "add sp, sp, x20",
            inlateout("x21") overflow_count => _,
            inlateout("x22") overflow_src => _,
            in("x20") overflow_bytes,
            in("x16") ptr,
            in("x17") padded.as_ptr(),
            lateout("x0") result,
            lateout("x9") _,
            lateout("x10") _,
            clobber_abi("C"),
        );
    }
    result
}

// ─── Value Location ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueLoc {
    GpReg(u8),
    FpReg(u8),
    Spill(i32), // offset from FP (negative)
    Unassigned,
}

#[derive(Debug, Clone)]
struct ValueInfo {
    loc: ValueLoc,
    spill_slot: Option<i32>, // if spilled, the offset from FP
    remaining_uses: u32,
    ty: Type,
}

// ─── Frame Layout State ────────────────────────────────────────────

struct FrameLayoutState {
    next_local_offset: i32,
    max_outgoing_arg_bytes: i32,
}

impl FrameLayoutState {
    fn new() -> Self {
        FrameLayoutState {
            next_local_offset: 16,
            max_outgoing_arg_bytes: 0,
        }
    }

    fn alloc_local_slot(&mut self) -> i32 {
        let offset = self.next_local_offset;
        self.next_local_offset += 8;
        offset
    }

    fn reserve_outgoing_arg_bytes(&mut self, bytes: i32) {
        self.max_outgoing_arg_bytes = self.max_outgoing_arg_bytes.max(bytes);
    }

    fn local_frame_size(&self) -> i32 {
        self.next_local_offset
    }

    fn total_frame_size(&self) -> i32 {
        align_up((self.next_local_offset + self.max_outgoing_arg_bytes) as usize, 16) as i32
    }
}

// ─── Register State ────────────────────────────────────────────────

const NUM_GP: usize = 28; // X0-X27
const NUM_FP: usize = 32; // D0-D31

// Allocatable GP regs: caller-saved only.
// We do not allocate callee-saved regs until the backend preserves them.
const ALLOCATABLE_GP: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// Allocatable FP regs: caller-saved only.
const ALLOCATABLE_FP: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

// Caller-saved GP: X0-X15 (X16-X18 are special)
const CALLER_SAVED_GP: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

// Caller-saved FP: D0-D7, D16-D31
const CALLER_SAVED_FP: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

struct RegState {
    gp_occupant: [Option<Value>; NUM_GP],
    fp_occupant: [Option<Value>; NUM_FP],
    values: Vec<ValueInfo>,
    frame: FrameLayoutState,
    gp_evict_cursor: usize,
    fp_evict_cursor: usize,
}

impl RegState {
    fn new(num_values: usize) -> Self {
        RegState {
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
            frame: FrameLayoutState::new(),
            gp_evict_cursor: 0,
            fp_evict_cursor: 0,
        }
    }

    fn is_float_type(ty: Type) -> bool {
        ty == Type::F64
    }

    fn alloc_gp(&mut self, buf: &mut CodeBuffer<Arm64>) -> u8 {
        // Try to find a free allocatable GP reg
        for &r in ALLOCATABLE_GP {
            if self.gp_occupant[r as usize].is_none() {
                return r;
            }
        }
        // Evict: round-robin through allocatable regs
        let start = self.gp_evict_cursor;
        for i in 0..ALLOCATABLE_GP.len() {
            let idx = (start + i) % ALLOCATABLE_GP.len();
            let r = ALLOCATABLE_GP[idx];
            if let Some(val) = self.gp_occupant[r as usize] {
                self.spill_gp_reg(buf, r, val);
                self.gp_evict_cursor = (idx + 1) % ALLOCATABLE_GP.len();
                return r;
            }
        }
        panic!("no GP register available");
    }

    fn alloc_fp(&mut self, buf: &mut CodeBuffer<Arm64>) -> u8 {
        for &r in ALLOCATABLE_FP {
            if self.fp_occupant[r as usize].is_none() {
                return r;
            }
        }
        let start = self.fp_evict_cursor;
        for i in 0..ALLOCATABLE_FP.len() {
            let idx = (start + i) % ALLOCATABLE_FP.len();
            let r = ALLOCATABLE_FP[idx];
            if let Some(val) = self.fp_occupant[r as usize] {
                self.spill_fp_reg(buf, r, val);
                self.fp_evict_cursor = (idx + 1) % ALLOCATABLE_FP.len();
                return r;
            }
        }
        panic!("no FP register available");
    }

    fn alloc_spill_slot(&mut self) -> i32 {
        self.frame.alloc_local_slot()
    }

    fn spill_gp_reg(&mut self, buf: &mut CodeBuffer<Arm64>, r: u8, val: Value) {
        if self.values[val.index()].remaining_uses == 0 {
            self.gp_occupant[r as usize] = None;
            self.values[val.index()].loc = ValueLoc::Unassigned;
            return;
        }
        let offset = match self.values[val.index()].spill_slot {
            Some(off) => off,
            None => {
                let off = self.frame.alloc_local_slot();
                self.values[val.index()].spill_slot = Some(off);
                off
            }
        };
        let reg = gp_reg(r, RegSize::X64);
        emit_store_to_fp(buf, reg, offset);
        self.values[val.index()].loc = ValueLoc::Spill(offset);
        self.gp_occupant[r as usize] = None;
    }

    fn spill_fp_reg(&mut self, buf: &mut CodeBuffer<Arm64>, r: u8, val: Value) {
        if self.values[val.index()].remaining_uses == 0 {
            self.fp_occupant[r as usize] = None;
            self.values[val.index()].loc = ValueLoc::Unassigned;
            return;
        }
        let offset = match self.values[val.index()].spill_slot {
            Some(off) => off,
            None => {
                let off = self.frame.alloc_local_slot();
                self.values[val.index()].spill_slot = Some(off);
                off
            }
        };
        let dreg = Arm64Reg::new(r, RegSize::X64);
        emit_fp_store_to_fp(buf, dreg, offset);
        self.values[val.index()].loc = ValueLoc::Spill(offset);
        self.fp_occupant[r as usize] = None;
    }

    fn assign_gp(&mut self, val: Value, r: u8) {
        self.gp_occupant[r as usize] = Some(val);
        self.values[val.index()].loc = ValueLoc::GpReg(r);
    }

    fn assign_fp(&mut self, val: Value, r: u8) {
        self.fp_occupant[r as usize] = Some(val);
        self.values[val.index()].loc = ValueLoc::FpReg(r);
    }

    /// Ensure a value is in a GP register, returning the reg index.
    fn ensure_in_gp_reg(&mut self, buf: &mut CodeBuffer<Arm64>, val: Value) -> u8 {
        match self.values[val.index()].loc {
            ValueLoc::GpReg(r) => r,
            ValueLoc::Spill(offset) => {
                let r = self.alloc_gp(buf);
                let reg = gp_reg(r, RegSize::X64);
                emit_load_from_fp(buf, reg, offset);
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
    fn ensure_in_fp_reg(&mut self, buf: &mut CodeBuffer<Arm64>, val: Value) -> u8 {
        match self.values[val.index()].loc {
            ValueLoc::FpReg(r) => r,
            ValueLoc::Spill(offset) => {
                let r = self.alloc_fp(buf);
                let dreg = Arm64Reg::new(r, RegSize::X64);
                emit_fp_load_from_fp(buf, dreg, offset);
                self.assign_fp(val, r);
                r
            }
            ValueLoc::GpReg(_) => panic!("expected FP value, got GP"),
            ValueLoc::Unassigned => panic!("value v{} is unassigned", val.index()),
        }
    }

    /// Decrement use count and free register if dead.
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

    /// Spill all caller-saved registers that have live values.
    fn spill_caller_saved(&mut self, buf: &mut CodeBuffer<Arm64>) {
        for &r in CALLER_SAVED_GP {
            if let Some(val) = self.gp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_gp_reg(buf, r, val);
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
        for &r in CALLER_SAVED_FP {
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_fp_reg(buf, r, val);
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
    fn spill_all_live(&mut self, buf: &mut CodeBuffer<Arm64>) {
        for r in 0..NUM_GP as u8 {
            if let Some(val) = self.gp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_gp_reg(buf, r, val);
                } else {
                    self.gp_occupant[r as usize] = None;
                }
            }
        }
        for r in 0..NUM_FP as u8 {
            if let Some(val) = self.fp_occupant[r as usize] {
                if self.values[val.index()].remaining_uses > 0 {
                    self.spill_fp_reg(buf, r, val);
                } else {
                    self.fp_occupant[r as usize] = None;
                }
            }
        }
    }

    /// Free all register mappings. Values in registers revert to their spill
    /// slot if one exists, otherwise become Unassigned.
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
}

// ─── Block metadata ────────────────────────────────────────────────

struct BlockMeta {
    label: Label,
    /// Canonical spill slot offsets for each block param.
    param_spill_slots: Vec<i32>,
}

// ─── Lowerer ───────────────────────────────────────────────────────

struct Lowerer<'a, S: ValueLayout> {
    func: &'a Function,
    externs: &'a [*const u8],
    /// When Some, all calls go through an indirect call table at this
    /// address. `call_table[func_ref.index()]` holds the function pointer.
    call_table_base: Option<u64>,
    /// When Some, safepoints emit a call to this handler function.
    /// Signature: `extern "C" fn(frame_ptr: *mut u8, frame_size: usize)`.
    safepoint_handler: Option<u64>,
    buf: CodeBuffer<Arm64>,
    regs: RegState,
    block_meta: Vec<BlockMeta>,
    prologue_stp_offset: usize,
    epilogue_ldp_offsets: Vec<usize>, // offsets of LDP instructions to patch
    _scheme: PhantomData<S>,
}

impl<'a, S: ValueLayout> Lowerer<'a, S> {
    fn new(func: &'a Function, externs: &'a [*const u8]) -> Self {
        Self::new_inner(func, externs, None, None)
    }

    fn new_module(
        func: &'a Function,
        call_table_base: u64,
        safepoint_handler: Option<u64>,
    ) -> Self {
        Self::new_inner(func, &[], Some(call_table_base), safepoint_handler)
    }

    fn new_inner(
        func: &'a Function,
        externs: &'a [*const u8],
        call_table_base: Option<u64>,
        safepoint_handler: Option<u64>,
    ) -> Self {
        let num_values = func.value_types.len();
        let mut buf = CodeBuffer::<Arm64>::new();

        // Create labels for all blocks
        let mut block_meta = Vec::new();
        for (_i, _block) in func.blocks.iter().enumerate() {
            let label = buf.create_label();
            block_meta.push(BlockMeta {
                label,
                param_spill_slots: Vec::new(),
            });
        }

        let mut regs = RegState::new(num_values);

        // Liveness pass: count uses
        for block in &func.blocks {
            for inst_node in &block.insts {
                inst_node.inst.for_each_value(|v| {
                    regs.values[v.index()].remaining_uses += 1;
                });
            }
            block.terminator.for_each_value(|v| {
                regs.values[v.index()].remaining_uses += 1;
            });
        }

        // Set types for all values
        for (i, ty) in func.value_types.iter().enumerate() {
            regs.values[i].ty = *ty;
        }

        // Allocate canonical spill slots for ALL blocks with params.
        // This simplifies block transitions: every block param has a
        // canonical spill location, regardless of predecessor count.
        for (i, block) in func.blocks.iter().enumerate() {
            if i > 0 && !block.params.is_empty() {
                for _ in &block.params {
                    let offset = regs.alloc_spill_slot();
                    block_meta[i].param_spill_slots.push(offset);
                }
            }
        }

        Lowerer {
            func,
            externs,
            call_table_base,
            safepoint_handler,
            buf,
            regs,
            block_meta,
            prologue_stp_offset: 0,
            epilogue_ldp_offsets: Vec::new(),
            _scheme: PhantomData,
        }
    }

    fn run(&mut self) {
        self.emit_prologue();

        for block_idx in 0..self.func.blocks.len() {
            self.lower_block(block_idx);
        }

        self.patch_prologue();
    }

    fn emit_prologue(&mut self) {
        // Prologue: MOV X28, SP   ; incoming stack-arg base
        //           SUB SP, SP, #frame_size (placeholder)
        //           STP X29, X30, [SP]
        //           MOV X29, SP
        // The SUB will be patched after codegen with the actual frame size.
        self.buf.emit(Arm64Inst::mov(X28, SP));
        self.prologue_stp_offset = self.buf.emit(Arm64Inst::sub_imm(SP, SP, 16)); // placeholder
        self.buf
            .emit(Arm64Inst::stp(X29, X30, SP, 0, StpMode::SignedOffset));
        self.buf.emit(Arm64Inst::mov(X29, SP));
    }

    fn emit_epilogue(&mut self) {
        // MOV SP, X29
        self.buf.emit(Arm64Inst::mov(SP, X29));
        // LDP X29, X30, [SP]
        self.buf
            .emit(Arm64Inst::ldp(X29, X30, SP, 0, LdpMode::SignedOffset));
        // ADD SP, SP, #frame_size — placeholder, patched later
        let add_offset = self.buf.emit(Arm64Inst::add_imm(SP, SP, 16)); // placeholder
        self.epilogue_ldp_offsets.push(add_offset);
        self.buf.emit(Arm64Inst::ret());
    }

    fn patch_prologue(&mut self) {
        let total_frame = self.regs.frame.total_frame_size().max(16);

        // Patch the SUB SP at prologue
        let sub_inst = Arm64Inst::sub_imm(SP, SP, total_frame);
        self.patch_word(self.prologue_stp_offset, &sub_inst.encode().to_le_bytes());

        // Patch all tracked ADD SP epilogues
        let add_inst = Arm64Inst::add_imm(SP, SP, total_frame);
        let add_bytes = add_inst.encode().to_le_bytes();
        for &offset in &self.epilogue_ldp_offsets.clone() {
            self.patch_word(offset, &add_bytes);
        }
    }

    fn patch_word(&mut self, offset: usize, bytes: &[u8]) {
        self.buf.patch_bytes(offset, bytes);
    }

    fn lower_block(&mut self, block_idx: usize) {
        let block = &self.func.blocks[block_idx];

        // Bind label
        self.buf.bind_label(self.block_meta[block_idx].label);

        // Clear register state at block entry (conservative approach)
        if block_idx > 0 {
            self.regs.clear_regs();
        }

        // Handle block params
        if block_idx == 0 {
            // Entry block: params come from 64-bit argument slots.
            // Slots 0..15 are in X0..X15; overflow slots arrive on the stack
            // at the incoming SP captured in X28 by the prologue.
            let mut gp_arg = 0u8;
            for &(val, ty) in &block.params {
                if gp_arg < 16 {
                    if RegState::is_float_type(ty) {
                        let gp = gp_reg(gp_arg, RegSize::X64);
                        let fp_idx = self.regs.alloc_fp(&mut self.buf);
                        let fp = Arm64Reg::new(fp_idx, RegSize::X64);
                        self.buf.emit(Arm64Inst::fmov_gp_to_fp(fp, gp));
                        self.regs.assign_fp(val, fp_idx);
                    } else {
                        self.regs.assign_gp(val, gp_arg);
                    }
                } else {
                    // Beyond register window: copy from the caller's stack
                    // arg area into a normal local spill slot.
                    let slot = self.regs.alloc_spill_slot();
                    self.regs.values[val.index()].spill_slot = Some(slot);
                    self.regs.values[val.index()].loc = ValueLoc::Spill(slot);
                    let incoming_offset = ((gp_arg - 16) as i32) * 8;
                    if RegState::is_float_type(ty) {
                        self.buf.emit(Arm64Inst::ldr(X27, X28, incoming_offset));
                        let dtmp = Arm64Reg::new(31, RegSize::X64);
                        self.buf.emit(Arm64Inst::fmov_gp_to_fp(dtmp, X27));
                        emit_fp_store_to_fp(&mut self.buf, dtmp, slot);
                    } else {
                        self.buf.emit(Arm64Inst::ldr(X27, X28, incoming_offset));
                        emit_store_to_fp(&mut self.buf, X27, slot);
                    }
                }
                gp_arg += 1;
            }
        } else if !self.block_meta[block_idx].param_spill_slots.is_empty() {
            // Block params are in canonical spill slots
            for (i, &(val, _ty)) in block.params.iter().enumerate() {
                let offset = self.block_meta[block_idx].param_spill_slots[i];
                self.regs.values[val.index()].loc = ValueLoc::Spill(offset);
            }
        }

        // Lower instructions
        for inst_node in &block.insts {
            self.lower_inst(inst_node);
        }

        // Lower terminator
        self.lower_terminator(block_idx);
    }

    fn lower_inst(&mut self, inst_node: &InstNode) {
        let result_val = inst_node.value;

        match &inst_node.inst {
            Inst::Iconst(ty, imm) => {
                let val = result_val.unwrap();
                let r = self.regs.alloc_gp(&mut self.buf);
                // Always use X register for mov_imm64 (W32 MOVZ can't encode all shifts)
                let reg = gp_reg(r, RegSize::X64);
                let imm_val = match ty {
                    Type::I8 => (*imm as u8) as u64,
                    Type::I32 => (*imm as u32) as u64,
                    _ => *imm as u64,
                };
                self.emit_mov_imm(reg, imm_val);
                self.regs.assign_gp(val, r);
            }

            Inst::F64Const(f) => {
                let val = result_val.unwrap();
                let bits = f.to_bits();
                // Load bits into scratch GP reg, then FMOV to FP reg
                let scratch = X28;
                self.emit_mov_imm(scratch, bits);
                let fr = self.regs.alloc_fp(&mut self.buf);
                let dreg = Arm64Reg::new(fr, RegSize::X64);
                self.buf.emit(Arm64Inst::fmov_gp_to_fp(dreg, scratch));
                self.regs.assign_fp(val, fr);
            }

            Inst::Add(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Add),
            Inst::Sub(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Sub),
            Inst::Mul(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Mul),
            Inst::SDiv(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::SDiv),
            Inst::UDiv(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::UDiv),
            Inst::And(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::And),
            Inst::Or(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Or),
            Inst::Xor(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Xor),
            Inst::Shl(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::Shl),
            Inst::LShr(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::LShr),
            Inst::AShr(a, b) => self.lower_gp_binop(result_val.unwrap(), *a, *b, BinOp::AShr),

            Inst::Neg(a) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                let ty = self.regs.values[a.index()].ty;
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let size = type_to_regsize(ty);
                let rd = gp_reg(rd_idx, size);
                let rn = gp_reg(ra, size);
                let zr = if size == RegSize::W32 { WZR } else { XZR };
                self.buf.emit(Arm64Inst::sub(rd, zr, rn));
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Not(a) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                let ty = self.regs.values[a.index()].ty;
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, type_to_regsize(ty));
                let rm = gp_reg(ra, type_to_regsize(ty));
                self.buf.emit(Arm64Inst::mvn(rd, rm));
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::FAdd(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Add),
            Inst::FSub(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Sub),
            Inst::FMul(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Mul),
            Inst::FDiv(a, b) => self.lower_fp_binop(result_val.unwrap(), *a, *b, FpBinOp::Div),

            Inst::FNeg(a) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_fp_reg(&mut self.buf, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_fp(&mut self.buf);
                let rd = Arm64Reg::new(rd_idx, RegSize::X64);
                let rn = Arm64Reg::new(ra, RegSize::X64);
                self.buf.emit(Arm64Inst::fneg(rd, rn));
                self.regs.assign_fp(val, rd_idx);
            }

            Inst::Icmp(op, a, b) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                let rb = self.regs.ensure_in_gp_reg(&mut self.buf, *b);
                let ty = self.regs.values[a.index()].ty;
                self.regs.dec_use(*a);
                self.regs.dec_use(*b);
                let size = type_to_regsize(ty);
                let rn = gp_reg(ra, size);
                let rm = gp_reg(rb, size);
                self.buf.emit(Arm64Inst::cmp(rn, rm));
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, RegSize::W32);
                let cond = cmpop_to_cond(*op);
                self.buf.emit(Arm64Inst::cset(rd, cond));
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Fcmp(op, a, b) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_fp_reg(&mut self.buf, *a);
                let rb = self.regs.ensure_in_fp_reg(&mut self.buf, *b);
                self.regs.dec_use(*a);
                self.regs.dec_use(*b);
                let rn = Arm64Reg::new(ra, RegSize::X64);
                let rm = Arm64Reg::new(rb, RegSize::X64);
                self.buf.emit(Arm64Inst::fcmp_double(rn, rm));
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, RegSize::W32);
                let cond = cmpop_to_cond(*op);
                self.buf.emit(Arm64Inst::cset(rd, cond));
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Select(cond, t, f) => {
                let val = result_val.unwrap();
                let rc = self.regs.ensure_in_gp_reg(&mut self.buf, *cond);
                let ty = self.regs.values[t.index()].ty;

                if RegState::is_float_type(ty) {
                    let rt = self.regs.ensure_in_fp_reg(&mut self.buf, *t);
                    let rf = self.regs.ensure_in_fp_reg(&mut self.buf, *f);
                    let rc_reg = gp_reg(rc, RegSize::W32);
                    self.buf.emit(Arm64Inst::cmp_imm(rc_reg, 0));
                    self.regs.dec_use(*cond);
                    self.regs.dec_use(*t);
                    self.regs.dec_use(*f);
                    let rd_idx = self.regs.alloc_fp(&mut self.buf);
                    let rd = Arm64Reg::new(rd_idx, RegSize::X64);
                    let rn = Arm64Reg::new(rt, RegSize::X64);
                    let rm = Arm64Reg::new(rf, RegSize::X64);
                    self.buf.emit(Arm64Inst::fcsel(rd, rn, rm, Arm64Cond::NE));
                    self.regs.assign_fp(val, rd_idx);
                } else {
                    let rt = self.regs.ensure_in_gp_reg(&mut self.buf, *t);
                    let rf = self.regs.ensure_in_gp_reg(&mut self.buf, *f);
                    let size = type_to_regsize(ty);
                    let rc_reg = gp_reg(rc, RegSize::W32);
                    self.buf.emit(Arm64Inst::cmp_imm(rc_reg, 0));
                    self.regs.dec_use(*cond);
                    self.regs.dec_use(*t);
                    self.regs.dec_use(*f);
                    let rd_idx = self.regs.alloc_gp(&mut self.buf);
                    let rd = gp_reg(rd_idx, size);
                    let rn = gp_reg(rt, size);
                    let rm = gp_reg(rf, size);
                    self.buf.emit(Arm64Inst::csel(rd, rn, rm, Arm64Cond::NE));
                    self.regs.assign_gp(val, rd_idx);
                }
            }

            Inst::Sext(a, target_ty) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                let src_ty = self.regs.values[a.index()].ty;
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, type_to_regsize(*target_ty));
                let rn = gp_reg(ra, RegSize::W32);
                match (src_ty, target_ty) {
                    (Type::I32, Type::I64) => {
                        self.buf.emit(Arm64Inst::sxtw(rd, rn));
                    }
                    (Type::I8, Type::I32) | (Type::I8, Type::I64) => {
                        self.buf.emit(Arm64Inst::sxtb(rd, rn));
                    }
                    _ => {
                        // General case: just mov
                        self.buf.emit(Arm64Inst::mov(rd, rn));
                    }
                }
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Zext(a, target_ty) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                let src_ty = self.regs.values[a.index()].ty;
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                match (src_ty, target_ty) {
                    (Type::I8, _) => {
                        // AND Wd, Wn, #0xFF
                        let rd = gp_reg(rd_idx, RegSize::W32);
                        let rn = gp_reg(ra, RegSize::W32);
                        self.buf.emit(Arm64Inst::AndImm {
                            sf: 0,
                            n: 0,
                            immr: 0,
                            imms: 7,
                            rn,
                            rd,
                        });
                    }
                    (Type::I32, Type::I64) => {
                        // MOV Wd, Wn (implicitly zero-extends to 64 bits)
                        let rd = gp_reg(rd_idx, RegSize::W32);
                        let rn = gp_reg(ra, RegSize::W32);
                        self.buf.emit(Arm64Inst::mov(rd, rn));
                    }
                    _ => {
                        let rd = gp_reg(rd_idx, type_to_regsize(*target_ty));
                        let rn = gp_reg(ra, type_to_regsize(src_ty));
                        self.buf.emit(Arm64Inst::mov(rd, rn));
                    }
                }
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Trunc(a, target_ty) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                match target_ty {
                    Type::I8 => {
                        let rd = gp_reg(rd_idx, RegSize::W32);
                        let rn = gp_reg(ra, RegSize::W32);
                        self.buf.emit(Arm64Inst::AndImm {
                            sf: 0,
                            n: 0,
                            immr: 0,
                            imms: 7,
                            rn,
                            rd,
                        });
                    }
                    Type::I32 => {
                        // MOV Wd, Wn (truncates by using 32-bit reg)
                        let rd = gp_reg(rd_idx, RegSize::W32);
                        let rn = gp_reg(ra, RegSize::W32);
                        self.buf.emit(Arm64Inst::mov(rd, rn));
                    }
                    _ => {
                        let rd = gp_reg(rd_idx, type_to_regsize(*target_ty));
                        let rn = gp_reg(ra, type_to_regsize(*target_ty));
                        self.buf.emit(Arm64Inst::mov(rd, rn));
                    }
                }
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::IntToFloat(a) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                let src_ty = self.regs.values[a.index()].ty;
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_fp(&mut self.buf);
                let rd = Arm64Reg::new(rd_idx, RegSize::X64);
                let rn = gp_reg(ra, type_to_regsize(src_ty));
                self.buf.emit(Arm64Inst::scvtf_to_double(rd, rn));
                self.regs.assign_fp(val, rd_idx);
            }

            Inst::FloatToInt(a) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_fp_reg(&mut self.buf, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, RegSize::X64);
                let rn = Arm64Reg::new(ra, RegSize::X64);
                self.buf.emit(Arm64Inst::fcvtzs_from_double(rd, rn));
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Bitcast(a, _target_ty) => {
                let val = result_val.unwrap();
                let src_ty = self.regs.values[a.index()].ty;
                let dst_ty = self.regs.values[val.index()].ty;

                if RegState::is_float_type(src_ty) && !RegState::is_float_type(dst_ty) {
                    // FP -> GP: FMOV Xd, Dn
                    let ra = self.regs.ensure_in_fp_reg(&mut self.buf, *a);
                    self.regs.dec_use(*a);
                    let rd_idx = self.regs.alloc_gp(&mut self.buf);
                    let rd = gp_reg(rd_idx, RegSize::X64);
                    let rn = Arm64Reg::new(ra, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_fp_to_gp(rd, rn));
                    self.regs.assign_gp(val, rd_idx);
                } else if !RegState::is_float_type(src_ty) && RegState::is_float_type(dst_ty) {
                    // GP -> FP: FMOV Dd, Xn
                    let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                    self.regs.dec_use(*a);
                    let rd_idx = self.regs.alloc_fp(&mut self.buf);
                    let rd = Arm64Reg::new(rd_idx, RegSize::X64);
                    let rn = gp_reg(ra, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_gp_to_fp(rd, rn));
                    self.regs.assign_fp(val, rd_idx);
                } else {
                    // Same class: just rename
                    if RegState::is_float_type(src_ty) {
                        let ra = self.regs.ensure_in_fp_reg(&mut self.buf, *a);
                        self.regs.dec_use(*a);
                        self.regs.assign_fp(val, ra);
                    } else {
                        let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                        self.regs.dec_use(*a);
                        self.regs.assign_gp(val, ra);
                    }
                }
            }

            Inst::Load(ty, addr, offset) => {
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *addr);
                self.regs.dec_use(*addr);
                let base = gp_reg(ra, RegSize::X64);

                if RegState::is_float_type(*ty) {
                    let rd_idx = self.regs.alloc_fp(&mut self.buf);
                    let rd = Arm64Reg::new(rd_idx, RegSize::X64);
                    if *offset != 0 {
                        // Add offset to base in scratch
                        self.emit_mov_imm(X28, *offset as u64);
                        self.buf.emit(Arm64Inst::add(X28, base, X28));
                        self.buf.emit(Arm64Inst::ldur_fp(rd, X28, 0));
                    } else {
                        self.buf.emit(Arm64Inst::ldur_fp(rd, base, 0));
                    }
                    self.regs.assign_fp(val, rd_idx);
                } else {
                    let rd_idx = self.regs.alloc_gp(&mut self.buf);
                    let size = type_to_regsize(*ty);
                    let rd = gp_reg(rd_idx, size);
                    if *offset >= -256 && *offset <= 255 {
                        self.buf.emit(Arm64Inst::ldur(rd, base, *offset));
                    } else {
                        self.emit_mov_imm(X28, *offset as u64);
                        self.buf.emit(Arm64Inst::add(X28, base, X28));
                        self.buf.emit(Arm64Inst::ldur(rd, X28, 0));
                    }
                    self.regs.assign_gp(val, rd_idx);
                }
            }

            Inst::Store(val_to_store, addr, offset) => {
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *addr);
                let base = gp_reg(ra, RegSize::X64);
                let val_ty = self.regs.values[val_to_store.index()].ty;

                if RegState::is_float_type(val_ty) {
                    let rv = self.regs.ensure_in_fp_reg(&mut self.buf, *val_to_store);
                    let src = Arm64Reg::new(rv, RegSize::X64);
                    if *offset != 0 {
                        self.emit_mov_imm(X28, *offset as u64);
                        self.buf.emit(Arm64Inst::add(X28, base, X28));
                        self.buf.emit(Arm64Inst::stur_fp(src, X28, 0));
                    } else {
                        self.buf.emit(Arm64Inst::stur_fp(src, base, 0));
                    }
                } else {
                    let rv = self.regs.ensure_in_gp_reg(&mut self.buf, *val_to_store);
                    let size = type_to_regsize(val_ty);
                    let src = gp_reg(rv, size);
                    if *offset >= -256 && *offset <= 255 {
                        self.buf.emit(Arm64Inst::stur(src, base, *offset));
                    } else {
                        self.emit_mov_imm(X28, *offset as u64);
                        self.buf.emit(Arm64Inst::add(X28, base, X28));
                        self.buf.emit(Arm64Inst::stur(src, X28, 0));
                    }
                }
                self.regs.dec_use(*val_to_store);
                self.regs.dec_use(*addr);
            }

            Inst::Call(func_ref, args) => {
                self.lower_call(*func_ref, args, result_val);
            }

            Inst::CallIndirect(callee, args, _ret_ty) => {
                self.lower_call_indirect(*callee, args, result_val);
            }

            // ── Tagged value operations (TagScheme-generic) ─────────
            //
            // Uses S: TagScheme constants to emit the correct bit
            // manipulation for any tagging scheme (NanBox, LowBit, etc).
            Inst::Payload(a) => {
                // S::extract_payload(bits)
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, RegSize::X64);
                let rn = gp_reg(ra, RegSize::X64);
                if S::HAS_UNBOXED_FLOAT {
                    // NanBox-style: payload is in low PAYLOAD_BITS → AND with mask
                    let mask = (1u64 << S::PAYLOAD_BITS) - 1;
                    self.emit_mov_imm(X28, mask);
                    self.buf.emit(Arm64Inst::and(rd, rn, X28));
                } else {
                    // LowBit-style: payload is in upper bits → LSR by tag_bits
                    let tag_bits = 64 - S::PAYLOAD_BITS;
                    self.emit_mov_imm(X28, tag_bits as u64);
                    self.buf.emit(Arm64Inst::LsrReg {
                        sf: 1,
                        rm: X28,
                        rn,
                        rd,
                    });
                }
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::IsTag(a, tag) => {
                // S::has_tag(bits, tag)
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                self.regs.dec_use(*a);
                let rn = gp_reg(ra, RegSize::X64);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                if S::HAS_UNBOXED_FLOAT {
                    // NanBox-style: shift right by PAYLOAD_BITS, compare upper bits
                    let expected = S::encode_tagged(*tag, 0) >> S::PAYLOAD_BITS;
                    self.emit_mov_imm(X28, S::PAYLOAD_BITS as u64);
                    self.buf.emit(Arm64Inst::LsrReg {
                        sf: 1,
                        rm: X28,
                        rn,
                        rd: X28,
                    });
                    let tmp = gp_reg(rd_idx, RegSize::X64);
                    self.emit_mov_imm(tmp, expected);
                    self.buf.emit(Arm64Inst::cmp(X28, tmp));
                } else {
                    // LowBit-style: mask low tag bits, compare with tag
                    let tag_mask = S::TAG_COUNT as u64 - 1;
                    self.emit_mov_imm(X28, tag_mask);
                    self.buf.emit(Arm64Inst::and(X28, rn, X28));
                    let tmp = gp_reg(rd_idx, RegSize::X64);
                    self.emit_mov_imm(tmp, *tag as u64);
                    self.buf.emit(Arm64Inst::cmp(X28, tmp));
                }
                let rd = gp_reg(rd_idx, RegSize::W32);
                self.buf.emit(Arm64Inst::cset(rd, Arm64Cond::EQ));
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::MakeTagged(tag, payload) => {
                // S::encode_tagged(tag, payload)
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *payload);
                self.regs.dec_use(*payload);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, RegSize::X64);
                let rn = gp_reg(ra, RegSize::X64);
                if S::HAS_UNBOXED_FLOAT {
                    // NanBox-style: OR payload (in low bits) with tag pattern
                    let pattern = S::encode_tagged(*tag, 0);
                    self.emit_mov_imm(X28, pattern);
                    self.buf.emit(Arm64Inst::orr(rd, rn, X28));
                } else {
                    // LowBit-style: shift payload left by tag_bits, OR with tag
                    let tag_bits = 64 - S::PAYLOAD_BITS;
                    self.emit_mov_imm(X28, tag_bits as u64);
                    self.buf.emit(Arm64Inst::LslReg {
                        sf: 1,
                        rm: X28,
                        rn,
                        rd,
                    });
                    self.emit_mov_imm(X28, *tag as u64);
                    self.buf.emit(Arm64Inst::orr(rd, rd, X28));
                }
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::TagOf(a) => {
                // Extract tag from bits
                let val = result_val.unwrap();
                let ra = self.regs.ensure_in_gp_reg(&mut self.buf, *a);
                self.regs.dec_use(*a);
                let rd_idx = self.regs.alloc_gp(&mut self.buf);
                let rd = gp_reg(rd_idx, RegSize::X64);
                let rn = gp_reg(ra, RegSize::X64);
                let tag_mask = S::TAG_COUNT as u64 - 1;
                if S::HAS_UNBOXED_FLOAT {
                    // NanBox-style: tag is in upper bits after payload
                    self.emit_mov_imm(X28, S::PAYLOAD_BITS as u64);
                    self.buf.emit(Arm64Inst::LsrReg {
                        sf: 1,
                        rm: X28,
                        rn,
                        rd,
                    });
                    self.emit_mov_imm(X28, tag_mask);
                    self.buf.emit(Arm64Inst::and(rd, rd, X28));
                } else {
                    // LowBit-style: tag is in low bits
                    self.emit_mov_imm(X28, tag_mask);
                    self.buf.emit(Arm64Inst::and(rd, rn, X28));
                }
                self.regs.assign_gp(val, rd_idx);
            }

            Inst::Safepoint(live) => {
                if let Some(handler) = self.safepoint_handler {
                    // Spill all live values so the handler can scan them
                    self.regs.spill_all_live(&mut self.buf);

                    // Call handler(frame_ptr, frame_size)
                    self.buf.emit(Arm64Inst::mov(X0, X29));
                    let frame_size = self.regs.frame.local_frame_size() as u64;
                    self.emit_mov_imm(gp_reg(1, RegSize::X64), frame_size);
                    self.emit_mov_imm(X28, handler);
                    self.buf.emit(Arm64Inst::blr(X28));

                    // After handler, all values are in spill slots
                    self.regs.clear_regs();
                }
                // Dec uses for live values
                for v in live.iter() {
                    self.regs.dec_use(*v);
                }
            }

            // Instructions we don't lower in the JIT (deopt/guard/etc)
            _ => {
                if result_val.is_some() {
                    panic!("unsupported instruction with result: {:?}", inst_node.inst);
                }
                // Consume uses for instructions without results
                inst_node.inst.for_each_value(|v| {
                    self.regs.dec_use(v);
                });
            }
        }
    }

    fn lower_gp_binop(&mut self, val: Value, a: Value, b: Value, op: BinOp) {
        let ra = self.regs.ensure_in_gp_reg(&mut self.buf, a);
        let rb = self.regs.ensure_in_gp_reg(&mut self.buf, b);
        let ty = self.regs.values[a.index()].ty;
        self.regs.dec_use(a);
        self.regs.dec_use(b);

        let rd_idx = self.regs.alloc_gp(&mut self.buf);
        let size = type_to_regsize(ty);
        let rd = gp_reg(rd_idx, size);
        let rn = gp_reg(ra, size);
        let rm = gp_reg(rb, size);

        match op {
            BinOp::Add => self.buf.emit(Arm64Inst::add(rd, rn, rm)),
            BinOp::Sub => self.buf.emit(Arm64Inst::sub(rd, rn, rm)),
            BinOp::Mul => self.buf.emit(Arm64Inst::mul(rd, rn, rm)),
            BinOp::SDiv => self.buf.emit(Arm64Inst::sdiv(rd, rn, rm)),
            BinOp::UDiv => self.buf.emit(Arm64Inst::udiv(rd, rn, rm)),
            BinOp::And => self.buf.emit(Arm64Inst::and(rd, rn, rm)),
            BinOp::Or => self.buf.emit(Arm64Inst::orr(rd, rn, rm)),
            BinOp::Xor => self.buf.emit(Arm64Inst::eor(rd, rn, rm)),
            BinOp::Shl => self.buf.emit(Arm64Inst::LslReg {
                sf: rd.sf(),
                rm,
                rn,
                rd,
            }),
            BinOp::LShr => self.buf.emit(Arm64Inst::LsrReg {
                sf: rd.sf(),
                rm,
                rn,
                rd,
            }),
            BinOp::AShr => self.buf.emit(Arm64Inst::AsrReg {
                sf: rd.sf(),
                rm,
                rn,
                rd,
            }),
        };

        self.regs.assign_gp(val, rd_idx);
    }

    fn lower_fp_binop(&mut self, val: Value, a: Value, b: Value, op: FpBinOp) {
        let ra = self.regs.ensure_in_fp_reg(&mut self.buf, a);
        let rb = self.regs.ensure_in_fp_reg(&mut self.buf, b);
        self.regs.dec_use(a);
        self.regs.dec_use(b);

        let rd_idx = self.regs.alloc_fp(&mut self.buf);
        let rd = Arm64Reg::new(rd_idx, RegSize::X64);
        let rn = Arm64Reg::new(ra, RegSize::X64);
        let rm = Arm64Reg::new(rb, RegSize::X64);

        match op {
            FpBinOp::Add => self.buf.emit(Arm64Inst::fadd(rd, rn, rm)),
            FpBinOp::Sub => self.buf.emit(Arm64Inst::fsub(rd, rn, rm)),
            FpBinOp::Mul => self.buf.emit(Arm64Inst::fmul(rd, rn, rm)),
            FpBinOp::Div => self.buf.emit(Arm64Inst::fdiv(rd, rn, rm)),
        };

        self.regs.assign_fp(val, rd_idx);
    }

    fn lower_call(&mut self, func_ref: FuncRef, args: &[Value], result_val: Option<Value>) {
        // Spill caller-saved regs
        self.regs.spill_caller_saved(&mut self.buf);

        let overflow = args.len().saturating_sub(16);
        let outgoing_size = align_up(overflow * 8, 16) as i32;
        self.regs.frame.reserve_outgoing_arg_bytes(outgoing_size);
        if outgoing_size > 0 {
            self.buf.emit(Arm64Inst::sub_imm(SP, SP, outgoing_size));
        }

        for (slot, &arg) in args.iter().enumerate() {
            let ty = self.regs.values[arg.index()].ty;
            if slot < 16 {
                let target = gp_reg(slot as u8, RegSize::X64);
                if RegState::is_float_type(ty) {
                    let src = self.regs.ensure_in_fp_reg(&mut self.buf, arg);
                    let rn = Arm64Reg::new(src, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_fp_to_gp(target, rn));
                } else {
                    let src = self.regs.ensure_in_gp_reg(&mut self.buf, arg);
                    if src != slot as u8 {
                        let rn = gp_reg(src, RegSize::X64);
                        self.buf.emit(Arm64Inst::mov(target, rn));
                    }
                }
            } else {
                let stack_offset = ((slot - 16) * 8) as i32;
                if RegState::is_float_type(ty) {
                    let src = self.regs.ensure_in_fp_reg(&mut self.buf, arg);
                    let rn = Arm64Reg::new(src, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_fp_to_gp(X27, rn));
                    self.buf.emit(Arm64Inst::str(X27, SP, stack_offset));
                } else {
                    let src = self.regs.ensure_in_gp_reg(&mut self.buf, arg);
                    let rn = gp_reg(src, RegSize::X64);
                    if src != 27 {
                        self.buf.emit(Arm64Inst::mov(X27, rn));
                    }
                    self.buf.emit(Arm64Inst::str(X27, SP, stack_offset));
                }
            }
        }

        // Load function pointer into X28 and BLR
        let func_idx = func_ref.index();
        if let Some(table_base) = self.call_table_base {
            // Module mode: load from indirect call table
            self.emit_mov_imm(X28, table_base);
            let offset = (func_idx * 8) as i32;
            self.buf.emit(Arm64Inst::ldr(X28, X28, offset));
        } else if func_idx < self.externs.len() {
            let ptr = self.externs[func_idx] as u64;
            self.emit_mov_imm(X28, ptr);
        } else {
            // null pointer - will crash, which is better than silently wrong behavior
            self.emit_mov_imm(X28, 0);
        }
        self.buf.emit(Arm64Inst::blr(X28));

        if outgoing_size > 0 {
            self.buf.emit(Arm64Inst::add_imm(SP, SP, outgoing_size));
        }

        for &arg in args {
            self.regs.dec_use(arg);
        }

        // Values loaded into caller-saved regs during arg setup must have
        // their locs restored to their spill slots (BLR clobbered the regs).
        for &r in CALLER_SAVED_GP {
            if let Some(val) = self.regs.gp_occupant[r as usize] {
                if let Some(slot) = self.regs.values[val.index()].spill_slot {
                    self.regs.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.regs.gp_occupant[r as usize] = None;
            }
        }
        for &r in CALLER_SAVED_FP {
            if let Some(val) = self.regs.fp_occupant[r as usize] {
                if let Some(slot) = self.regs.values[val.index()].spill_slot {
                    self.regs.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.regs.fp_occupant[r as usize] = None;
            }
        }

        // Result in X0 or D0
        if let Some(val) = result_val {
            let ty = self.regs.values[val.index()].ty;
            if RegState::is_float_type(ty) {
                self.regs.assign_fp(val, 0);
            } else {
                self.regs.assign_gp(val, 0);
            }
        }
    }

    fn lower_call_indirect(&mut self, callee: Value, args: &[Value], result_val: Option<Value>) {
        // Get callee pointer first
        let callee_reg = self.regs.ensure_in_gp_reg(&mut self.buf, callee);
        // Move to X28 before spilling (spilling might clobber the reg)
        let callee_hw = gp_reg(callee_reg, RegSize::X64);
        self.buf.emit(Arm64Inst::mov(X28, callee_hw));
        self.regs.dec_use(callee);

        // Spill caller-saved regs
        self.regs.spill_caller_saved(&mut self.buf);

        let overflow = args.len().saturating_sub(16);
        let outgoing_size = align_up(overflow * 8, 16) as i32;
        self.regs.frame.reserve_outgoing_arg_bytes(outgoing_size);
        if outgoing_size > 0 {
            self.buf.emit(Arm64Inst::sub_imm(SP, SP, outgoing_size));
        }

        for (slot, &arg) in args.iter().enumerate() {
            let ty = self.regs.values[arg.index()].ty;
            if slot < 16 {
                let target = gp_reg(slot as u8, RegSize::X64);
                if RegState::is_float_type(ty) {
                    let src = self.regs.ensure_in_fp_reg(&mut self.buf, arg);
                    let rn = Arm64Reg::new(src, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_fp_to_gp(target, rn));
                } else {
                    let src = self.regs.ensure_in_gp_reg(&mut self.buf, arg);
                    if src != slot as u8 {
                        let rn = gp_reg(src, RegSize::X64);
                        self.buf.emit(Arm64Inst::mov(target, rn));
                    }
                }
            } else {
                let stack_offset = ((slot - 16) * 8) as i32;
                if RegState::is_float_type(ty) {
                    let src = self.regs.ensure_in_fp_reg(&mut self.buf, arg);
                    let rn = Arm64Reg::new(src, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_fp_to_gp(X27, rn));
                    self.buf.emit(Arm64Inst::str(X27, SP, stack_offset));
                } else {
                    let src = self.regs.ensure_in_gp_reg(&mut self.buf, arg);
                    let rn = gp_reg(src, RegSize::X64);
                    if src != 27 {
                        self.buf.emit(Arm64Inst::mov(X27, rn));
                    }
                    self.buf.emit(Arm64Inst::str(X27, SP, stack_offset));
                }
            }
        }

        self.buf.emit(Arm64Inst::blr(X28));

        if outgoing_size > 0 {
            self.buf.emit(Arm64Inst::add_imm(SP, SP, outgoing_size));
        }

        for &arg in args {
            self.regs.dec_use(arg);
        }

        for &r in CALLER_SAVED_GP {
            if let Some(val) = self.regs.gp_occupant[r as usize] {
                if let Some(slot) = self.regs.values[val.index()].spill_slot {
                    self.regs.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.regs.gp_occupant[r as usize] = None;
            }
        }
        for &r in CALLER_SAVED_FP {
            if let Some(val) = self.regs.fp_occupant[r as usize] {
                if let Some(slot) = self.regs.values[val.index()].spill_slot {
                    self.regs.values[val.index()].loc = ValueLoc::Spill(slot);
                }
                self.regs.fp_occupant[r as usize] = None;
            }
        }

        if let Some(val) = result_val {
            let ty = self.regs.values[val.index()].ty;
            if RegState::is_float_type(ty) {
                self.regs.assign_fp(val, 0);
            } else {
                self.regs.assign_gp(val, 0);
            }
        }
    }

    fn lower_terminator(&mut self, block_idx: usize) {
        let block = &self.func.blocks[block_idx];
        match &block.terminator {
            Terminator::Ret(v) => {
                let ty = self.regs.values[v.index()].ty;
                if RegState::is_float_type(ty) {
                    // Return float bits in X0 (call_jit reads X0 as u64)
                    let r = self.regs.ensure_in_fp_reg(&mut self.buf, *v);
                    let rn = Arm64Reg::new(r, RegSize::X64);
                    self.buf.emit(Arm64Inst::fmov_fp_to_gp(X0, rn));
                } else {
                    let r = self.regs.ensure_in_gp_reg(&mut self.buf, *v);
                    if r != 0 {
                        let rn = gp_reg(r, RegSize::X64);
                        self.buf.emit(Arm64Inst::mov(X0, rn));
                    }
                }
                self.regs.dec_use(*v);
                self.emit_epilogue();
            }

            Terminator::RetVoid => {
                self.emit_epilogue();
            }

            Terminator::Jump(target, args) => {
                // Spill all live values so they survive the register state
                // reset at the target block entry
                self.regs.spill_all_live(&mut self.buf);
                self.emit_block_args(*target, args);
                let label = self.block_meta[target.index()].label;
                let offset = self.buf.emit(Arm64Inst::b(0));
                self.buf.add_reloc(offset, label, Arm64RelocKind::Branch26);
            }

            Terminator::BrIf {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                let rc = self.regs.ensure_in_gp_reg(&mut self.buf, *cond);
                // Don't dec_use cond yet - keep its register occupied so
                // spill_all_live won't allocate it for something else.
                // spill_all_live will spill it (harmless, just a redundant store).

                // Spill all live values before the conditional branch
                self.regs.spill_all_live(&mut self.buf);

                // Now safe to use rc - spill_all_live only emits STR (reads regs)
                let cond_reg = gp_reg(rc, RegSize::W32);
                self.regs.dec_use(*cond);

                let else_tramp = self.buf.create_label();
                let cbz_offset = self.buf.emit(Arm64Inst::cbz(cond_reg, 0));
                self.buf
                    .add_reloc(cbz_offset, else_tramp, Arm64RelocKind::Cond19);

                // Save value locs and reg state before the then path, so we can
                // restore them for the else path. At runtime only one path
                // executes, so they must each start from the post-spill state.
                let saved_values: Vec<(ValueLoc, Option<i32>)> = self
                    .regs
                    .values
                    .iter()
                    .map(|v| (v.loc, v.spill_slot))
                    .collect();
                let saved_gp = self.regs.gp_occupant;
                let saved_fp = self.regs.fp_occupant;

                // Then path: store args and branch
                self.store_block_args_for_branch(*then_block, then_args);
                let then_label = self.block_meta[then_block.index()].label;
                let b_offset = self.buf.emit(Arm64Inst::b(0));
                self.buf
                    .add_reloc(b_offset, then_label, Arm64RelocKind::Branch26);

                // Restore value locs and reg state for else path
                for (i, (loc, spill)) in saved_values.into_iter().enumerate() {
                    self.regs.values[i].loc = loc;
                    self.regs.values[i].spill_slot = spill;
                }
                self.regs.gp_occupant = saved_gp;
                self.regs.fp_occupant = saved_fp;

                // Else trampoline: store args and branch
                self.buf.bind_label(else_tramp);
                self.store_block_args_for_branch(*else_block, else_args);
                let else_label = self.block_meta[else_block.index()].label;
                if else_block.index() != block_idx + 1 {
                    let b_offset = self.buf.emit(Arm64Inst::b(0));
                    self.buf
                        .add_reloc(b_offset, else_label, Arm64RelocKind::Branch26);
                }
                // If else is next block, fall through
            }

            Terminator::Switch {
                val,
                cases,
                default_block,
                default_args,
            } => {
                // Spill everything first
                let rv = self.regs.ensure_in_gp_reg(&mut self.buf, *val);
                self.regs.spill_all_live(&mut self.buf);

                // For each case: CMP + B.EQ
                for (case_val, target, args) in cases {
                    self.emit_mov_imm(X28, *case_val as u64);
                    let rn = gp_reg(rv, RegSize::X64);
                    self.buf.emit(Arm64Inst::cmp(rn, X28));

                    if !args.is_empty() {
                        self.store_block_args_to_canonical(*target, args);
                    }

                    let label = self.block_meta[target.index()].label;
                    let offset = self.buf.emit(Arm64Inst::b_cond(Arm64Cond::EQ, 0));
                    self.buf.add_reloc(offset, label, Arm64RelocKind::Cond19);
                }

                self.regs.dec_use(*val);

                // Default
                self.emit_block_args(*default_block, default_args);
                let label = self.block_meta[default_block.index()].label;
                let offset = self.buf.emit(Arm64Inst::b(0));
                self.buf.add_reloc(offset, label, Arm64RelocKind::Branch26);
            }

            Terminator::Unreachable => {
                self.buf.emit(Arm64Inst::brk(1));
            }

            _ => {
                panic!("unsupported terminator: {:?}", block.terminator);
            }
        }
    }

    /// Store block args to canonical spill slots for multi-pred blocks,
    /// or set up renaming for single-pred blocks.
    fn emit_block_args(&mut self, target: BlockId, args: &[Value]) {
        if args.is_empty() {
            return;
        }

        // All blocks with params use canonical spill slots
        self.store_block_args_to_canonical(target, args);
    }

    /// Store block args to canonical spill slots for a branch target.
    fn store_block_args_for_branch(&mut self, target: BlockId, args: &[Value]) {
        if args.is_empty() {
            return;
        }
        self.store_block_args_to_canonical(target, args);
    }

    fn store_block_args_to_canonical(&mut self, target: BlockId, args: &[Value]) {
        let target_idx = target.index();
        for (i, &arg) in args.iter().enumerate() {
            let slot_offset = self.block_meta[target_idx].param_spill_slots[i];
            let ty = self.regs.values[arg.index()].ty;

            if RegState::is_float_type(ty) {
                let r = self.regs.ensure_in_fp_reg(&mut self.buf, arg);
                let dreg = Arm64Reg::new(r, RegSize::X64);
                emit_fp_store_to_fp(&mut self.buf, dreg, slot_offset);
            } else {
                let r = self.regs.ensure_in_gp_reg(&mut self.buf, arg);
                let reg = gp_reg(r, RegSize::X64);
                emit_store_to_fp(&mut self.buf, reg, slot_offset);
            }
            self.regs.dec_use(arg);
        }
    }

    fn emit_mov_imm(&mut self, rd: Arm64Reg, value: u64) {
        let insts = Arm64Inst::mov_imm64(rd, value);
        for inst in insts {
            self.buf.emit(inst);
        }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

#[derive(Clone, Copy)]
enum FpBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn type_to_regsize(ty: Type) -> RegSize {
    match ty {
        Type::I8 | Type::I32 => RegSize::W32,
        Type::I64 | Type::Ptr | Type::GcPtr => RegSize::X64,
        Type::F64 => RegSize::X64, // shouldn't be used for GP regs
    }
}

fn gp_reg(index: u8, size: RegSize) -> Arm64Reg {
    Arm64Reg::new(index, size)
}

fn cmpop_to_cond(op: CmpOp) -> Arm64Cond {
    match op {
        CmpOp::Eq => Arm64Cond::EQ,
        CmpOp::Ne => Arm64Cond::NE,
        CmpOp::Slt => Arm64Cond::LT,
        CmpOp::Sle => Arm64Cond::LE,
        CmpOp::Sgt => Arm64Cond::GT,
        CmpOp::Sge => Arm64Cond::GE,
        CmpOp::Ult => Arm64Cond::CC, // LO
        CmpOp::Ule => Arm64Cond::LS,
        CmpOp::Ugt => Arm64Cond::HI,
        CmpOp::Uge => Arm64Cond::CS, // HS
    }
}

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Store a GP reg to [X29, #offset] where offset is positive and 8-byte aligned.
fn emit_store_to_fp(buf: &mut CodeBuffer<Arm64>, rt: Arm64Reg, offset: i32) {
    debug_assert!(offset >= 0 && offset % 8 == 0);
    buf.emit(Arm64Inst::str(rt, X29, offset));
}

/// Load a GP reg from [X29, #offset].
fn emit_load_from_fp(buf: &mut CodeBuffer<Arm64>, rt: Arm64Reg, offset: i32) {
    debug_assert!(offset >= 0 && offset % 8 == 0);
    buf.emit(Arm64Inst::ldr(rt, X29, offset));
}

/// Store an FP reg to [X29, #offset].
fn emit_fp_store_to_fp(buf: &mut CodeBuffer<Arm64>, rt: Arm64Reg, offset: i32) {
    debug_assert!(offset >= 0 && offset % 8 == 0);
    buf.emit(Arm64Inst::str_fp(rt, X29, offset));
}

/// Load an FP reg from [X29, #offset].
fn emit_fp_load_from_fp(buf: &mut CodeBuffer<Arm64>, rt: Arm64Reg, offset: i32) {
    debug_assert!(offset >= 0 && offset % 8 == 0);
    buf.emit(Arm64Inst::ldr_fp(rt, X29, offset));
}
