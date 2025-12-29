use crate::arm_instructions as arm;
use crate::gc::types::BuiltInTypes;
use crate::ir::{CallTarget, Condition, Instruction, IrValue, Label, VirtualRegister};
use crate::register_allocation::linear_scan::LinearScan;
use crate::trampoline::Trampoline;
use std::collections::{BTreeMap, HashMap};

/// Result of compiling a function, includes code pointer and stack map
pub struct CompiledFunction {
    /// Pointer to executable code
    pub code_ptr: usize,
    /// Length of the code in 32-bit instructions
    pub code_len: usize,
    /// Stack map entries: (absolute_pc, stack_size)
    pub stack_map: Vec<(usize, usize)>,
    /// Total number of stack slots (spills + exceptions + locals)
    /// Used by GC to know how many slots to scan before dynamic saves
    pub num_stack_slots: usize,
}

/// ARM64 code generator - compiles IR to ARM64 machine code
///
/// This is based on Beagle's ARM64 backend but simplified for our needs.
pub struct Arm64CodeGen {
    /// Generated ARM64 machine code (32-bit instructions)
    code: Vec<u32>,

    /// Map from virtual registers to physical ARM64 registers (from linear scan)
    register_map: BTreeMap<VirtualRegister, VirtualRegister>,

    /// Map from labels to code positions (for fixups)
    label_positions: HashMap<Label, usize>,

    /// Pending jump fixups: (code_index, label)
    pending_fixups: Vec<(usize, Label)>,

    /// Pending ADR fixups: (code_index, label)
    /// ADR instructions need to be patched with PC-relative offsets
    pending_adr_fixups: Vec<(usize, Label)>,

    /// Pool of temporary registers for spill loads (x9, x10, x11)
    temp_register_pool: Vec<usize>,

    /// Counter for generating unique labels
    label_counter: usize,

    /// Track placeholder positions for deferred stack allocation
    /// Maps function label -> (prologue_index, epilogue_index)
    placeholder_positions: HashMap<Label, (usize, usize)>,

    /// Track current function's accumulated stack bytes (for CallWithSaves)
    current_function_stack_bytes: usize,

    /// Number of stack slots needed (from register allocator)
    num_stack_slots: usize,

    /// Stack space allocated for spills in per-function compilation
    function_stack_space: usize,

    // === Stack Map for GC ===
    /// Maps instruction offset → stack size (in words) at that point
    /// Used by GC to find roots on the stack after calls
    stack_map: HashMap<usize, usize>,

    /// Current stack size in words (for stack map tracking)
    current_stack_size: usize,

    /// Maximum stack size in words (for stack map metadata)
    max_stack_size: usize,

    /// Number of locals in current function
    num_locals: usize,

    /// Number of spill slots from register allocation
    /// Used to compute exception slot offsets (exception slots come after spill slots)
    num_spill_slots: usize,

    /// Number of reserved exception slots (from compiler)
    reserved_exception_slots: usize,

    /// Current instruction being compiled (for error reporting)
    current_instruction: Option<String>,
}

impl Default for Arm64CodeGen {
    fn default() -> Self {
        Self::new()
    }
}

impl Arm64CodeGen {
    pub fn new() -> Self {
        Arm64CodeGen {
            code: Vec::new(),
            register_map: BTreeMap::new(),
            label_positions: HashMap::new(),
            pending_fixups: Vec::new(),
            pending_adr_fixups: Vec::new(),
            temp_register_pool: vec![11, 10, 9],
            label_counter: 0,
            placeholder_positions: HashMap::new(),
            current_function_stack_bytes: 0,
            num_stack_slots: 0,
            function_stack_space: 0,
            // Stack map for GC
            stack_map: HashMap::new(),
            current_stack_size: 0,
            max_stack_size: 0,
            num_locals: 0,
            num_spill_slots: 0,
            reserved_exception_slots: 0,
            // Current instruction for error reporting
            current_instruction: None,
        }
    }

    fn new_label(&mut self) -> Label {
        // Use "__internal_" prefix to avoid conflicts with IR-generated labels (L0, L1, etc.)
        let label = format!("__internal_{}", self.label_counter);
        self.label_counter += 1;
        label
    }

    /// Compile a single function's IR to executable machine code
    ///
    /// This is used for per-function compilation (Beagle's approach):
    /// - Each function is compiled separately with its own register allocation
    /// - Returns a CompiledFunction with code pointer and stack map
    ///
    /// Stack frame layout (following Beagle ABI):
    /// - FP, LR saved at [sp, #-16]!
    /// - Spill slots at FP-relative offsets (initialized to null)
    /// - Exception slots after spill slots (initialized to null)
    /// - Local variable slots after exception slots (initialized to null)
    /// - Dynamic save slots from CallWithSaves (pushed/popped as needed)
    ///
    /// num_locals: Number of local variable slots from IrBuilder (for storing arguments)
    pub fn compile_function(
        instructions: &[Instruction],
        num_locals: usize,
        reserved_exception_slots: usize,
    ) -> Result<CompiledFunction, String> {
        let mut codegen = Arm64CodeGen::new();
        codegen.num_locals = num_locals;
        codegen.reserved_exception_slots = reserved_exception_slots;

        // Run register allocation for THIS function only
        let mut allocator = LinearScan::new(instructions.to_vec(), 0);
        allocator.allocate();
        let num_spill_slots = allocator.num_stack_slots();

        // Store spill slot count for exception offset calculation
        codegen.num_spill_slots = num_spill_slots;

        // Store allocation map
        codegen.register_map = allocator.allocated_registers.clone();

        let allocated_instructions = allocator.finish();

        // Calculate total stack slots: spill slots + exception slots + local variable slots
        // Layout: [FP] [spills] [exceptions] [locals]
        let total_stack_slots = num_spill_slots + reserved_exception_slots + num_locals;

        // Calculate stack space (round up to 16-byte alignment)
        let stack_space = if total_stack_slots > 0 {
            (total_stack_slots * 8).div_ceil(16) * 16
        } else {
            0
        };

        // Emit function prologue (Beagle-style with placeholder for patching)
        // Stack layout after prologue (following Beagle ABI):
        //   [FP + 8]:       saved x30 (LR)
        //   [FP + 0]:       saved x29 (old FP) <- FP points here
        //   [FP - 8]:       spill slot 0 / local 0 (initialized to null)
        //   [FP - 16]:      spill slot 1 / local 1 (initialized to null)
        //   ...
        //   [FP - N*8]:     dynamic save slot 0 (from CallWithSaves push_to_stack)
        //   ...
        //   [SP]:           <- SP after stack allocation (patched later)
        //
        // ABI: Every function saves its own registers via CallWithSaves.
        // NO callee-saved registers are saved in prologue/epilogue.
        // Beagle approach: Emit placeholder SUB, patch with actual size after compilation.

        // Step 1: Save FP and LR
        codegen.emit_stp(29, 30, 31, -2); // stp x29, x30, [sp, #-16]!

        // Step 2: Set FP to current SP (before any stack allocation)
        codegen.emit_mov(29, 31); // mov x29, sp

        // Step 3: Emit placeholder SUB for stack allocation (will be patched)
        // Magic value 0xAAA identifies this as a placeholder (fits in 12-bit immediate)
        let prologue_placeholder_index = codegen.code.len();
        codegen.emit_sub_sp_imm(0xAAA); // Placeholder - will be patched

        // Store num_stack_slots BEFORE set_all_locals_to_null so it knows how many slots to init
        codegen.num_stack_slots = total_stack_slots;

        // Step 4: Initialize all local/spill slots to null (Beagle pattern)
        // This ensures GC won't see garbage as heap pointers during stack scanning.
        // Use x15 as scratch register (it's caller-saved and not used for arguments)
        codegen.emit_mov_imm(15, BuiltInTypes::null_value() as i64); // Load null (0b111)
        codegen.set_all_locals_to_null(15);

        // Store info for Ret to emit correct epilogue and for patching
        codegen.function_stack_space = stack_space; // Initial estimate (spills only)

        // Reset stack tracking for push/pop during compilation
        codegen.current_stack_size = 0;
        codegen.max_stack_size = 0;

        // Compile instructions
        for inst in &allocated_instructions {
            codegen.compile_instruction(inst)?;
        }

        // Emit epilogue (Beagle-style: add sp placeholder, ldp, ret)
        codegen.emit_add_sp_imm(0xAAA); // Placeholder - will be patched
        codegen.emit_ldp(29, 30, 31, 2); // ldp x29, x30, [sp], #16
        codegen.emit_ret();

        // Patch prologue and epilogue with actual stack size (Beagle-style)
        // Total stack = spill slots + max dynamic stack (from push_to_stack)
        let total_stack_words = total_stack_slots + codegen.max_stack_size;
        // Round up to even for 16-byte alignment
        let aligned_stack_words = if total_stack_words % 2 != 0 {
            total_stack_words + 1
        } else {
            total_stack_words
        };
        let stack_bytes = aligned_stack_words * 8;
        codegen.patch_prologue_epilogue(prologue_placeholder_index, stack_bytes);

        // Apply fixups
        codegen.apply_fixups()?;

        // Allocate and copy to executable memory
        let code_ptr = Trampoline::execute_code(&codegen.code);

        // Translate stack map to absolute addresses
        let stack_map = codegen.translate_stack_map(code_ptr);

        Ok(CompiledFunction {
            code_ptr,
            code_len: codegen.code.len(),
            stack_map,
            num_stack_slots: codegen.num_stack_slots,
        })
    }

    fn compile_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        // Track current instruction for error reporting
        self.current_instruction = Some(format!("{:?}", inst));

        match inst {
            Instruction::LoadConstant(dst, value) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                match value {
                    IrValue::TaggedConstant(c) => {
                        self.emit_mov_imm(dst_reg, *c as i64);
                    }
                    IrValue::True => {
                        // true: (1 << 3) | 0b011 = 11
                        self.emit_mov_imm(dst_reg, 11);
                    }
                    IrValue::False => {
                        // false: (0 << 3) | 0b011 = 3
                        self.emit_mov_imm(dst_reg, 3);
                    }
                    IrValue::Null => {
                        // nil: 0b111 = 7
                        self.emit_mov_imm(dst_reg, 7);
                    }
                    IrValue::RawConstant(c) => {
                        // Raw (untagged) constant - used for type IDs, field counts, etc.
                        self.emit_mov_imm(dst_reg, *c);
                    }
                    IrValue::FramePointer => {
                        // Use FP (x29) for GC stack walking via frame pointer chain traversal.
                        // The GC walks the FP chain to find stack frames.
                        self.emit_mov(dst_reg, 29);
                    }
                    IrValue::ReturnAddress => {
                        // Use LR (x30) for GC stack map lookup.
                        // The return address identifies which stack map entry describes the frame.
                        self.emit_mov(dst_reg, 30);
                    }
                    _ => return Err(format!("Invalid constant: {:?}", value)),
                }
                self.store_spill(dst_reg, dest_spill);
            }

            // Note: LoadVarBySymbol, LoadVarBySymbolDynamic, StoreVarBySymbol, EnsureVarBySymbol,
            // and LoadKeyword have been converted to builtin function calls.
            // They are now handled by the regular Call instruction codegen.
            Instruction::LoadTrue(dst) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                // true: (1 << 3) | 0b011 = 11
                self.emit_mov_imm(dst_reg, 11);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadFalse(dst) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                // false: (0 << 3) | 0b011 = 3
                self.emit_mov_imm(dst_reg, 3);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Untag(dst, src) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                // Handle TaggedConstant: compute untagged value at compile time
                match src {
                    IrValue::TaggedConstant(val) => {
                        // Untag: arithmetic right shift by 3 (preserves sign for negative integers)
                        let untagged = *val >> 3;
                        self.emit_mov_imm(dst_reg, untagged as i64);
                    }
                    IrValue::Null => {
                        // nil >> 3 = 0
                        self.emit_mov_imm(dst_reg, 0);
                    }
                    _ => {
                        let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                        // Untag: arithmetic right shift by 3 (preserves sign for negative integers)
                        self.emit_asr_imm(dst_reg, src_reg, 3);
                    }
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Tag(dst, src, tag) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                // Check if tag is a RawConstant specifying a non-zero tag (e.g., heap object tag 0b110)
                // If so, we OR with the tag. Otherwise, we shift left by 3 (integer tagging).
                let tag_value: Option<u64> = match tag {
                    IrValue::RawConstant(t) if *t != 0 => Some(*t as u64),
                    IrValue::Register(_) => None, // Tag is in a register, will OR dynamically
                    _ => None,                    // Zero tag or TaggedConstant - use shift
                };

                match (src, tag_value) {
                    // TaggedConstant with explicit tag: shift then OR
                    (IrValue::TaggedConstant(val), Some(t)) => {
                        let tagged = ((*val as u64) << 3) | t;
                        self.emit_mov_imm(dst_reg, tagged as i64);
                    }
                    // TaggedConstant without explicit tag: just shift
                    (IrValue::TaggedConstant(val), None) => {
                        let tagged = (*val as i64) << 3;
                        self.emit_mov_imm(dst_reg, tagged);
                    }
                    (IrValue::Null, _) => {
                        // 0 << 3 = 0
                        self.emit_mov_imm(dst_reg, 0);
                    }
                    // Register/Spill with explicit tag: shift left by 3 and OR with tag
                    // This is the correct tagging scheme: (ptr << 3) | tag
                    // Untagging is done by >> 3
                    (_, Some(t)) => {
                        let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                        // Step 1: Shift pointer left by 3
                        self.emit_lsl_imm(dst_reg, src_reg, 3);
                        // Step 2: OR with tag value using immediate form (no scratch register needed)
                        self.emit_orr_imm(dst_reg, dst_reg, t as u32);
                    }
                    // Register/Spill without explicit tag: shift left by 3 (integer tagging)
                    (_, None) => {
                        let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                        // Check if tag is a register (dynamic tagging)
                        match tag {
                            IrValue::Register(_) | IrValue::Spill(_, _) => {
                                // Dynamic tag: OR src with tag register
                                let tag_reg = self.get_physical_reg_for_irvalue(tag, false)?;
                                self.emit_orr(dst_reg, src_reg, tag_reg);
                            }
                            _ => {
                                // Default: left shift by 3 (int tag is 000)
                                self.emit_lsl_imm(dst_reg, src_reg, 3);
                            }
                        }
                    }
                }
                self.store_spill(dst_reg, dest_spill);
            }

            // Memory operations
            Instruction::HeapStore(ptr, offset, value) => {
                let ptr_reg = self.get_physical_reg_for_irvalue(ptr, false)?;
                let value_reg = self.get_physical_reg_for_irvalue(value, false)?;
                // Store 64-bit value to ptr + offset*8
                self.emit_str_offset(value_reg, ptr_reg, *offset * 8);
            }

            Instruction::AddInt(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                // eprintln!("DEBUG: AddInt - dst={:?} (x{}), src1={:?} (x{}), src2={:?} (x{})",
                //          dst, dst_reg, src1, src1_reg, src2, src2_reg);
                self.emit_add(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Sub(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_sub(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Mul(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_mul(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Div(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_sdiv(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            // Bitwise operations (work on untagged values)
            Instruction::BitAnd(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_and(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::BitOr(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_orr(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::BitXor(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_eor(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::BitNot(dst, src) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                self.emit_mvn(dst_reg, src_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::BitShiftLeft(dst, src, amt) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                let amt_reg = self.get_physical_reg_for_irvalue(amt, false)?;
                self.emit_lsl(dst_reg, src_reg, amt_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::BitShiftRight(dst, src, amt) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                let amt_reg = self.get_physical_reg_for_irvalue(amt, false)?;
                self.emit_asr(dst_reg, src_reg, amt_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::UnsignedBitShiftRight(dst, src, amt) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                let amt_reg = self.get_physical_reg_for_irvalue(amt, false)?;
                self.emit_lsr(dst_reg, src_reg, amt_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            // Float arithmetic operations
            // For floats, src1/src2 contain untagged f64 bits in general registers
            // We move to FP registers, perform the op, and move result back
            Instruction::AddFloat(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                // Move to FP registers d0, d1
                self.emit_fmov_general_to_float(0, src1_reg);
                self.emit_fmov_general_to_float(1, src2_reg);
                // Add
                self.emit_fadd(0, 0, 1);
                // Move result back to general register
                self.emit_fmov_float_to_general(dst_reg, 0);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::SubFloat(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_fmov_general_to_float(0, src1_reg);
                self.emit_fmov_general_to_float(1, src2_reg);
                self.emit_fsub(0, 0, 1);
                self.emit_fmov_float_to_general(dst_reg, 0);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::MulFloat(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_fmov_general_to_float(0, src1_reg);
                self.emit_fmov_general_to_float(1, src2_reg);
                self.emit_fmul(0, 0, 1);
                self.emit_fmov_float_to_general(dst_reg, 0);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::DivFloat(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_fmov_general_to_float(0, src1_reg);
                self.emit_fmov_general_to_float(1, src2_reg);
                self.emit_fdiv(0, 0, 1);
                self.emit_fmov_float_to_general(dst_reg, 0);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::IntToFloat(dst, src) => {
                // Convert signed integer to double
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // SCVTF converts integer in general reg to float in FP reg
                self.emit_scvtf(0, src_reg);
                // Move result back to general register (as raw bits)
                self.emit_fmov_float_to_general(dst_reg, 0);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::GetTag(dst, src) => {
                // Extract the tag bits (last 3 bits) from a tagged value
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // AND with 0b111 to extract tag
                self.emit_and_imm(dst_reg, src_reg, 0b111);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadFloat(dst, src) => {
                // Load f64 bits from a heap-allocated float
                // src contains tagged float pointer
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // Untag: logical shift right by 3 to get heap pointer
                self.emit_lsr_imm(dst_reg, src_reg, 3);
                // Load f64 bits from offset 8 (skip header)
                self.emit_ldr_offset(dst_reg, dst_reg, 8);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Assign(dst, src) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                // Handle TaggedConstant inline (like Beagle does)
                match src {
                    IrValue::TaggedConstant(val) => {
                        self.emit_mov_imm(dst_reg, *val as i64);
                    }
                    IrValue::Null => {
                        self.emit_mov_imm(dst_reg, 7); // nil is tagged as 7
                    }
                    _ => {
                        let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                        if dst_reg != src_reg {
                            self.emit_mov(dst_reg, src_reg);
                        }
                    }
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Label(label) => {
                // Record position of this label
                let pos = self.code.len();
                self.label_positions.insert(label.clone(), pos);
                // Note: With per-function compilation, functions are NOT compiled inline.
                // Each function has its own prologue/epilogue emitted by compile_function().
            }

            Instruction::Jump(label) => {
                // Emit unconditional branch
                // We'll fix up the offset later
                let fixup_index = self.code.len();
                // eprintln!("DEBUG: Jump to {} from code position {}", label, fixup_index);
                self.pending_fixups.push((fixup_index, label.clone()));
                // Placeholder - will be patched in apply_fixups
                self.code.push(0x14000000); // B #0
            }

            Instruction::JumpIf(label, cond, src1, src2) => {
                // Compare src1 and src2
                // Handle src1 being a constant (load into temp register)
                let src1_reg = match src1 {
                    IrValue::TaggedConstant(val) => {
                        let temp_reg = self.allocate_temp_register();
                        self.emit_mov_imm(temp_reg, *val as i64);
                        temp_reg
                    }
                    IrValue::Null => {
                        let temp_reg = self.allocate_temp_register();
                        self.emit_mov_imm(temp_reg, 7); // nil is tagged as 7
                        temp_reg
                    }
                    _ => self.get_physical_reg_for_irvalue(src1, false)?,
                };

                // Handle comparing with immediate values
                match src2 {
                    IrValue::TaggedConstant(imm) => {
                        // Use CMP with immediate
                        self.emit_cmp_imm(src1_reg, *imm as i64);
                    }
                    IrValue::Null => {
                        self.emit_cmp_imm(src1_reg, 7); // nil is tagged as 7
                    }
                    _ => {
                        let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                        self.emit_cmp(src1_reg, src2_reg);
                    }
                }
                self.clear_temp_registers();

                // Emit conditional branch
                let fixup_index = self.code.len();
                self.pending_fixups.push((fixup_index, label.clone()));

                // Placeholder conditional branch - will be patched in apply_fixups
                let branch_cond = match cond {
                    Condition::Equal => 0,               // EQ
                    Condition::NotEqual => 1,            // NE
                    Condition::LessThan => 11,           // LT
                    Condition::LessThanOrEqual => 13,    // LE
                    Condition::GreaterThan => 12,        // GT
                    Condition::GreaterThanOrEqual => 10, // GE
                };

                // B.cond #0 (placeholder)
                self.code.push(0x54000000 | branch_cond);
            }

            Instruction::Compare(dst, src1, src2, cond) => {
                // Compare and set result to true/false (tagged bools: 11 or 3)
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;

                // CMP src1, src2
                self.emit_cmp(src1_reg, src2_reg);

                // CSET dst, condition (sets dst to 1 if condition is true, 0 otherwise)
                let cond_code = match cond {
                    Condition::Equal => 0,               // EQ
                    Condition::NotEqual => 1,            // NE
                    Condition::LessThan => 11,           // LT
                    Condition::LessThanOrEqual => 13,    // LE
                    Condition::GreaterThan => 12,        // GT
                    Condition::GreaterThanOrEqual => 10, // GE
                };

                // CSET is CSINC dst, XZR, XZR, invert(cond)
                // This sets dst to 1 if true, 0 if false
                let inverted_cond = cond_code ^ 1; // Invert the condition
                let instruction = 0x9A9F07E0 | (inverted_cond << 12) | (dst_reg as u32);
                self.code.push(instruction);

                // Now convert 0/1 to tagged bools: 3 (false) or 11 (true)
                // LSL dst, dst, #3  - Shift left by 3: 0→0, 1→8
                self.emit_lsl_imm(dst_reg, dst_reg, 3);
                // ADD dst, dst, #3  - Add 3: 0→3, 8→11
                self.emit_add_imm(dst_reg, dst_reg, 3);

                self.store_spill(dst_reg, dest_spill);
            }

            // Stack operations for closure handling (Beagle pattern)
            Instruction::CurrentStackPosition(dst) => {
                // Get pointer to where next push will go
                // This is FP - ((num_stack_slots + current_stack_size + 1) * 8)
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                self.get_current_stack_position(dst_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::PushToStack(value) => {
                // Push value to FP-relative stack, increment stack_size
                let src_reg = self.get_physical_reg_for_irvalue(value, false)?;
                self.push_to_stack(src_reg);
            }

            Instruction::PopFromStack(count) => {
                // Logically pop N values (just decrements stack_size)
                self.decrement_stack_size(*count);
            }

            Instruction::MakeFunctionPtr(dst, code_ptr, values_ptr, closure_count) => {
                // MakeFunctionPtr: create function with raw code pointer
                // For closures: values_ptr points to FP-relative stack slots
                // The compiler already emitted CurrentStackPosition + PushToStack for closure values

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                if *closure_count == 0 {
                    // Regular function (no closures) - just tag the code pointer
                    // Tagged value = (code_ptr << 3) | 0b100

                    // Load raw code pointer into temp register
                    let temp_reg = 10;
                    self.emit_mov_imm(temp_reg, *code_ptr as i64);

                    // Shift left by 3 and add Function tag (0b100)
                    let lsl_instruction =
                        0xD37DF000u32 | ((temp_reg as u32) << 5) | (dst_reg as u32);
                    self.code.push(lsl_instruction);

                    // ADD Xd, Xn, #0b100 (set tag bits)
                    let add_instruction =
                        0x91000000u32 | (0b100 << 10) | ((dst_reg as u32) << 5) | (dst_reg as u32);
                    self.code.push(add_instruction);
                } else {
                    // Closure - allocate heap object
                    // values_ptr already points to pushed closure values (from CurrentStackPosition)
                    // Values already pushed via PushToStack instructions

                    let values_ptr_reg = self.get_physical_reg_for_irvalue(values_ptr, false)?;

                    // Set up arguments for trampoline call
                    // Args: x0=frame_pointer, x1=name_ptr, x2=code_ptr, x3=closure_count, x4=values_ptr
                    // gc_return_addr is computed internally by the builtin
                    self.emit_mov(0, 29); // x0 = FP for GC
                    self.emit_mov_imm(1, 0); // x1 = 0 (anonymous)
                    self.emit_mov_imm(2, *code_ptr as i64); // x2 = code_ptr
                    self.emit_mov_imm(3, *closure_count as i64); // x3 = closure_count
                    self.emit_mov(4, values_ptr_reg); // x4 = values_ptr

                    // Call trampoline to allocate closure heap object
                    // emit_external_call -> emit_blr -> update_stack_map()
                    // Stack map includes the pushed closure values!
                    let func_addr = crate::trampoline::builtin_allocate_function as usize;
                    self.emit_external_call(func_addr, "builtin_allocate_function");

                    // Result is in x0 (tagged closure pointer)
                    if dst_reg != 0 {
                        self.emit_mov(dst_reg, 0);
                    }
                }

                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadClosure(dst, fn_obj, index) => {
                // LoadClosure: Load a captured variable from closure object
                // The closure object is in fn_obj register (x0 for closures, passed as first arg)
                // IMPORTANT: fn_obj is TAGGED with Closure tag (0b101), must untag first!
                // Layout: [header(8), name_ptr(8), code_ptr(8), closure_count(8), closure_values...]
                // Using constants from gc_runtime::closure_layout

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let fn_obj_reg = self.get_physical_reg_for_irvalue(fn_obj, false)?;

                // Untag the closure pointer (shift right by 3)
                // LSR Xd, Xn, #3 - From Beagle: 0xD343FC00
                let untagged_reg = 11; // Use x11 as temporary for untagged pointer
                let lsr_instruction =
                    0xD343FC00u32 | ((fn_obj_reg as u32) << 5) | (untagged_reg as u32);
                self.code.push(lsr_instruction);

                // Load closure value from heap object using untagged pointer
                // Use closure_layout constant for offset calculation
                use crate::gc_runtime::closure_layout;
                let offset = closure_layout::value_offset(*index) as i32;
                // eprintln!("DEBUG: LoadClosure - index={}, offset={}, dst={:?} -> x{}, spill={:?}",
                //          index, offset, dst, dst_reg, dest_spill);
                self.emit_ldr_offset(dst_reg, untagged_reg, offset);

                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Call(dst, fn_val, args) => {
                // Call: Invoke a function with arguments
                //
                // Following Beagle's approach - NO hardcoded x19!
                // 1. Get function pointer from its allocated register
                // 2. Extract tag (fn_val & 0b111)
                // 3. If Function tag (0b100): untag to get code_ptr, args in x0-x7
                // 4. If Closure tag (0b101): load code_ptr from heap, closure in x0, user args in x1-x7
                // 5. Call the function
                // 6. Get result from x0
                //
                // Since allocator uses x19-x28, fn_reg won't conflict with x0-x7 argument setup.

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                if args.len() > 8 {
                    return Err("Call with more than 8 arguments not yet supported".to_string());
                }

                // Get function pointer in its allocated register (no move to hardcoded x19!)
                let fn_reg = self.get_physical_reg_for_irvalue(fn_val, false)?;

                // Collect argument source registers
                // x0-x7 are never used by the allocator (which only uses x19-x28).
                let mut arg_source_regs = Vec::new();
                for arg in args.iter() {
                    let arg_reg = self.get_physical_reg_for_irvalue(arg, false)?;
                    arg_source_regs.push(arg_reg);
                }

                // Use temp registers from pool for tag/closure_ptr, but NOT for code_ptr!
                // x18 is RESERVED by macOS and must NEVER be used!
                // x9 is used for arg_count in closure calling convention, so can't use it for code_ptr
                self.clear_temp_registers();
                let tag_reg = self.allocate_temp_register(); // x11 from pool
                let closure_ptr_reg = self.allocate_temp_register(); // x10 from pool
                let code_ptr_reg = 15; // x15 - caller-saved, safe for code pointer

                // Extract tag (fn_val & 0b111)
                self.emit_and_imm(tag_reg, fn_reg, 0b111);

                // Check if Function (0b100), Closure (0b101), or IFn (other)
                self.emit_cmp_imm(tag_reg, 0b100);

                let is_function_label = self.new_label();
                self.emit_branch_cond(is_function_label.clone(), 0); // 0 = EQ (if tag == 0b100)

                // Not a function, check if closure (0b101)
                self.emit_cmp_imm(tag_reg, 0b101);
                let is_closure_label = self.new_label();
                self.emit_branch_cond(is_closure_label.clone(), 0); // 0 = EQ (if tag == 0b101)

                // === IFn path (tag != 0b100 and != 0b101) ===
                // Call builtin_ifn_invoke(obj, arg_count, arg0, arg1, ...)
                let done_label = self.new_label();
                {
                    let arg_count = args.len();

                    // Set up args for builtin_ifn_invoke:
                    // x0 = obj (fn_reg), x1 = arg_count, x2-x8 = user args
                    self.emit_mov(0, fn_reg); // x0 = obj to invoke
                    self.emit_mov_imm(1, arg_count as i64); // x1 = arg count

                    // Move user args to x2-x8
                    // First collect all source registers to avoid overwrites
                    for (i, &src_reg) in arg_source_regs.iter().enumerate() {
                        if i + 2 != src_reg {
                            // Only move if source != destination
                            self.emit_mov(i + 2, src_reg); // x2, x3, x4, etc.
                        }
                    }

                    // Pad remaining args with 0 (nil) - trampoline expects 7 args
                    for i in arg_count..7 {
                        self.emit_mov_imm(i + 2, 7); // 7 = nil tagged value
                    }

                    // Call builtin_ifn_invoke
                    let ifn_invoke_addr = crate::trampoline::builtin_ifn_invoke as usize;
                    self.emit_external_call(ifn_invoke_addr, "builtin_ifn_invoke");
                }

                // Result is in x0, jump to done
                self.emit_jump(&done_label);

                // === Closure path (tag == 0b101) ===
                self.emit_label(is_closure_label);

                // Untag closure pointer (shift right by 3)
                // LSR Xd, Xn, #3 - Logical shift right by 3 = UBFM Xd, Xn, #3, #63
                let lsr_instruction =
                    0xD343FC00u32 | ((fn_reg as u32) << 5) | (closure_ptr_reg as u32);
                self.code.push(lsr_instruction);

                // Load type_id from header (byte 7) to distinguish single vs multi-arity
                // Header layout: | flags(1) | pad(1) | size(1) | type_data(4) | type_id(1) |
                let type_id_reg = tag_reg; // reuse tag_reg since we're done with it
                self.emit_ldrb_offset(type_id_reg, closure_ptr_reg, 7);

                // Check if multi-arity (type_id == TYPE_MULTI_ARITY_FN)
                use crate::gc_runtime::TYPE_MULTI_ARITY_FN;
                self.emit_cmp_imm(type_id_reg, TYPE_MULTI_ARITY_FN as i64);

                let is_multi_arity_label = self.new_label();
                self.emit_branch_cond(is_multi_arity_label.clone(), 0); // 0 = EQ

                // === Single-arity closure path ===
                use crate::gc_runtime::closure_layout;
                self.emit_ldr_offset(
                    code_ptr_reg,
                    closure_ptr_reg,
                    closure_layout::FIELD_1_CODE_PTR as i32,
                );

                // Set up closure calling convention: x0 = closure object, user args in x1-x7
                self.emit_mov(0, fn_reg); // x0 = tagged closure pointer
                for (i, &src_reg) in arg_source_regs.iter().enumerate() {
                    if i + 1 != src_reg {
                        // Only move if source != destination
                        self.emit_mov(i + 1, src_reg); // x1, x2, x3, etc.
                    }
                }

                let do_blr_label = self.new_label();
                self.emit_jump(&do_blr_label);

                // === Multi-arity closure path ===
                self.emit_label(is_multi_arity_label);
                // Use builtin_invoke_multi_arity which handles variadic
                // New calling convention: x0 = fn_ptr, x1 = arg_count, x2 = args_ptr
                // All args are pushed to stack and pointer passed in x2
                //
                // Save fn_reg to x16 (NOT x15, because emit_external_call uses x15)
                let saved_fn_reg = 16;
                self.emit_mov(saved_fn_reg, fn_reg);

                // Push ALL args to stack (in reverse order so args[0] is at lowest address)
                let arg_count = args.len();
                let stack_space = ((arg_count * 8) + 15) / 16 * 16; // 16-byte aligned
                if stack_space > 0 {
                    self.emit_sub_sp_imm(stack_space as i64);
                    for (i, &src_reg) in arg_source_regs.iter().enumerate() {
                        self.emit_str_offset(src_reg, 31, (i * 8) as i32);
                    }
                }

                // Set up args for builtin_invoke_multi_arity:
                // x0 = fn_ptr, x1 = arg_count, x2 = args_ptr (SP)
                self.emit_mov(0, saved_fn_reg);
                self.emit_mov_imm(1, arg_count as i64);
                if stack_space > 0 {
                    self.emit_mov(2, 31); // x2 = SP (points to args array)
                } else {
                    self.emit_mov_imm(2, 0); // x2 = null (no args)
                }

                let invoke_addr = crate::trampoline::builtin_invoke_multi_arity as usize;
                self.emit_external_call(invoke_addr, "builtin_invoke_multi_arity");

                // Clean up stack
                if stack_space > 0 {
                    self.emit_add_sp_imm(stack_space as i64);
                }

                // Result in x0, jump to done
                self.emit_jump(&done_label);

                // === Function path (tag == 0b100) ===
                self.emit_label(is_function_label);

                // Untag function pointer to get code_ptr (shift right by 3)
                let lsr_instruction =
                    0xD343FC00u32 | ((fn_reg as u32) << 5) | (code_ptr_reg as u32);
                self.code.push(lsr_instruction);

                // Set up normal calling convention: args in x0-x7
                for (i, &src_reg) in arg_source_regs.iter().enumerate() {
                    if i != src_reg {
                        // Only move if source != destination
                        self.emit_mov(i, src_reg); // x0, x1, x2, etc.
                    }
                }

                // === Call the function (for function and closure paths) ===
                self.emit_label(do_blr_label);
                self.emit_blr(code_ptr_reg);

                // === Done (result in x0) ===
                self.emit_label(done_label);

                // Result is in x0
                // eprintln!("DEBUG: Call - result will be moved from x0 to x{}", dst_reg);
                if dst_reg != 0 {
                    self.emit_mov(dst_reg, 0);
                }

                self.store_spill(dst_reg, dest_spill);
                // eprintln!("DEBUG: Call - done, dst_reg=x{}, spill={:?}", dst_reg, dest_spill);
            }

            // NOTE: CallWithSaves is handled later in the match with multi-arity support

            // NOTE: Println has been refactored out - now uses ExternalCall to builtin_println_regs
            Instruction::PushExceptionHandler(catch_label, exception_slot) => {
                // PushExceptionHandler: Setup exception handler
                // Call builtin_push_exception_handler(handler_addr, result_local, LR, SP, FP)

                // Compute FP-relative offset for the exception slot
                // Exception slots come AFTER spill slots in the stack frame:
                //   FP - 8:  spill slot 0
                //   FP - 16: spill slot 1
                //   ...
                //   FP - (num_spill_slots * 8):     last spill slot
                //   FP - ((num_spill_slots + 1) * 8): exception slot 0
                //   FP - ((num_spill_slots + 2) * 8): exception slot 1
                //   ...
                let slot_offset = self.num_spill_slots + exception_slot + 1;
                let result_local_offset = -((slot_offset as i64) * 8);

                // Save SP, FP, LR to temp registers
                self.emit_mov_sp(9, 31); // x9 = SP
                self.emit_mov(10, 29); // x10 = FP
                self.emit_mov(11, 30); // x11 = LR

                // Get label address using ADR (into x0)
                let adr_offset = self.code.len();
                // ADR x0, label - will be patched later
                self.code.push(0x10000000);
                self.pending_adr_fixups
                    .push((adr_offset, catch_label.clone()));

                // x0 = handler address (will be filled by ADR fixup)
                // x1 = result local offset (negative offset from FP)
                self.emit_mov_imm(1, result_local_offset);
                // x2 = LR
                self.emit_mov(2, 11);
                // x3 = SP
                self.emit_mov(3, 9);
                // x4 = FP
                self.emit_mov(4, 10);

                // Call builtin_push_exception_handler
                let func_addr = crate::trampoline::builtin_push_exception_handler as usize;
                self.emit_mov_imm(15, func_addr as i64);
                self.emit_blr(15);
            }

            Instruction::PopExceptionHandler => {
                // PopExceptionHandler: Remove exception handler (normal exit from try)
                let func_addr = crate::trampoline::builtin_pop_exception_handler as usize;
                self.emit_mov_imm(15, func_addr as i64);
                self.emit_blr(15);
                // Result in x0 is nil, we ignore it
            }

            Instruction::Throw(exc) => {
                // Throw: Throw exception, never returns
                // Call builtin_throw(SP, exception_value)
                let exc_reg = self.get_physical_reg_for_irvalue(exc, false)?;

                // IMPORTANT: Move exception value to x1 FIRST, before clobbering x0 with SP
                // Otherwise if exc is in x0, we would clobber it with SP
                if exc_reg != 1 {
                    self.emit_mov(1, exc_reg);
                }
                // x0 = SP (for potential stack trace) - use ADD x0, sp, #0
                self.emit_mov_sp(0, 31);

                // Call builtin_throw (never returns)
                let func_addr = crate::trampoline::builtin_throw as usize;
                self.emit_mov_imm(15, func_addr as i64);
                self.emit_blr(15);
                // Never returns, but ARM64 codegen continues
            }

            Instruction::LoadExceptionLocal(dest, exception_slot) => {
                // Load exception value from stack (where throw stored it)
                // Compute offset using the same formula as PushExceptionHandler
                let slot_offset = self.num_spill_slots + *exception_slot + 1;
                let offset = -((slot_offset as i32) * 8);

                let dest_reg = self.get_physical_reg_for_irvalue(dest, true)?;

                // LDR dest, [x29, #offset]  (load from FP + offset)
                self.emit_load_from_fp(dest_reg, offset);

                // Store to spill location if this is a spilled register
                let spill_offset = self.dest_spill(dest);
                self.store_spill(dest_reg, spill_offset);
            }

            // ========== Local Variable Instructions ==========
            // Following Beagle's pattern: arguments are stored to FP-relative slots at function entry
            Instruction::StoreLocal(slot, value) => {
                // Store value to local slot on stack (FP-relative, negative offset)
                // Local slots come AFTER spill slots and exception slots
                // Layout: [FP] [spills] [exceptions] [locals]
                let slot_offset = self.num_spill_slots + self.reserved_exception_slots + slot + 1;
                let offset = -((slot_offset as i32) * 8);

                // Get the source register (may be an Argument register like x0)
                let src_reg = self.get_physical_reg_for_irvalue(value, false)?;

                // Use emit_store_to_fp which handles negative offsets with STUR
                self.emit_store_to_fp(src_reg, offset);
            }

            Instruction::LoadLocal(dest, slot) => {
                // Load from local slot to destination register
                // Same layout as StoreLocal
                let slot_offset = self.num_spill_slots + self.reserved_exception_slots + slot + 1;
                let offset = -((slot_offset as i32) * 8);

                let dest_spill = self.dest_spill(dest);
                let dest_reg = self.get_physical_reg_for_irvalue(dest, true)?;

                // LDR dest, [x29, #offset]  (load from FP + offset)
                self.emit_load_from_fp(dest_reg, offset);

                // Store to spill location if this is a spilled register
                self.store_spill(dest_reg, dest_spill);
            }

            Instruction::Breakpoint => {
                // BRK #0 - trap for debugger
                self.code.push(arm::brk(0));
            }

            Instruction::Ret(value) => {
                // Move result to x0 (return register)
                // Epilogue is emitted separately at end of compile_function
                match value {
                    IrValue::Register(_) => {
                        let src_reg = self.get_physical_reg_for_irvalue(value, false)?;
                        if src_reg != 0 {
                            self.emit_mov(0, src_reg);
                        }
                    }
                    IrValue::TaggedConstant(val) => {
                        // Direct constant load to return register
                        self.emit_mov_imm(0, *val as i64);
                    }
                    IrValue::Null => {
                        // Null is tagged value 7 (nil tag)
                        self.emit_mov_imm(0, 7);
                    }
                    IrValue::RawConstant(val) => {
                        // Raw constant (untagged)
                        self.emit_mov_imm(0, *val);
                    }
                    _ => {
                        // For spills and other values, use generic handler
                        let src_reg = self.get_physical_reg_for_irvalue(value, false)?;
                        if src_reg != 0 {
                            self.emit_mov(0, src_reg);
                        }
                    }
                }
                // Note: Epilogue (add sp, ldp, ret) is emitted at end of compile_function
            }

            Instruction::CallWithSaves(dst, target, args, saves) => {
                // Unified call with register preservation
                // Uses FBP-relative push_to_stack/pop_from_stack (like beagle)

                // STEP 1: Save registers using FBP-relative addressing
                // Note: saves are always Registers (never Spills) - see linear_scan.rs line 1205
                for save in saves.iter() {
                    let save_reg = self.get_physical_reg_for_irvalue(save, false)?;
                    self.push_to_stack(save_reg);
                    self.clear_temp_registers();
                }

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                self.clear_temp_registers();

                // STEP 2: Dispatch based on CallTarget
                match target {
                    CallTarget::External(func_addr) => {
                        // External call to known function address (trampoline)
                        if args.len() > 8 {
                            return Err(
                                "External call with more than 8 arguments not yet supported"
                                    .to_string(),
                            );
                        }

                        // Set up args in x0-x7
                        for (i, arg) in args.iter().enumerate() {
                            match arg {
                                IrValue::RawConstant(val) => {
                                    self.emit_mov_imm(i, *val);
                                }
                                _ => {
                                    let arg_reg = self.get_physical_reg_for_irvalue(arg, false)?;
                                    if i != arg_reg {
                                        self.emit_mov(i, arg_reg);
                                    }
                                    self.clear_temp_registers();
                                }
                            }
                        }

                        // Call the external function
                        self.emit_external_call(*func_addr, "external_call");

                        // Move result from x0 to destination
                        if dst_reg != 0 {
                            self.emit_mov(dst_reg, 0);
                        }
                        self.store_spill(dst_reg, dest_spill);
                    }

                    CallTarget::Dynamic(fn_val) => {
                        // Dynamic dispatch - must check tag at runtime
                        // Supports:
                        //   - Raw function pointers (tag 0b100)
                        //   - Single-arity closures (tag 0b101, type_id = TYPE_FUNCTION)
                        //   - Multi-arity functions (tag 0b101, type_id = TYPE_MULTI_ARITY_FN)
                        //   - More than 7 user args (for closures) or 8 args (for raw functions) via stack

                        // Calculate stack args needed for closures (7 user args in registers, rest on stack)
                        // For raw functions: 8 args in registers
                        // We'll use the closure limit (7) since that's the more common case
                        let max_reg_user_args = 7usize;
                        let stack_arg_count = if args.len() > max_reg_user_args {
                            args.len() - max_reg_user_args
                        } else {
                            0
                        };
                        let stack_args_bytes = (stack_arg_count * 8).div_ceil(16) * 16; // 16-byte aligned

                        // Get function pointer - may be in a temp register if spilled
                        // IMPORTANT: If fn_val is spilled, we get a temp register (x9).
                        // We MUST save this to a dedicated register before loading args,
                        // because args loading might also use x9 and clobber our value.
                        let fn_temp = self.get_physical_reg_for_irvalue(fn_val, false)?;
                        let fn_reg = 16; // x16 - dedicated register for fn pointer
                        if fn_temp != fn_reg {
                            self.emit_mov(fn_reg, fn_temp);
                        }
                        self.clear_temp_registers();

                        // DON'T load args here - they'll be loaded directly into x1, x2, etc.
                        // later in the single-arity path. Loading into temps now causes them
                        // to be clobbered by tag checking operations that also use temps.
                        let arg_count = args.len();

                        // STEP 2a: Allocate stack space for extra arguments (if any)
                        // For closures: x1-x7 = first 7 user args, stack = args 7+
                        // For raw functions: x0-x7 = first 8 args, stack = args 8+
                        // We allocate for closures (the more restrictive case)
                        if stack_args_bytes > 0 {
                            self.emit_sub_sp_imm(stack_args_bytes as i64);
                            self.current_function_stack_bytes += stack_args_bytes;
                            self.increment_stack_size(stack_args_bytes / 8);

                            // Store extra args to stack
                            self.clear_temp_registers();
                            for i in 0..stack_arg_count {
                                let arg_idx = max_reg_user_args + i;
                                let arg = &args[arg_idx];
                                let src_reg = match arg {
                                    IrValue::Spill(_, slot) => {
                                        let temp = self.allocate_temp_register();
                                        // Adjust offset for the new stack space we just allocated
                                        let offset = -((*slot as i32 + 1) * 8);
                                        self.emit_load_from_fp(temp, offset);
                                        temp
                                    }
                                    _ => self.get_physical_reg_for_irvalue(arg, false)?,
                                };
                                // Store to stack at SP + i * 8
                                self.emit_str_offset(src_reg, 31, (i * 8) as i32);
                                self.clear_temp_registers();
                            }
                        }

                        // Use temp registers from pool for tag/closure_ptr, but NOT for code_ptr!
                        // x18 is RESERVED by macOS and must NEVER be used!
                        // x9 is used for arg_count in closure calling convention, so can't use it for code_ptr
                        self.clear_temp_registers();
                        let tag_reg = self.allocate_temp_register();
                        let closure_ptr_reg = self.allocate_temp_register();
                        let code_ptr_reg = 15; // x15 - caller-saved, safe for code pointer

                        // Extract tag (fn_val & 0b111)
                        self.emit_and_imm(tag_reg, fn_reg, 0b111);

                        // Check if Function (0b100), Closure (0b101), or IFn (other)
                        self.emit_cmp_imm(tag_reg, 0b100);

                        let is_raw_function_label = self.new_label();
                        self.emit_branch_cond(is_raw_function_label.clone(), 0); // 0 = EQ (if tag == 0b100)

                        // Not a function, check if closure (0b101)
                        self.emit_cmp_imm(tag_reg, 0b101);
                        let is_closure_label = self.new_label();
                        self.emit_branch_cond(is_closure_label.clone(), 0); // 0 = EQ (if tag == 0b101)

                        // === IFn path (tag != 0b100 and != 0b101) ===
                        // Call builtin_ifn_invoke(obj, arg_count, arg0, arg1, ...)
                        let done_label = self.new_label();
                        {
                            // Set up args for builtin_ifn_invoke:
                            // x0 = obj (fn_reg), x1 = arg_count, x2-x8 = user args
                            self.emit_mov(0, fn_reg); // x0 = obj to invoke
                            self.emit_mov_imm(1, arg_count as i64); // x1 = arg count

                            // Load user args directly into x2-x8
                            self.clear_temp_registers();
                            for (i, arg) in args.iter().enumerate() {
                                let target_reg = i + 2; // x2, x3, x4, etc.
                                match arg {
                                    IrValue::Spill(_, slot) => {
                                        let offset = -((*slot as i32 + 1) * 8);
                                        self.emit_load_from_fp(target_reg, offset);
                                    }
                                    _ => {
                                        let src_reg =
                                            self.get_physical_reg_for_irvalue(arg, false)?;
                                        if target_reg != src_reg {
                                            self.emit_mov(target_reg, src_reg);
                                        }
                                        self.clear_temp_registers();
                                    }
                                }
                            }

                            // Pad remaining args with nil (7) - trampoline expects 7 args
                            for i in arg_count..7 {
                                self.emit_mov_imm(i + 2, 7); // 7 = nil tagged value
                            }

                            // Call builtin_ifn_invoke
                            let ifn_invoke_addr = crate::trampoline::builtin_ifn_invoke as usize;
                            self.emit_external_call(ifn_invoke_addr, "builtin_ifn_invoke");
                        }

                        // Result is in x0, jump to done
                        self.emit_jump(&done_label);

                        // === Closure path (tag == 0b101) ===
                        self.emit_label(is_closure_label);
                        // Could be single-arity (TYPE_ID_FUNCTION=12) or multi-arity (TYPE_ID_MULTI_ARITY_FN=14)

                        // Untag closure pointer (shift right by 3)
                        let lsr_instruction =
                            0xD343FC00u32 | ((fn_reg as u32) << 5) | (closure_ptr_reg as u32);
                        self.code.push(lsr_instruction);

                        // Load type_id from header (byte 7 of header)
                        // Header layout: | flags(1) | pad(1) | size(1) | type_data(4) | type_id(1) |
                        // So type_id is at byte offset 7
                        // Use LDRB to load just the type_id byte - reuse tag_reg since we're done with it
                        let type_id_reg = tag_reg;
                        self.emit_ldrb_offset(type_id_reg, closure_ptr_reg, 7);

                        // Check if multi-arity (type_id == TYPE_MULTI_ARITY_FN)
                        use crate::gc_runtime::TYPE_MULTI_ARITY_FN;
                        self.emit_cmp_imm(type_id_reg, TYPE_MULTI_ARITY_FN as i64);

                        let is_multi_arity_label = self.new_label();
                        self.emit_branch_cond(is_multi_arity_label.clone(), 0); // 0 = EQ

                        // === Single-arity closure path (type_id == TYPE_FUNCTION) ===
                        // Single-arity closures are never variadic (variadic functions become
                        // multi-arity even with one arity). No x9 arg count needed.
                        use crate::gc_runtime::closure_layout;
                        // code_ptr_reg already allocated from temp pool above
                        self.emit_ldr_offset(
                            code_ptr_reg,
                            closure_ptr_reg,
                            closure_layout::FIELD_1_CODE_PTR as i32,
                        );

                        // Set up closure calling convention:
                        // - x0 = closure object
                        // - x1-x7 = first 7 user args (rest are on stack, already set up)
                        self.emit_mov(0, fn_reg); // x0 = tagged closure pointer

                        // Load args directly into x1, x2, etc.
                        // Only load up to max_reg_user_args (7) - rest are already on stack
                        // This is done HERE (after tag checking) to avoid temp register clobbering
                        self.clear_temp_registers();
                        let reg_args_to_load = std::cmp::min(args.len(), max_reg_user_args);
                        for (i, arg) in args.iter().take(reg_args_to_load).enumerate() {
                            let target_reg = i + 1; // x1, x2, x3, etc.
                            match arg {
                                IrValue::Spill(_, slot) => {
                                    // Load directly into target register from spill slot
                                    let offset = -((*slot as i32 + 1) * 8);
                                    self.emit_load_from_fp(target_reg, offset);
                                }
                                _ => {
                                    let src_reg = self.get_physical_reg_for_irvalue(arg, false)?;
                                    if target_reg != src_reg {
                                        self.emit_mov(target_reg, src_reg);
                                    }
                                    self.clear_temp_registers();
                                }
                            }
                        }

                        let do_call_label = self.new_label();
                        self.emit_jump(&do_call_label);

                        // === Multi-arity function path (type_id == TYPE_MULTI_ARITY_FN) ===
                        self.emit_label(is_multi_arity_label);

                        // Use builtin_invoke_multi_arity which handles:
                        // 1. Looking up the correct arity
                        // 2. Building IndexedSeq for variadic if needed
                        // 3. Calling the function with proper convention
                        //
                        // New calling convention: x0 = fn_ptr, x1 = arg_count, x2 = args_ptr
                        // All args are pushed to stack and pointer passed in x2

                        // Push ALL args to stack (so args[0] is at lowest address)
                        // Note: We need to allocate additional stack space beyond what
                        // was allocated for stack_args (if any)
                        let multi_arity_stack_space = ((arg_count * 8) + 15) / 16 * 16;
                        if multi_arity_stack_space > 0 {
                            self.emit_sub_sp_imm(multi_arity_stack_space as i64);

                            // Load and store each arg
                            self.clear_temp_registers();
                            for (i, arg) in args.iter().enumerate() {
                                match arg {
                                    IrValue::Spill(_, slot) => {
                                        let temp = self.allocate_temp_register();
                                        // Adjust offset for the new stack space we allocated
                                        let offset = -((*slot as i32 + 1) * 8);
                                        self.emit_load_from_fp(temp, offset);
                                        self.emit_str_offset(temp, 31, (i * 8) as i32);
                                        self.clear_temp_registers();
                                    }
                                    _ => {
                                        let src_reg = self.get_physical_reg_for_irvalue(arg, false)?;
                                        self.emit_str_offset(src_reg, 31, (i * 8) as i32);
                                        self.clear_temp_registers();
                                    }
                                }
                            }
                        }

                        // Set up args: x0 = fn_ptr, x1 = arg_count, x2 = args_ptr
                        self.emit_mov(0, fn_reg);
                        self.emit_mov_imm(1, arg_count as i64);
                        if multi_arity_stack_space > 0 {
                            self.emit_mov(2, 31); // x2 = SP (points to args array)
                        } else {
                            self.emit_mov_imm(2, 0); // x2 = null (no args)
                        }

                        let invoke_addr = crate::trampoline::builtin_invoke_multi_arity as usize;
                        self.emit_external_call(invoke_addr, "builtin_invoke_multi_arity");

                        // Clean up stack
                        if multi_arity_stack_space > 0 {
                            self.emit_add_sp_imm(multi_arity_stack_space as i64);
                        }

                        // Result is in x0, jump to done (skip the do_call_label BLR)
                        self.emit_jump(&done_label);

                        // === Raw function path (tag == 0b100) ===
                        self.emit_label(is_raw_function_label);

                        // Untag function pointer to get code_ptr (shift right by 3)
                        let lsr_instruction =
                            0xD343FC00u32 | ((fn_reg as u32) << 5) | (code_ptr_reg as u32);
                        self.code.push(lsr_instruction);

                        // Set up normal calling convention: args in x0-x7 (first 8 only)
                        // Note: For raw functions, 8 args fit in registers. If there are more,
                        // they are already on stack from the closure path's allocation.
                        // But raw functions rarely have >8 args, so we just load min(args, 8).
                        self.clear_temp_registers();
                        let raw_fn_reg_args = std::cmp::min(args.len(), 8);
                        for (target_reg, arg) in args.iter().take(raw_fn_reg_args).enumerate() {
                            // target_reg is x0, x1, x2, etc.
                            match arg {
                                IrValue::Spill(_, slot) => {
                                    let offset = -((*slot as i32 + 1) * 8);
                                    self.emit_load_from_fp(target_reg, offset);
                                }
                                _ => {
                                    let src_reg = self.get_physical_reg_for_irvalue(arg, false)?;
                                    if target_reg != src_reg {
                                        self.emit_mov(target_reg, src_reg);
                                    }
                                    self.clear_temp_registers();
                                }
                            }
                        }

                        // === Call the function ===
                        self.emit_label(do_call_label);
                        self.emit_blr(code_ptr_reg);

                        // === Done (result in x0) ===
                        self.emit_label(done_label);

                        // Move result from x0 to destination
                        if dst_reg != 0 {
                            self.emit_mov(dst_reg, 0);
                        }

                        self.store_spill(dst_reg, dest_spill);

                        // Clean up stack space for extra arguments (if any were allocated)
                        if stack_args_bytes > 0 {
                            self.emit_add_sp_imm(stack_args_bytes as i64);
                            self.decrement_stack_size(stack_args_bytes / 8);
                        }
                    } // end CallTarget::Dynamic
                } // end match target

                // STEP 3: Restore registers (reverse order)
                for save in saves.iter().rev() {
                    let save_reg = self.get_physical_reg_for_irvalue(save, true)?;
                    self.pop_from_stack(save_reg);
                    self.clear_temp_registers();
                }
            }

            // NOTE: MakeType, MakeTypeWithSaves, LoadTypeField, and StoreTypeField have been refactored out.
            // - Deftype construction uses ExternalCall + HeapStore + Tag
            // - Field access uses ExternalCall to builtin_load_type_field_by_symbol with pre-interned symbol IDs

            Instruction::ExternalCall(_, _, _) => {
                // ExternalCall should be transformed to CallWithSaves by register allocator
                panic!("ExternalCall should have been transformed to CallWithSaves");
            }

            // Multi-arity function instructions
            Instruction::MakeMultiArityFn(dst, arities, variadic_min, variadic_index, values_ptr, closure_count) => {
                // Create a multi-arity function object on the heap
                // Similar to MakeFunctionPtr but stores multiple (param_count, code_ptr) pairs
                //
                // ARM64 Calling Convention for builtin_allocate_multi_arity_fn:
                // - x0 = frame_pointer (x29 for GC)
                // - x1 = name_ptr (0 for anonymous)
                // - x2 = arity_count
                // - x3 = arities_ptr (pointer to (param_count, code_ptr) pairs on stack)
                // - x4 = variadic_min (usize::MAX if no variadic)
                // - x5 = variadic_index (usize::MAX if no variadic)
                // - x6 = closure_count
                // - x7 = closures_ptr (pointer to closure values on stack)
                // - Returns: x0 = tagged closure pointer
                // gc_return_addr is computed internally by the builtin
                //
                // Closure values already pushed via PushToStack instructions
                // values_ptr points to FP-relative closure values (from CurrentStackPosition)
                // Arity table uses SP-relative slots (not heap pointers, no GC needed)

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                // Get closures_ptr from values_ptr (already computed by compiler)
                let closures_ptr_reg = if *closure_count > 0 {
                    self.get_physical_reg_for_irvalue(values_ptr, false)?
                } else {
                    10 // Will be set to 0 below
                };

                // Allocate SP space for arity table only (no stack arg needed now)
                // Arity table contains integers (param_count, code_ptr) - not heap pointers
                let arities_size = arities.len() * 2 * 8; // 2 words per arity
                let aligned_sp_space = arities_size.div_ceil(16) * 16;
                if aligned_sp_space > 0 {
                    self.emit_sub_sp_imm(aligned_sp_space as i64);
                }

                // Store arity table to SP-relative stack
                for (i, (param_count, code_ptr)) in arities.iter().enumerate() {
                    let offset = i * 16; // 2 words per entry
                    // Store param_count
                    let temp_reg = 11;
                    self.emit_mov_imm(temp_reg, *param_count as i64);
                    self.emit_str_offset(temp_reg, 31, offset as i32);
                    // Store code_ptr
                    self.emit_mov_imm(temp_reg, *code_ptr as i64);
                    self.emit_str_offset(temp_reg, 31, (offset + 8) as i32);
                }

                // Set closures_ptr to 0 if no closures
                if *closure_count == 0 {
                    self.emit_mov_imm(closures_ptr_reg, 0);
                }

                // Save arities_ptr for argument setup
                let arities_ptr_reg = 12;
                self.emit_mov(arities_ptr_reg, 31); // arities_ptr = SP (arity table starts at SP+0)

                // Set up arguments for trampoline call
                // gc_return_addr is computed internally by the builtin
                self.emit_mov(0, 29); // x0 = FP for GC
                self.emit_mov_imm(1, 0); // x1 = name_ptr (0 for anonymous)
                self.emit_mov_imm(2, arities.len() as i64); // x2 = arity_count
                self.emit_mov(3, arities_ptr_reg); // x3 = arities_ptr
                self.emit_mov_imm(4, variadic_min.unwrap_or(usize::MAX) as i64); // x4 = variadic_min
                self.emit_mov_imm(5, variadic_index.unwrap_or(usize::MAX) as i64); // x5 = variadic_index
                self.emit_mov_imm(6, *closure_count as i64); // x6 = closure_count
                self.emit_mov(7, closures_ptr_reg); // x7 = closures_ptr

                // Call trampoline to allocate multi-arity function
                // emit_external_call -> emit_blr -> update_stack_map()
                // Stack map includes the pushed closure values!
                let func_addr = crate::trampoline::builtin_allocate_multi_arity_fn as usize;
                self.emit_external_call(func_addr, "builtin_allocate_multi_arity_fn");

                // Clean up SP stack
                if aligned_sp_space > 0 {
                    self.emit_add_sp_imm(aligned_sp_space as i64);
                }

                // Result is in x0 (tagged closure pointer)
                if dst_reg != 0 {
                    self.emit_mov(dst_reg, 0);
                }

                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadClosureMultiArity(dst, fn_obj, arity_count, index) => {
                // Load closure value from a multi-arity function
                // Layout differs from single-arity: need to skip arity table
                //
                // Multi-arity layout (after header):
                //   field 0: name_ptr
                //   field 1: arity_count
                //   field 2: variadic_min
                //   field 3: closure_count
                //   fields 4..(4 + arity_count*2): arity table
                //   fields (4 + arity_count*2)...: closure values

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let fn_reg = self.get_physical_reg_for_irvalue(fn_obj, false)?;

                use crate::gc_runtime::multi_arity_layout;

                // Untag closure pointer (shift right by 3)
                // Use temp register from pool instead of hardcoded register
                self.clear_temp_registers();
                let ptr_reg = self.allocate_temp_register();
                let lsr_instruction = 0xD343FC00u32 | ((fn_reg as u32) << 5) | (ptr_reg as u32);
                self.code.push(lsr_instruction);

                // Calculate offset to closure value
                let closure_field = multi_arity_layout::closure_value_field(*arity_count, *index);
                let offset = (closure_field + 1) * 8; // +1 for header

                // Load closure value
                self.emit_ldr_offset(dst_reg, ptr_reg, offset as i32);

                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::AssertPre(condition, index) => {
                // Check if condition is truthy; if falsy, call builtin_pre_condition_failed
                //
                // In Clojure, false and nil are falsy, everything else is truthy.
                // nil = 7, false = 3
                //
                // We need to check if condition is NOT falsy:
                // - condition != 7 (nil)
                // - condition != 3 (false)

                let cond_reg = self.get_physical_reg_for_irvalue(condition, false)?;

                let pass_label = self.new_label();

                // Check if condition == nil (7)
                self.emit_cmp_imm(cond_reg, 7);
                let not_nil_label = self.new_label();
                // B.NE not_nil (branch if not equal)
                self.emit_branch_cond(not_nil_label.clone(), 1); // NE = 1
                // If we're here, condition is nil - fail assertion
                self.emit_mov_imm(0, 0); // x0 = 0 (stack_pointer, unused)
                self.emit_mov_imm(1, *index as i64); // x1 = condition index
                let fail_addr = crate::trampoline::builtin_pre_condition_failed as usize;
                self.emit_external_call(fail_addr, "builtin_pre_condition_failed");
                // builtin_pre_condition_failed never returns

                self.emit_label(not_nil_label);

                // Check if condition == false (3)
                self.emit_cmp_imm(cond_reg, 3);
                // B.NE pass (branch if not equal)
                self.emit_branch_cond(pass_label.clone(), 1); // NE = 1
                // If we're here, condition is false - fail assertion
                self.emit_mov_imm(0, 0); // x0 = 0 (stack_pointer, unused)
                self.emit_mov_imm(1, *index as i64); // x1 = condition index
                self.emit_external_call(fail_addr, "builtin_pre_condition_failed");
                // builtin_pre_condition_failed never returns

                self.emit_label(pass_label);
                // Condition passed - continue execution
            }

            Instruction::AssertPost(condition, index) => {
                // Same logic as AssertPre but calls builtin_post_condition_failed

                let cond_reg = self.get_physical_reg_for_irvalue(condition, false)?;

                let pass_label = self.new_label();

                // Check if condition == nil (7)
                self.emit_cmp_imm(cond_reg, 7);
                let not_nil_label = self.new_label();
                // B.NE not_nil (branch if not equal)
                self.emit_branch_cond(not_nil_label.clone(), 1); // NE = 1
                // If we're here, condition is nil - fail assertion
                self.emit_mov_imm(0, 0); // x0 = 0 (stack_pointer, unused)
                self.emit_mov_imm(1, *index as i64); // x1 = condition index
                let fail_addr = crate::trampoline::builtin_post_condition_failed as usize;
                self.emit_external_call(fail_addr, "builtin_post_condition_failed");
                // builtin_post_condition_failed never returns

                self.emit_label(not_nil_label);

                // Check if condition == false (3)
                self.emit_cmp_imm(cond_reg, 3);
                // B.NE pass (branch if not equal)
                self.emit_branch_cond(pass_label.clone(), 1); // NE = 1
                // If we're here, condition is false - fail assertion
                self.emit_mov_imm(0, 0); // x0 = 0 (stack_pointer, unused)
                self.emit_mov_imm(1, *index as i64); // x1 = condition index
                self.emit_external_call(fail_addr, "builtin_post_condition_failed");
                // builtin_post_condition_failed never returns

                self.emit_label(pass_label);
                // Condition passed - continue execution
            }

        }

        // Clear temporary registers after each instruction (like Beagle does)
        self.clear_temp_registers();

        Ok(())
    }

    /// Get physical register for an IR value
    /// If is_dest=true and value is Spill, returns temp register without loading
    /// If is_dest=false and value is Spill, loads from stack into temp register
    fn get_physical_reg_for_irvalue(
        &mut self,
        value: &IrValue,
        is_dest: bool,
    ) -> Result<usize, String> {
        match value {
            IrValue::Register(vreg) => Ok(self.get_physical_reg(vreg)),
            IrValue::Spill(_vreg, stack_offset) => {
                // Allocate a temporary register from the pool
                let temp_reg = self.allocate_temp_register();

                if !is_dest {
                    // Load spilled value from stack
                    // Stack layout after prologue:
                    //   [FP + 8]:  saved x30 (LR)
                    //   [FP + 0]:  saved x29 (old FP) <- x29 points here
                    //   [FP - 8]:  spill slot 0
                    //   [FP - 16]: spill slot 1
                    //   [FP - 24]: spill slot 2
                    //   ...
                    //   [FP - (N+1)*8]: spill slot N
                    let offset = -((*stack_offset as i32 + 1) * 8);
                    self.emit_load_from_fp(temp_reg, offset);
                }
                // For destination, just return temp_reg without loading
                Ok(temp_reg)
            }
            IrValue::FramePointer => {
                // Return FP (x29) for GC stack walking via frame pointer chain traversal.
                Ok(29)
            }
            IrValue::ReturnAddress => {
                // Return LR (x30) for GC stack map lookup.
                Ok(30)
            }
            IrValue::TaggedConstant(_) | IrValue::Null => {
                // TaggedConstant and Null should be converted to registers via assign_new
                // in the compiler BEFORE reaching codegen. If we hit this, it means
                // the compiler is missing an assign_new call.
                // Beagle handles these only in specific instructions (Ret, Assign, JumpIf, etc.),
                // not via a generic "get register for value" function.
                let inst_info = self.current_instruction.as_deref().unwrap_or("unknown");
                Err(format!(
                    "Codegen received {:?} where a register was expected.\n\
                     This value should have been converted to a register via assign_new() in the compiler.\n\
                     Instruction: {}",
                    value, inst_info
                ))
            }
            _ => Err(format!(
                "Expected register, spill, or constant, got {:?}",
                value
            )),
        }
    }

    fn get_physical_reg(&mut self, vreg: &VirtualRegister) -> usize {
        // IMPORTANT: Don't look up Argument registers in the map!
        // The allocator maps Argument(n) -> Temp(n) to represent physical xn,
        // but Temp(n) might also exist as a virtual register with a different allocation.
        // Instead, get Argument registers' physical location directly from their variant.
        match vreg {
            VirtualRegister::Argument(n) => *n, // Arguments are already physical (x0-x7)
            _ => {
                // After register allocation, the IR has been rewritten with physical registers
                // Physical registers are represented as VirtualRegister::Temp(X) where X is the physical register number.
                //
                // If the register is in the allocation map, it's an original virtual register
                // and we need to look up its physical register.
                // If not in the map, it's already a physical register, so just use its index.
                match self.register_map.get(vreg) {
                    Some(physical) => physical.index(),
                    None => {
                        // Check if this is a valid physical register (x19-x28)
                        let idx = vreg.index();
                        if (19..=28).contains(&idx) {
                            idx // Already physical, use index directly
                        } else {
                            panic!("Result register {:?} not allocated", vreg);
                        }
                    }
                }
            }
        }
    }

    /// Check if a destination is a spill and return its stack offset
    fn dest_spill(&self, dest: &IrValue) -> Option<usize> {
        match dest {
            IrValue::Spill(_, stack_offset) => Some(*stack_offset),
            _ => None,
        }
    }

    /// Store a register to its spill location if needed
    fn store_spill(&mut self, src_reg: usize, dest_spill: Option<usize>) {
        if let Some(stack_offset) = dest_spill {
            // Stack layout after prologue:
            //   [FP + 8]:  saved x30 (LR)
            //   [FP + 0]:  saved x29 (old FP) <- x29 points here
            //   [FP - 8]:  spill slot 0
            //   [FP - 16]: spill slot 1
            //   [FP - 24]: spill slot 2
            //   ...
            //   [FP - (N+1)*8]: spill slot N
            //   [FP - stack_space]: SP
            let offset = -((stack_offset as i32 + 1) * 8);
            // eprintln!("DEBUG store_spill: slot {} -> offset {} (x{} to [FP{}])", stack_offset, offset, src_reg, offset);
            if offset >= 0 {
                eprintln!("ERROR: Positive spill offset would corrupt saved registers!");
            }
            self.emit_store_to_fp(src_reg, offset);
        }
    }

    /// Allocate a temporary register for loading spills
    fn allocate_temp_register(&mut self) -> usize {
        self.temp_register_pool
            .pop()
            .expect("Out of temporary registers! Need to clear temps between instructions")
    }

    /// Reset temporary register pool (called after each instruction)
    fn clear_temp_registers(&mut self) {
        self.temp_register_pool = vec![11, 10, 9];
    }

    fn apply_fixups(&mut self) -> Result<(), String> {
        // Apply branch fixups
        for (code_index, label) in &self.pending_fixups {
            let target_pos = self
                .label_positions
                .get(label)
                .ok_or_else(|| format!("Undefined label: {}", label))?;

            // Calculate offset in instructions (not bytes)
            let offset = (*target_pos as isize) - (*code_index as isize);

            // Check if offset fits in the instruction encoding
            if !(-1048576..=1048575).contains(&offset) {
                return Err(format!("Jump offset too large: {}", offset));
            }

            // Patch the instruction
            let instruction = self.code[*code_index];

            // Check if it's a conditional branch (B.cond) or unconditional branch (B)
            if (instruction & 0xFF000000) == 0x54000000 {
                // B.cond - 19-bit signed offset in bits [23:5]
                let offset_bits = (offset as u32) & 0x7FFFF; // 19 bits
                self.code[*code_index] = (instruction & 0xFF00001F) | (offset_bits << 5);
            } else if (instruction & 0xFC000000) == 0x14000000 {
                // B - 26-bit signed offset in bits [25:0]
                let offset_bits = (offset as u32) & 0x03FFFFFF; // 26 bits
                self.code[*code_index] = (instruction & 0xFC000000) | offset_bits;
            } else {
                return Err(format!(
                    "Unknown branch instruction at {}: {:08x}",
                    code_index, instruction
                ));
            }
        }

        // Apply ADR fixups
        for (code_index, label) in &self.pending_adr_fixups {
            let target_pos = self
                .label_positions
                .get(label)
                .ok_or_else(|| format!("Undefined label: {}", label))?;

            // Calculate offset in instructions
            let offset_instructions = (*target_pos as isize) - (*code_index as isize);

            // ADR uses byte offsets, so multiply by 4
            let byte_offset = offset_instructions * 4;

            // eprintln!("DEBUG ADR fixup: code_index={}, label={}, target_pos={}, offset_instructions={}, byte_offset={}",
            //           code_index, label, target_pos, offset_instructions, byte_offset);

            // Check if offset fits in 21-bit signed immediate
            if !(-1048576..=1048575).contains(&byte_offset) {
                return Err(format!("ADR offset too large: {}", byte_offset));
            }

            // ADR encoding: immlo (2 bits) | immhi (19 bits)
            let immlo = (byte_offset & 0x3) as u32; // Lower 2 bits
            let immhi = ((byte_offset >> 2) & 0x7FFFF) as u32; // Upper 19 bits

            // Patch the instruction
            let instruction = self.code[*code_index];
            self.code[*code_index] = (instruction & 0x9F00001F) | (immlo << 29) | (immhi << 5);

            // eprintln!("DEBUG ADR: patched instruction at {} from {:08x} to {:08x}",
            //           code_index, instruction, self.code[*code_index]);
        }

        Ok(())
    }

    // ARM64 instruction encoding

    fn emit_mov(&mut self, dst: usize, src: usize) {
        // eprintln!("DEBUG: emit_mov(x{}, x{})", dst, src);
        // Special handling when either source OR destination is register 31 (SP)
        // Following Beagle's pattern: check both directions
        // ORR treats register 31 as XZR, but we need it as SP
        // Use ADD instruction which properly interprets register 31 as SP
        if dst == 31 || src == 31 {
            self.emit_mov_sp(dst, src);
        } else {
            self.emit_mov_reg(dst, src);
        }
    }

    /// Generate MOV for regular registers (uses ORR)
    /// Based on Beagle's mov_reg pattern
    fn emit_mov_reg(&mut self, dst: usize, src: usize) {
        // MOV is ORR Xd, XZR, Xm
        // This works for normal registers but treats register 31 as XZR
        let instruction = 0xAA0003E0 | ((src as u32) << 16) | (dst as u32);
        self.code.push(instruction);
    }

    /// Generate ORR with immediate (for setting tag bits)
    /// ORR Xd, Xn, #imm
    #[allow(dead_code)]
    fn emit_orr_imm(&mut self, dst: usize, src: usize, imm: u32) {
        // ARM64 logical immediate encoding for small tag values
        // Format: sf(1) opc(01) 100100 N(1) immr(6) imms(6) Rn(5) Rd(5)
        // Base: 0xB2400000 for 64-bit ORR immediate
        let instruction = match imm {
            // 0b100 (4) - Function tag: single bit at position 2
            // immr=0, imms=0 encodes element size 64, pattern of 1 bit at position 2
            0b100 => 0xB2400C00u32 | ((src as u32) << 5) | (dst as u32),

            // 0b110 (6) - HeapObject tag: two bits at positions 1 and 2
            // For 0b110 = binary 110, we need a 2-bit pattern (11) rotated to position 1
            // N=1, immr=63 (rotate right by 1 = rotate left by 63), imms=1 (2-bit pattern)
            // Base encoding: 0xB27F0400 for ORR X, X, #6
            0b110 => 0xB27F0400u32 | ((src as u32) << 5) | (dst as u32),

            // 0b111 (7) - nil tag: three bits at positions 0, 1, 2
            // N=1, immr=0, imms=2 (3-bit pattern)
            0b111 => 0xB2400800u32 | ((src as u32) << 5) | (dst as u32),

            _ => panic!("ORR immediate not implemented for {:#b} ({})", imm, imm),
        };
        self.code.push(instruction);
    }

    /// Generate AND with immediate (for extracting tag bits)
    /// AND Xd, Xn, #imm
    fn emit_and_imm(&mut self, dst: usize, src: usize, imm: u32) {
        // AND Xd, Xn, #0b111 to extract the last 3 bits (tag)
        // ARM64 logical immediate encoding for 0b111 (3 ones)
        // Format: sf(1) opc(00) 100100 N(1) immr(6) imms(6) Rn(5) Rd(5)
        // For 64-bit AND with 3 consecutive ones: sf=1, N=1, immr=0, imms=2
        // imms=2 means (2+1)=3 consecutive ones = 0b111
        let instruction = if imm == 0b111 {
            // AND X, X, #0b111 - base encoding 0x92400800
            0x92400800u32 | ((src as u32) << 5) | (dst as u32)
        } else {
            panic!("AND immediate only implemented for 0b111");
        };
        self.code.push(instruction);
    }

    /// Generate LDR with offset (load from memory)
    /// LDR Xt, [Xn, #offset]
    fn emit_ldr_offset(&mut self, dst: usize, base: usize, offset: i32) {
        // LDR Xt, [Xn, #offset]
        // Offset is in bytes, needs to be divided by 8 for encoding (unsigned 12-bit)
        let offset_scaled = (offset / 8) as u32;
        let instruction = 0xF9400000 | (offset_scaled << 10) | ((base as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    /// Load with register offset: LDR Xt, [Xn, Xm]
    #[allow(dead_code)]
    fn emit_ldr_reg_offset(&mut self, dst: usize, base: usize, offset_reg: usize) {
        // LDR Xt, [Xn, Xm] - Load with register offset
        // Encoding: 11 111000 01 1 Rm 011 0 10 Rn Rt
        // = F8606800 | (Rm << 16) | (Rn << 5) | Rt
        let instruction =
            0xF8606800 | ((offset_reg as u32) << 16) | ((base as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    /// Generate MOV involving SP (uses ADD with immediate 0)
    /// Based on Beagle's mov_sp pattern
    fn emit_mov_sp(&mut self, dst: usize, src: usize) {
        // ADD Xd, Xn, #0
        // Works for both MOV from SP and MOV to SP
        // ADD instruction properly interprets register 31 as SP, not XZR
        let instruction = 0x910003E0 | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_mov_imm(&mut self, dst: usize, imm: i64) {
        let imm = imm as u64; // Treat as unsigned for bitwise ops

        // Extract 16-bit chunks
        let chunk0 = (imm & 0xFFFF) as u32;
        let chunk1 = ((imm >> 16) & 0xFFFF) as u32;
        let chunk2 = ((imm >> 32) & 0xFFFF) as u32;
        let chunk3 = ((imm >> 48) & 0xFFFF) as u32;

        // MOVZ Xd, #chunk0 (always emit this)
        let movz = 0xD2800000 | (chunk0 << 5) | (dst as u32);
        self.code.push(movz);

        // MOVK Xd, #chunk1, LSL #16 (if non-zero)
        if chunk1 != 0 {
            let movk = 0xF2A00000 | (chunk1 << 5) | (dst as u32);
            self.code.push(movk);
        }

        // MOVK Xd, #chunk2, LSL #32 (if non-zero)
        if chunk2 != 0 {
            let movk = 0xF2C00000 | (chunk2 << 5) | (dst as u32);
            self.code.push(movk);
        }

        // MOVK Xd, #chunk3, LSL #48 (if non-zero)
        if chunk3 != 0 {
            let movk = 0xF2E00000 | (chunk3 << 5) | (dst as u32);
            self.code.push(movk);
        }
    }

    fn emit_add(&mut self, dst: usize, src1: usize, src2: usize) {
        // ADD Xd, Xn, Xm
        let instruction = 0x8B000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sub(&mut self, dst: usize, src1: usize, src2: usize) {
        // SUB Xd, Xn, Xm
        let instruction = 0xCB000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sub_sp_imm(&mut self, imm: i64) {
        // SUB sp, sp, #imm
        let instruction = 0xD10003FF | ((imm as u32 & 0xFFF) << 10);
        self.code.push(instruction);
    }

    fn emit_add_sp_imm(&mut self, imm: i64) {
        // ADD sp, sp, #imm
        let instruction = 0x910003FF | ((imm as u32 & 0xFFF) << 10);
        self.code.push(instruction);
    }

    #[allow(dead_code)]
    fn emit_add_sp_offset_to_reg(&mut self, dst: usize, imm: i64) {
        // ADD Xd, SP, #imm - compute address relative to stack pointer
        // Using register 31 as src means SP in this context
        let instruction = 0x910003E0 | ((imm as u32 & 0xFFF) << 10) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_add_imm(&mut self, dst: usize, src: usize, imm: i64) {
        // ADD Xd, Xn, #imm
        let instruction =
            0x91000000 | ((imm as u32 & 0xFFF) << 10) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sub_imm(&mut self, dst: usize, src: usize, imm: i64) {
        // SUB Xd, Xn, #imm
        let instruction =
            0xD1000000 | ((imm as u32 & 0xFFF) << 10) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_mul(&mut self, dst: usize, src1: usize, src2: usize) {
        // MUL Xd, Xn, Xm (MADD with XZR)
        let instruction = 0x9B007C00 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sdiv(&mut self, dst: usize, src1: usize, src2: usize) {
        // SDIV Xd, Xn, Xm - signed division
        let instruction = 0x9AC00C00 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    // Bitwise operations

    fn emit_and(&mut self, dst: usize, src1: usize, src2: usize) {
        // AND Xd, Xn, Xm - bitwise AND
        let instruction = 0x8A000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_orr(&mut self, dst: usize, src1: usize, src2: usize) {
        // ORR Xd, Xn, Xm - bitwise OR
        let instruction = 0xAA000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_eor(&mut self, dst: usize, src1: usize, src2: usize) {
        // EOR Xd, Xn, Xm - bitwise XOR (exclusive OR)
        let instruction = 0xCA000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_mvn(&mut self, dst: usize, src: usize) {
        // MVN Xd, Xm - bitwise NOT (move with NOT)
        // MVN is actually ORN Xd, XZR, Xm
        let instruction = 0xAA2003E0 | ((src as u32) << 16) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_lsl(&mut self, dst: usize, src1: usize, src2: usize) {
        // LSL Xd, Xn, Xm - logical shift left (variable)
        // This is actually LSLV
        let instruction = 0x9AC02000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_asr(&mut self, dst: usize, src1: usize, src2: usize) {
        // ASR Xd, Xn, Xm - arithmetic shift right (variable)
        // This is actually ASRV - preserves sign bit
        let instruction = 0x9AC02800 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_lsr(&mut self, dst: usize, src1: usize, src2: usize) {
        // LSR Xd, Xn, Xm - logical shift right (variable)
        // This is actually LSRV - zero-fills from left
        let instruction = 0x9AC02400 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    // Floating-point operations (double precision)
    // We use D registers (which are the low 64 bits of V registers)
    // Register encoding: d0-d31 use same encoding as v0-v31

    fn emit_fmov_general_to_float(&mut self, dst_fp: usize, src_gp: usize) {
        // FMOV Dd, Xn - move from general register to FP register (as raw bits)
        // Encoding: 0x9E670000 | (Rn << 5) | Rd
        let instruction = 0x9E670000 | ((src_gp as u32) << 5) | (dst_fp as u32);
        self.code.push(instruction);
    }

    fn emit_fmov_float_to_general(&mut self, dst_gp: usize, src_fp: usize) {
        // FMOV Xd, Dn - move from FP register to general register (as raw bits)
        // Encoding: 0x9E660000 | (Rn << 5) | Rd
        let instruction = 0x9E660000 | ((src_fp as u32) << 5) | (dst_gp as u32);
        self.code.push(instruction);
    }

    fn emit_fadd(&mut self, dst: usize, src1: usize, src2: usize) {
        // FADD Dd, Dn, Dm - double-precision add
        // Encoding: 0x1E602800 | (Rm << 16) | (Rn << 5) | Rd
        let instruction = 0x1E602800 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_fsub(&mut self, dst: usize, src1: usize, src2: usize) {
        // FSUB Dd, Dn, Dm - double-precision subtract
        // Encoding: 0x1E603800 | (Rm << 16) | (Rn << 5) | Rd
        let instruction = 0x1E603800 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_fmul(&mut self, dst: usize, src1: usize, src2: usize) {
        // FMUL Dd, Dn, Dm - double-precision multiply
        // Encoding: 0x1E600800 | (Rm << 16) | (Rn << 5) | Rd
        let instruction = 0x1E600800 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_fdiv(&mut self, dst: usize, src1: usize, src2: usize) {
        // FDIV Dd, Dn, Dm - double-precision divide
        // Encoding: 0x1E601800 | (Rm << 16) | (Rn << 5) | Rd
        let instruction = 0x1E601800 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_scvtf(&mut self, dst_fp: usize, src_gp: usize) {
        // SCVTF Dd, Xn - convert signed integer to double
        // Encoding: 0x9E620000 | (Rn << 5) | Rd
        let instruction = 0x9E620000 | ((src_gp as u32) << 5) | (dst_fp as u32);
        self.code.push(instruction);
    }

    fn emit_lsl_imm(&mut self, dst: usize, src: usize, shift: u32) {
        // LSL Xd, Xn, #shift (logical shift left)
        // This is actually UBFM (Unsigned Bitfield Move)
        // LSL #shift is: UBFM Xd, Xn, #(-shift mod 64), #(63-shift)
        let shift = shift & 0x3F; // 6 bits
        let immr = (64 - shift) & 0x3F;
        let imms = 63 - shift;
        let instruction =
            0xD3400000 | (immr << 16) | (imms << 10) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_asr_imm(&mut self, dst: usize, src: usize, shift: u32) {
        // ASR Xd, Xn, #shift (arithmetic shift right)
        // This is SBFM (Signed Bitfield Move)
        // ASR #shift is: SBFM Xd, Xn, #shift, #63
        let shift = shift & 0x3F; // 6 bits
        let instruction = 0x9340FC00 | (shift << 16) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_lsr_imm(&mut self, dst: usize, src: usize, shift: usize) {
        // LSR Xd, Xn, #shift (logical shift right - unsigned)
        // This is UBFM (Unsigned Bitfield Move)
        // LSR #shift is: UBFM Xd, Xn, #shift, #63
        // Encoding: sf=1 opc=10 N=1 immr=shift imms=63
        // 1 10 100110 1 immr imms Rn Rd
        // = 0xD340FC00 | (shift << 16) | (src << 5) | dst
        let shift = (shift as u32) & 0x3F; // 6 bits
        let instruction = 0xD340FC00 | (shift << 16) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_cmp(&mut self, src1: usize, src2: usize) {
        // CMP Xn, Xm (compare - this is SUBS XZR, Xn, Xm)
        let instruction = 0xEB00001F | ((src2 as u32) << 16) | ((src1 as u32) << 5);
        self.code.push(instruction);
    }

    fn emit_cmp_imm(&mut self, src: usize, imm: i64) {
        // CMP Xn, #imm (compare - this is SUBS XZR, Xn, #imm)
        let imm12 = (imm & 0xFFF) as u32; // 12-bit immediate
        let instruction = 0xF100001F | (imm12 << 10) | ((src as u32) << 5);
        self.code.push(instruction);
    }

    fn emit_branch_cond(&mut self, label: Label, cond: u32) {
        // B.cond label
        // Record this as a pending fixup
        let fixup_index = self.code.len();
        self.pending_fixups.push((fixup_index, label));

        // Emit placeholder with condition
        let instruction = 0x54000000 | (cond & 0xF);
        self.code.push(instruction);
    }

    fn emit_label(&mut self, label: Label) {
        // Record the current position for this label
        let pos = self.code.len();
        self.label_positions.insert(label, pos);
    }

    fn emit_jump(&mut self, label: &Label) {
        // B label (unconditional branch)
        let fixup_index = self.code.len();
        self.pending_fixups.push((fixup_index, label.clone()));

        // Emit placeholder
        let instruction = 0x14000000;
        self.code.push(instruction);
    }

    #[allow(dead_code)]
    fn emit_conditional_branch(&mut self, label: &Label, condition: u32) {
        // B.cond label (conditional branch)
        // Encoding: 0101_0100 | imm19 << 5 | cond
        // condition: 0=EQ, 1=NE, 10=GE, 11=LT, 12=GT, 13=LE
        let fixup_index = self.code.len();
        self.pending_fixups.push((fixup_index, label.clone()));

        // Emit placeholder with condition
        let instruction = 0x54000000 | condition;
        self.code.push(instruction);
    }

    fn emit_str_offset(&mut self, src: usize, base: usize, offset: i32) {
        // STR Xt, [Xn, #offset]
        // Offset is in bytes, needs to be divided by 8 for encoding (unsigned 12-bit)
        let offset_scaled = (offset / 8) as u32;
        let instruction = 0xF9000000 | (offset_scaled << 10) | ((base as u32) << 5) | (src as u32);
        self.code.push(instruction);
    }

    /// Store with register offset: STR Xt, [Xn, Xm]
    #[allow(dead_code)]
    fn emit_str_reg_offset(&mut self, src: usize, base: usize, offset_reg: usize) {
        // STR Xt, [Xn, Xm] - Store with register offset
        // Encoding: 11 111000 00 1 Rm 011 0 10 Rn Rt
        // = F8206800 | (Rm << 16) | (Rn << 5) | Rt
        let instruction =
            0xF8206800 | ((offset_reg as u32) << 16) | ((base as u32) << 5) | (src as u32);
        self.code.push(instruction);
    }

    fn emit_ldrb_offset(&mut self, dst: usize, base: usize, offset: i32) {
        // LDRB Wt, [Xn, #offset]
        // Encoding: 0x39400000 | (imm12 << 10) | (Rn << 5) | Rt
        let offset_u = (offset as u32) & 0xFFF; // 12-bit unsigned offset
        let instruction = 0x39400000 | (offset_u << 10) | ((base as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_load_from_fp(&mut self, dst: usize, offset: i32) {
        // LDR Xd, [x29, #offset] with signed offset
        // LDUR only supports 9-bit signed offset (-256 to +255)
        // For larger offsets, we need to compute the address first
        if (-256..=255).contains(&offset) {
            // LDUR Xd, [x29, #offset]
            let offset_bits = (offset as u32) & 0x1FF; // 9-bit signed
            let instruction = 0xF8400000 | (offset_bits << 12) | (29 << 5) | (dst as u32);
            self.code.push(instruction);
        } else {
            // Large offset: compute address in dst, then load
            // ADD dst, x29, #offset (may need multiple instructions for large negative)
            self.emit_add_imm_large(dst, 29, offset as i64);
            // LDR dst, [dst, #0]
            // Encoding: 0xF9400000 | (imm12 << 10) | (Rn << 5) | Rt
            // For offset 0: imm12 = 0
            let instruction = 0xF9400000 | ((dst as u32) << 5) | (dst as u32);
            self.code.push(instruction);
        }
    }

    fn emit_store_to_fp(&mut self, src: usize, offset: i32) {
        // STR Xt, [x29, #offset] with signed offset
        // STUR only supports 9-bit signed offset (-256 to +255)
        // For larger offsets, we need to compute the address first
        if (-256..=255).contains(&offset) {
            // STUR Xt, [x29, #offset]
            let offset_bits = (offset as u32) & 0x1FF; // 9-bit signed
            let instruction = 0xF8000000 | (offset_bits << 12) | (29 << 5) | (src as u32);
            self.code.push(instruction);
        } else {
            // Large offset: compute address in a scratch register, then store
            // We need to pick a scratch that's NOT src
            let scratch = if src == 9 {
                10
            } else if src == 10 {
                11
            } else {
                9
            };
            // ADD scratch, x29, #offset
            self.emit_add_imm_large(scratch, 29, offset as i64);
            // STR src, [scratch, #0]
            // Encoding: 0xF9000000 | (imm12 << 10) | (Rn << 5) | Rt
            // For offset 0: imm12 = 0
            let instruction = 0xF9000000 | ((scratch as u32) << 5) | (src as u32);
            self.code.push(instruction);
        }
    }

    // === Beagle-style stack operations (FP-relative) ===
    // These methods mimic Beagle's approach exactly:
    // - Stack is addressed relative to frame pointer (x29), not SP
    // - push/pop track logical stack_size without modifying SP
    // - Actual stack space is reserved in prologue/epilogue

    /// Store register to stack at FP-relative offset (Beagle: store_on_stack)
    fn store_on_stack(&mut self, reg: usize, offset: i32) {
        let byte_offset = offset * 8;
        self.emit_store_to_fp(reg, byte_offset);
    }

    /// Load register from stack at FP-relative offset (Beagle: load_from_stack)
    fn load_from_stack(&mut self, reg: usize, offset: i32) {
        let byte_offset = offset * 8;
        self.emit_load_from_fp(reg, byte_offset);
    }

    /// Push register to stack (Beagle: push_to_stack)
    /// Increments logical stack_size and stores at FP-relative offset
    /// Does NOT modify SP - stack space is allocated in prologue
    fn push_to_stack(&mut self, reg: usize) {
        self.current_stack_size += 1;
        if self.current_stack_size > self.max_stack_size {
            self.max_stack_size = self.current_stack_size;
        }
        // Offset = -(num_stack_slots + current_stack_size)
        // This places saves AFTER all reserved slots (spills + exception + locals)
        let offset = -((self.num_stack_slots as i32) + (self.current_stack_size as i32));
        self.store_on_stack(reg, offset);
    }

    /// Pop register from stack (Beagle: pop_from_stack)
    /// Decrements logical stack_size and loads from FP-relative offset
    /// Does NOT modify SP - stack space is deallocated in epilogue
    fn pop_from_stack(&mut self, reg: usize) {
        // Load from the slot we're about to pop
        let offset = -((self.num_stack_slots as i32) + (self.current_stack_size as i32));
        self.load_from_stack(reg, offset);
        // Then decrement the logical stack size
        self.current_stack_size = self.current_stack_size.saturating_sub(1);
    }

    /// Get pointer to current stack position (where next push will go)
    /// Computes FP - ((num_stack_slots + current_stack_size + 1) * 8)
    /// Used for closure values: get pointer BEFORE pushing, then push values
    fn get_current_stack_position(&mut self, dest_reg: usize) {
        let offset = ((self.num_stack_slots + self.current_stack_size + 1) * 8) as i64;
        // SUB Xd, X29, #offset
        self.emit_sub_imm(dest_reg, 29, offset);
    }

    /// Initialize all local/spill slots to null (Beagle pattern)
    /// This ensures GC won't see garbage as heap pointers during stack scanning.
    /// Called after stack allocation in function prologues.
    /// Must initialize ALL stack slots: spill slots + exception slots + local slots
    fn set_all_locals_to_null(&mut self, null_register: usize) {
        for slot in 0..self.num_stack_slots {
            // Stack layout: [FP - (slot + 1) * 8] = local/spill slot
            let offset = -((slot as i32 + 1) * 8);
            self.emit_store_to_fp(null_register, offset);
        }
    }

    /// Patch prologue and epilogue with actual stack size (Beagle-style)
    /// Finds placeholder instructions and replaces with correct stack allocation
    fn patch_prologue_epilogue(&mut self, prologue_index: usize, stack_bytes: usize) {
        if stack_bytes == 0 {
            // No stack needed - replace with NOP
            self.code[prologue_index] = 0xD503201F; // NOP
            // Find and patch epilogue placeholder too
            if let Some(epilogue_index) = self.find_epilogue_placeholder() {
                self.code[epilogue_index] = 0xD503201F; // NOP
            }
            return;
        }

        // Patch prologue SUB
        if stack_bytes <= 4095 {
            // Single SUB instruction: SUB SP, SP, #stack_bytes
            let instruction = 0xD1000000  // SUB (immediate) base
                | ((stack_bytes as u32) << 10)  // imm12
                | (31 << 5)  // Rn = SP
                | 31; // Rd = SP
            self.code[prologue_index] = instruction;
        } else {
            // For large stack sizes, we need multiple instructions
            // For now, panic - we can add multi-instruction support later
            panic!(
                "Stack size {} too large for single SUB instruction",
                stack_bytes
            );
        }

        // Find and patch epilogue ADD placeholder
        if let Some(epilogue_index) = self.find_epilogue_placeholder() {
            if stack_bytes <= 4095 {
                // Single ADD instruction: ADD SP, SP, #stack_bytes
                let instruction = 0x91000000  // ADD (immediate) base
                    | ((stack_bytes as u32) << 10)  // imm12
                    | (31 << 5)  // Rn = SP
                    | 31; // Rd = SP
                self.code[epilogue_index] = instruction;
            } else {
                panic!(
                    "Stack size {} too large for single ADD instruction",
                    stack_bytes
                );
            }
        }
    }

    /// Find the epilogue placeholder ADD instruction (has magic value 0xAAA)
    fn find_epilogue_placeholder(&self) -> Option<usize> {
        // Look for ADD SP, SP, #0xAAA from the end
        // emit_add_sp_imm produces: 0x910003FF | (imm << 10)
        // So for imm=0xAAA: 0x910003FF | (0xAAA << 10) = 0x910003FF | 0x2AA800 = 0x912AABFF
        let expected = 0x910003FF_u32 | (0xAAA << 10);

        
        self.code.iter().rposition(|&inst| inst == expected)
    }

    /// Emit ADD with potentially large immediate (positive or negative)
    fn emit_add_imm_large(&mut self, dst: usize, src: usize, imm: i64) {
        if (0..=4095).contains(&imm) {
            // Simple ADD with 12-bit immediate
            self.emit_add_imm(dst, src, imm);
        } else if (-4095..0).contains(&imm) {
            // SUB with negated immediate
            self.emit_sub_imm(dst, src, -imm);
        } else {
            // Large immediate: load into dst first, then add
            self.emit_mov_imm(dst, imm);
            // ADD dst, src, dst
            let instruction =
                0x8B000000 | ((dst as u32) << 16) | ((src as u32) << 5) | (dst as u32);
            self.code.push(instruction);
        }
    }

    /// Emit function prologue with placeholder for stack allocation
    /// This follows Beagle's pattern of deferred stack allocation
    #[allow(dead_code)]
    fn emit_prologue_with_placeholder(&mut self, label: &Label) {
        // Save FP and LR
        self.emit_stp(29, 30, 31, -2); // stp x29, x30, [sp, #-16]!
        self.emit_mov(29, 31); // mov x29, sp

        // Emit placeholder SUB instruction (will be patched later)
        let prologue_idx = self.code.len();
        self.emit_sub_sp_imm(0xAAA); // Magic placeholder value (fits in 12-bit imm)

        // Record placeholder position
        self.placeholder_positions
            .entry(label.clone())
            .or_insert((prologue_idx, 0))
            .0 = prologue_idx;

        // Reset stack counter for this function
        self.current_function_stack_bytes = 0;
    }

    /// Emit function epilogue with placeholder for stack deallocation
    #[allow(dead_code)]
    fn emit_epilogue_with_placeholder(&mut self, label: &Label) {
        // Emit placeholder ADD instruction (will be patched later)
        let epilogue_idx = self.code.len();
        self.emit_add_sp_imm(0xAAA); // Magic placeholder value (fits in 12-bit imm)

        // Record placeholder position
        if let Some((_, epi)) = self.placeholder_positions.get_mut(label) {
            *epi = epilogue_idx;
        }

        // Restore FP and LR
        self.emit_ldp(29, 30, 31, 2); // ldp x29, x30, [sp], #16

        // Emit return
        self.emit_ret();
    }

    fn emit_stp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        // STP Xt, Xt2, [Xn, #offset]! (pre-index)
        // offset is in 8-byte units for STP, range -512 to 504
        let offset_scaled = ((offset & 0x7F) as u32) << 15; // 7-bit signed offset
        let instruction =
            0xA9800000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ldp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        // LDP Xt, Xt2, [Xn], #offset (post-index)
        // offset is in 8-byte units for LDP, range -512 to 504
        let offset_scaled = ((offset & 0x7F) as u32) << 15; // 7-bit signed offset
        let instruction =
            0xA8C00000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ret(&mut self) {
        // RET (returns to address in X30/LR)
        self.code.push(0xD65F03C0);
    }

    fn emit_blr(&mut self, rn: usize) {
        // BLR Xn - Branch with Link to Register
        // Calls function at address in Xn, stores return address in X30
        let instruction = 0xD63F0000 | ((rn as u32) << 5);
        self.code.push(instruction);
        // Record stack map entry after call for GC root scanning
        self.update_stack_map();
    }

    // === Stack Map Methods for GC ===

    /// Record current stack state after a call instruction
    /// The GC uses this to find roots on the stack
    fn update_stack_map(&mut self) {
        // Record instruction offset (index in code array, not byte offset)
        // The -1 is because we want the instruction we just emitted (the BLR)
        let offset = self.code.len() - 1;
        self.stack_map.insert(offset, self.current_stack_size);
    }

    /// Translate stack map from instruction offsets to absolute addresses
    /// Called after code is placed in memory to get actual PC values
    pub fn translate_stack_map(&self, base_pointer: usize) -> Vec<(usize, usize)> {
        #[cfg(feature = "debug-gc")]
        if !self.stack_map.is_empty() {
            eprintln!("[GC DEBUG] translate_stack_map: base_pointer={:#x}, code_len={}",
                base_pointer, self.code.len());
        }
        self.stack_map
            .iter()
            .map(|(offset, stack_size)| {
                // Each instruction is 4 bytes, offset is index of BLR instruction.
                // The return address (x30 after BLR) points to the NEXT instruction,
                // which is at (offset + 1) * 4 + base_pointer.
                let return_addr = ((*offset + 1) * 4) + base_pointer;
                (return_addr, *stack_size)
            })
            .collect()
    }

    /// Increment stack size tracking in words (call when allocating stack space)
    pub fn increment_stack_size(&mut self, words: usize) {
        self.current_stack_size += words;
        if self.current_stack_size > self.max_stack_size {
            self.max_stack_size = self.current_stack_size;
        }
    }

    /// Decrement stack size tracking in words (call when deallocating stack space)
    pub fn decrement_stack_size(&mut self, words: usize) {
        self.current_stack_size = self.current_stack_size.saturating_sub(words);
    }

    /// Emit an external function call
    ///
    /// X30 is already saved in the function prologue (stp x29, x30, [sp, #-16]!),
    /// so we don't need to save/restore it here. Just load the function address
    /// and call via BLR - like Beagle's call_builtin.
    ///
    /// # Parameters
    /// - `target_fn`: Address of the external function to call
    /// - `_description`: Human-readable description for debugging (currently unused)
    fn emit_external_call(&mut self, target_fn: usize, _description: &str) {
        // Load function address and call (X30 already saved in prologue)
        self.emit_mov_imm(15, target_fn as i64); // Use x15 as temp
        self.emit_blr(15);
    }
}
