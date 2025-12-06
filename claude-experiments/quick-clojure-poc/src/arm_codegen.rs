use crate::ir::{Instruction, IrValue, VirtualRegister, Condition, Label};
use crate::register_allocation::linear_scan::LinearScan;
use crate::trampoline::Trampoline;
use std::collections::{HashMap, BTreeMap};

/// ARM64 code generator - compiles IR to ARM64 machine code
///
/// This is based on Beagle's ARM64 backend but simplified for our needs.
pub struct Arm64CodeGen {
    /// Generated ARM64 machine code (32-bit instructions)
    code: Vec<u32>,

    /// Map from virtual registers to physical ARM64 registers (from linear scan)
    register_map: BTreeMap<VirtualRegister, VirtualRegister>,

    /// Next physical register to allocate
    next_physical_reg: usize,

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
            next_physical_reg: 0,
            label_positions: HashMap::new(),
            pending_fixups: Vec::new(),
            pending_adr_fixups: Vec::new(),
            temp_register_pool: vec![11, 10, 9],  // Start with x11, x10, x9 available
            label_counter: 0,
        }
    }

    fn new_label(&mut self) -> Label {
        let label = format!("L{}", self.label_counter);
        self.label_counter += 1;
        label
    }

    /// Compile IR instructions to ARM64 machine code
    ///
    /// # Parameters
    /// - `instructions`: IR instructions to compile
    /// - `result_reg`: The register containing the final result
    /// - `num_registers`: Number of registers available (0 = default/unlimited)
    pub fn compile(&mut self, instructions: &[Instruction], result_reg: &IrValue, num_registers: usize) -> Result<Vec<u32>, String> {
        // Reset state
        self.code.clear();
        self.register_map.clear();
        self.next_physical_reg = 0;
        self.label_positions.clear();
        self.pending_fixups.clear();
        self.pending_adr_fixups.clear();

        // Run linear scan register allocation
        let mut allocator = LinearScan::new(instructions.to_vec(), num_registers);

        // Mark result register as live until the end
        // This is critical - without this, the register allocator may reuse
        // the physical register for the result, causing wrong values to be returned
        if let IrValue::Register(vreg) = result_reg {
            allocator.mark_live_until_end(*vreg);
        }

        allocator.allocate();

        // Debug output BEFORE consuming allocator
        let num_stack_slots = allocator.next_stack_slot;
        let num_spills = allocator.spill_locations.len();
        eprintln!("DEBUG: {} spills, {} total stack slots", num_spills, num_stack_slots);
        eprintln!("DEBUG: Spill locations from codegen allocator:");
        for (vreg, slot) in &allocator.spill_locations {
            eprintln!("  {} -> slot {}", vreg.display_name(), slot);
        }

        // Store the register allocation map for use in codegen
        self.register_map = allocator.allocated_registers.clone();

        // Debug: print allocation for v27 if it exists
        let v27 = crate::ir::VirtualRegister::Temp(27);
        if let Some(physical) = self.register_map.get(&v27) {
            eprintln!("DEBUG: v27 allocated to x{}", physical.index());
        } else {
            eprintln!("DEBUG: v27 not in allocation map (might be physical already or not used)");
        }

        // Find the physical register for the result (before consuming allocator)
        let result_physical = if let IrValue::Register(vreg) = result_reg {
            allocator.allocated_registers.get(vreg)
                .ok_or_else(|| format!("Result register {:?} not allocated", vreg))?
                .index()
        } else {
            return Err(format!("Expected register for result, got {:?}", result_reg));
        };

        // Determine which callee-saved registers (x19-x28) are used
        let mut used_callee_saved: Vec<usize> = allocator.allocated_registers
            .values()
            .map(|vreg| vreg.index())
            .filter(|&idx| idx >= 19 && idx <= 28)
            .collect();
        used_callee_saved.sort_unstable();
        used_callee_saved.dedup();
        eprintln!("DEBUG: Saving/restoring callee-saved registers: {:?}", used_callee_saved);

        // Count spills to determine stack space needed
        // Add 8 bytes padding so spills are above SP (ARM64 requirement)
        let stack_space = if num_stack_slots > 0 {
            num_stack_slots * 8 + 8
        } else {
            0
        };

        eprintln!("DEBUG: Allocating {} bytes of stack space", stack_space);

        let allocated_instructions = allocator.finish();

        // Collect function entry point labels (used in MakeFunction)
        let mut function_labels = std::collections::HashSet::new();
        for inst in &allocated_instructions {
            if let Instruction::MakeFunction(_, label, _) = inst {
                function_labels.insert(label.clone());
            }
        }

        // Emit function prologue
        // FIXED: Save callee-saved registers (x19-x28) that are actually used
        // Previously relied on trampoline, but BLR calls don't go through trampoline!
        // Save FP and LR
        self.emit_stp(29, 30, 31, -2);  // stp x29, x30, [sp, #-16]!
        self.emit_mov(29, 31);           // mov x29, sp (set frame pointer)

        // Save used callee-saved registers in pairs (for 16-byte alignment)
        for chunk in used_callee_saved.chunks(2) {
            if chunk.len() == 2 {
                self.emit_stp(chunk[0], chunk[1], 31, -2);  // stp xN, xM, [sp, #-16]!
            } else {
                // Odd number - save single register with padding
                self.emit_stp(chunk[0], 31, 31, -2);  // stp xN, xzr, [sp, #-16]!
            }
        }

        // Allocate stack space for spills if needed
        if stack_space > 0 {
            // sub sp, sp, #stack_space
            self.emit_sub_sp_imm(stack_space as i64);
        }

        // Compile each instruction (now with physical registers)
        for inst in &allocated_instructions {
            self.compile_instruction(inst)?;
        }

        // Apply jump fixups AFTER all code is generated
        // (We'll apply fixups after emitting epilogue)

        // Emit epilogue label (where Ret instructions jump to)
        self.emit_label("__epilogue".to_string());

        // Move result to x0 (keep it tagged)
        // For Ret instructions, they've already moved their result to x0 before jumping here
        // But for top-level code that falls through (no explicit Ret), we need to move result to x0
        // Since both paths go through here, and Ret already ensures result is in x0,
        // this is a harmless mov x0, x0 for functions with Ret, but necessary for top-level code
        eprintln!("DEBUG: Epilogue - result_physical=x{}", result_physical);
        if result_physical != 0 {
            self.emit_mov(0, result_physical);
        }

        // Deallocate stack space for spills if needed
        if stack_space > 0 {
            // add sp, sp, #stack_space
            self.emit_add_sp_imm(stack_space as i64);
        }

        // Emit function epilogue
        // FIXED: Restore callee-saved registers in reverse order
        for chunk in used_callee_saved.chunks(2).rev() {
            if chunk.len() == 2 {
                self.emit_ldp(chunk[0], chunk[1], 31, 2);  // ldp xN, xM, [sp], #16
            } else {
                // Odd number - restore single register (ignore padding)
                self.emit_ldp(chunk[0], 31, 31, 2);  // ldp xN, xzr, [sp], #16
            }
        }

        // Restore FP and LR
        self.emit_ldp(29, 30, 31, 2);    // ldp x29, x30, [sp], #16

        // Emit return instruction
        self.emit_ret();

        // NOW apply jump fixups after all labels are defined
        self.apply_fixups()?;

        Ok(self.code.clone())
    }

    fn compile_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
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
                    _ => return Err(format!("Invalid constant: {:?}", value)),
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadVar(dst, var_ptr) => {
                // LoadVar: call trampoline to check dynamic bindings
                // ARM64 calling convention:
                // - x0 = argument (var_ptr, tagged)
                // - x0 = return value (tagged)
                // - x30 = link register (return address)
                // - x19-x28 are callee-saved (our allocator uses these, so they're safe)
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        eprintln!("DEBUG LoadVar codegen: var_ptr={:x}", tagged_ptr);

                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_var_get_value_dynamic as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        // BLR x15 - stores return address in x30, jumps to x15
                        self.emit_blr(15);

                        // Result is in x0, move to destination if needed
                        if dst_reg != 0 {
                            self.emit_mov(dst_reg, 0);
                        }
                    }
                    _ => return Err(format!("LoadVar requires constant var pointer: {:?}", var_ptr)),
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::StoreVar(var_ptr, value) => {
                eprintln!("DEBUG: StoreVar - var_ptr={:?}, value={:?}", var_ptr, value);
                eprintln!("DEBUG: StoreVar - code position before: {}", self.code.len());
                // StoreVar: store value into var at runtime
                // Var layout: [header(8)] [ns_ptr(8)] [symbol_ptr(8)] [value(8)]
                // We want to write to field 2 (value) which is at offset 24 bytes (3 * 8)
                let value_reg = self.get_physical_reg_for_irvalue(value, false)?;
                eprintln!("DEBUG: StoreVar - value_reg=x{}, code position: {}", value_reg, self.code.len());

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Untag the var pointer (shift right by 3)
                        let untagged_ptr = (*tagged_ptr as usize) >> 3;
                        eprintln!("DEBUG: StoreVar - untagged_ptr={:x}", untagged_ptr);

                        let code_pos_before_movimm = self.code.len();
                        // Load var pointer into a temp register
                        // Use x15 as temp register - it's the highest general purpose register
                        // and unlikely to be allocated by our simple allocator
                        let temp_reg = 15;
                        self.emit_mov_imm(temp_reg, untagged_ptr as i64);
                        let code_pos_after_movimm = self.code.len();
                        eprintln!("DEBUG: StoreVar - emit_mov_imm emitted {} instructions from {} to {}",
                                 code_pos_after_movimm - code_pos_before_movimm,
                                 code_pos_before_movimm, code_pos_after_movimm - 1);

                        // Store value into var (offset 24 = header + ns_ptr + symbol_ptr)
                        // str value_reg, [temp_reg, #24]
                        let code_pos_before_str = self.code.len();
                        self.emit_str_offset(value_reg, temp_reg, 24);
                        eprintln!("DEBUG: StoreVar - emit_str at position {}, storing x{} to [x{}+24]",
                                 code_pos_before_str, value_reg, temp_reg);
                        eprintln!("DEBUG: StoreVar - actual str instruction: 0x{:08x}",
                                 self.code[code_pos_before_str]);
                    }
                    _ => return Err(format!("StoreVar requires constant var pointer: {:?}", var_ptr)),
                }
            }

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
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // eprintln!("DEBUG: Untag - dst={:?} (x{}), src={:?} (x{})", dst, dst_reg, src, src_reg);
                // Untag: arithmetic right shift by 3
                self.emit_asr_imm(dst_reg, src_reg, 3);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Tag(dst, src, _tag) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // eprintln!("DEBUG: Tag - dst={:?} (x{}), src={:?} (x{})", dst, dst_reg, src, src_reg);
                // Tag: left shift by 3 (int tag is 000)
                self.emit_lsl_imm(dst_reg, src_reg, 3);
                self.store_spill(dst_reg, dest_spill);
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

            Instruction::Assign(dst, src) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                if dst_reg != src_reg {
                    self.emit_mov(dst_reg, src_reg);
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::PushBinding(var_ptr, value) => {
                // PushBinding: call trampoline to push a dynamic binding
                // ARM64 calling convention:
                // - x0 = var_ptr (tagged)
                // - x1 = value (tagged)
                // - x0 = return value (0 = success, 1 = error)

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Get the value register
                        let value_reg = self.get_physical_reg_for_irvalue(value, false)?;

                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Move value to x1 (second argument) if not already there
                        if value_reg != 1 {
                            self.emit_mov(1, value_reg);
                        }

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_push_binding as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        self.emit_blr(15);

                        // Return value in x0 (0 = success, 1 = error)
                        // For now, we ignore errors (could add error handling later)
                    }
                    _ => return Err(format!("PushBinding requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::PopBinding(var_ptr) => {
                // PopBinding: call trampoline to pop a dynamic binding
                // ARM64 calling convention:
                // - x0 = var_ptr (tagged)
                // - x0 = return value (0 = success, 1 = error)

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_pop_binding as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        self.emit_blr(15);

                        // Return value in x0 (0 = success, 1 = error)
                        // For now, we ignore errors (could add error handling later)
                    }
                    _ => return Err(format!("PopBinding requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::SetVar(var_ptr, value) => {
                // SetVar: call trampoline to modify a thread-local binding (for set!)
                // ARM64 calling convention:
                // - x0 = var_ptr (tagged)
                // - x1 = value (tagged)
                // - x0 = return value (0 = success, 1 = error)

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Get the value register
                        let value_reg = self.get_physical_reg_for_irvalue(value, false)?;

                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Move value to x1 (second argument) if not already there
                        if value_reg != 1 {
                            self.emit_mov(1, value_reg);
                        }

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_set_binding as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        self.emit_blr(15);

                        // Return value in x0 (0 = success, 1 = error)
                        // For now, we ignore errors (could add error handling later)
                    }
                    _ => return Err(format!("SetVar requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::Label(label) => {
                // Record position of this label
                let pos = self.code.len();
                eprintln!("DEBUG: Label {} at code position {}", label, pos);
                self.label_positions.insert(label.clone(), pos);
            }

            Instruction::Jump(label) => {
                // Emit unconditional branch
                // We'll fix up the offset later
                let fixup_index = self.code.len();
                eprintln!("DEBUG: Jump to {} from code position {}", label, fixup_index);
                self.pending_fixups.push((fixup_index, label.clone()));
                // Placeholder - will be patched in apply_fixups
                self.code.push(0x14000000); // B #0
            }

            Instruction::JumpIf(label, cond, src1, src2) => {
                // Compare src1 and src2
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;

                // Emit CMP instruction
                self.emit_cmp(src1_reg, src2_reg);

                // Emit conditional branch
                let fixup_index = self.code.len();
                self.pending_fixups.push((fixup_index, label.clone()));

                // Placeholder conditional branch - will be patched in apply_fixups
                let branch_cond = match cond {
                    Condition::Equal => 0,       // EQ
                    Condition::NotEqual => 1,    // NE
                    Condition::LessThan => 11,   // LT
                    Condition::LessThanOrEqual => 13, // LE
                    Condition::GreaterThan => 12, // GT
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
                    Condition::Equal => 0,       // EQ
                    Condition::NotEqual => 1,    // NE
                    Condition::LessThan => 11,   // LT
                    Condition::LessThanOrEqual => 13, // LE
                    Condition::GreaterThan => 12, // GT
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

            Instruction::MakeFunction(dst, label, closure_values) => {
                eprintln!("DEBUG: MakeFunction - dst={:?}, label={}, closure_values={:?}", dst, label, closure_values);

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                if closure_values.is_empty() {
                    // Regular function (no closures) - just tag the code pointer
                    // Tagged value = (code_ptr << 3) | 0b100
                    eprintln!("DEBUG: MakeFunction - creating regular function (no closures)");

                    // Load code address with ADR into a temporary register
                    let temp_reg = 10;  // Use x10 as temporary
                    self.emit_adr(temp_reg, label.clone());

                    // Shift left by 3 and add Function tag (0b100)
                    // LSL Xd, Xn, #3 = UBFM Xd, Xn, #61, #60
                    // From Beagle: immr = 64 - shift = 61, imms = immr - 1 = 60
                    // Base: 0b0_10_100110_0_000000_000000_00000_00000 = 0x53000000
                    // With sf=1, n=1: 0xD37DF000 | (rn << 5) | rd
                    let lsl_instruction = 0xD37DF000u32 | ((temp_reg as u32) << 5) | (dst_reg as u32);
                    self.code.push(lsl_instruction);

                    // ADD Xd, Xn, #0b100 (set tag bits)
                    // After LSL by 3, low 3 bits are 0, so we can add the tag
                    // ADD Xd, Xn, #imm12 = 0x91000000 | (imm12 << 10) | (rn << 5) | rd
                    let add_instruction = 0x91000000u32 | (0b100 << 10) | ((dst_reg as u32) << 5) | (dst_reg as u32);
                    self.code.push(add_instruction);

                    eprintln!("DEBUG: MakeFunction - tagged function pointer in x{}", dst_reg);
                } else {
                    // Closure - allocate heap object with closure values
                    // NEW APPROACH: Stack-based capture (no limit on number of values!)
                    eprintln!("DEBUG: MakeFunction - creating closure with {} captured values", closure_values.len());

                    // Step 1: Allocate stack space for all values at once (maintains 16-byte alignment)
                    let total_stack_space = closure_values.len() * 8;
                    // Round up to 16-byte alignment
                    let aligned_stack_space = ((total_stack_space + 15) / 16) * 16;
                    if aligned_stack_space > 0 {
                        self.emit_sub_sp_imm(aligned_stack_space as i64);
                    }

                    // Step 2: Store values directly to stack (avoids register conflicts)
                    for (i, value) in closure_values.iter().enumerate() {
                        let src_reg = self.get_physical_reg_for_irvalue(value, false)?;
                        let offset = i * 8;
                        eprintln!("DEBUG: MakeFunction - storing closure value {} from x{} (IrValue={:?}) at SP+{}", i, src_reg, value, offset);
                        self.emit_str_offset(src_reg, 31, offset as i32);  // x31 = sp
                    }

                    // Step 4: Save stack pointer (points to start of values array)
                    // Use x10 as temp (won't conflict with callee-saved x19-x28)
                    let values_ptr_reg = 10;  // x10 for values pointer
                    self.emit_mov(values_ptr_reg, 31);  // x31 = sp

                    // Step 4: Set up arguments for trampoline call
                    // x0 = 0 (anonymous function - no name for now)
                    self.emit_mov_imm(0, 0);

                    // x1 = code_ptr (address of the label)
                    self.emit_adr(1, label.clone());

                    // x2 = closure_count
                    self.emit_mov_imm(2, closure_values.len() as i64);

                    // x3 = values_ptr (pointer to values on stack)
                    self.emit_mov(3, values_ptr_reg);

                    // Step 5: Call trampoline to allocate closure heap object
                    let func_addr = crate::trampoline::trampoline_allocate_function as usize;
                    eprintln!("DEBUG: MakeFunction - calling trampoline_allocate_function for closure at {:x}", func_addr);
                    self.emit_external_call(func_addr, "trampoline_allocate_function");

                    // Step 6: Clean up stack (pop all values with correct alignment)
                    if aligned_stack_space > 0 {
                        eprintln!("DEBUG: MakeFunction - cleaning up {} bytes of stack", aligned_stack_space);
                        self.emit_add_sp_imm(aligned_stack_space as i64);
                    }

                    // Step 7: Result is in x0 (tagged closure pointer with 0b101 tag)
                    if dst_reg != 0 {
                        self.emit_mov(dst_reg, 0);
                        eprintln!("DEBUG: MakeFunction - moved closure result from x0 to x{}", dst_reg);
                    }
                }

                self.store_spill(dst_reg, dest_spill);
                eprintln!("DEBUG: MakeFunction - done, spill={:?}", dest_spill);
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
                let untagged_reg = 11;  // Use x11 as temporary for untagged pointer
                let lsr_instruction = 0xD343FC00u32 | ((fn_obj_reg as u32) << 5) | (untagged_reg as u32);
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
                eprintln!("DEBUG: Call - dst={:?}, fn_val={:?}", dst, fn_val);
                // Call: Invoke a function with arguments
                //
                // New approach - check tag inline (NO trampoline calls!):
                // 1. Save function and arguments to callee-saved registers
                // 2. Extract tag from function value (fn_val & 0b111)
                // 3. If Function tag (0b100): untag to get code_ptr, args in x0-x7
                // 4. If Closure tag (0b101): load code_ptr from heap, closure in x0, user args in x1-x7
                // 5. Call the function
                // 6. Get result from x0

                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                if args.len() > 8 {
                    return Err("Call with more than 8 arguments not yet supported".to_string());
                }

                // Step 1: Save function to callee-saved register
                let fn_reg = self.get_physical_reg_for_irvalue(fn_val, false)?;
                let saved_fn_reg = 19;  // Use x19 for function value
                if fn_reg != saved_fn_reg {
                    self.emit_mov(saved_fn_reg, fn_reg);
                }

                // IMPORTANT: x0-x7 are never used by the allocator (which only uses x19-x28).
                // So we don't need intermediate saves - just track the argument source registers.
                // We'll move them to x0-x7 later (after tag checking).
                let mut arg_source_regs = Vec::new();
                for arg in args.iter() {
                    let arg_reg = self.get_physical_reg_for_irvalue(arg, false)?;
                    arg_source_regs.push(arg_reg);
                }

                // Step 2: Extract tag (fn_val & 0b111)
                let tag_reg = 16;  // Use x16 for tag (IP0 register, safe to use)
                // AND Xd, Xn, #0b111
                self.emit_and_imm(tag_reg, saved_fn_reg, 0b111);

                // Step 3: Check if Function (0b100) or Closure (0b101)
                // Compare tag with 0b100 (Function)
                self.emit_cmp_imm(tag_reg, 0b100);

                let is_function_label = self.new_label();
                self.emit_branch_cond(is_function_label.clone(), 0); // 0 = EQ (if tag == 0b100)

                // === Closure path (tag == 0b101) ===
                eprintln!("DEBUG: Call - emitting closure path");

                // Untag closure pointer (shift right by 3)
                let closure_ptr_reg = 17;  // x17 = untagged closure pointer (IP1 register, safe to use)
                // LSR Xd, Xn, #3 - Logical shift right by 3 = UBFM Xd, Xn, #3, #63
                // From Beagle: base = 0x53000000, sf=1, n=1, immr=3, imms=63
                let lsr_instruction = 0xD343FC00u32 | ((saved_fn_reg as u32) << 5) | (closure_ptr_reg as u32);
                self.code.push(lsr_instruction);

                // Load code_ptr from heap object field 1 (using closure_layout constants)
                use crate::gc_runtime::closure_layout;
                let code_ptr_reg = 18;  // x18 = code pointer (PR register, safe to use)
                self.emit_ldr_offset(code_ptr_reg, closure_ptr_reg, closure_layout::FIELD_1_CODE_PTR as i32);

                // Set up closure calling convention: x0 = closure object, user args in x1-x7
                self.emit_mov(0, saved_fn_reg);  // x0 = tagged closure pointer
                for (i, &src_reg) in arg_source_regs.iter().enumerate() {
                    if i + 1 != src_reg {  // Only move if source != destination
                        self.emit_mov(i + 1, src_reg);  // x1, x2, x3, etc.
                    }
                }

                let after_call_label = self.new_label();
                self.emit_jump(after_call_label.clone());

                // === Function path (tag == 0b100) ===
                self.emit_label(is_function_label);
                eprintln!("DEBUG: Call - emitting function path");

                // Untag function pointer to get code_ptr (shift right by 3)
                // LSR Xd, Xn, #3 - Logical shift right by 3 = UBFM Xd, Xn, #3, #63
                let lsr_instruction = 0xD343FC00u32 | ((saved_fn_reg as u32) << 5) | (code_ptr_reg as u32);
                self.code.push(lsr_instruction);

                // Set up normal calling convention: args in x0-x7
                for (i, &src_reg) in arg_source_regs.iter().enumerate() {
                    eprintln!("DEBUG: Call - moving arg {} from x{} to x{}", i, src_reg, i);
                    if i != src_reg {  // Only move if source != destination
                        self.emit_mov(i, src_reg);  // x0, x1, x2, etc.
                    }
                }

                // === Call the function ===
                self.emit_label(after_call_label);
                self.emit_blr(code_ptr_reg);

                // Step 6: Result is in x0
                eprintln!("DEBUG: Call - result will be moved from x0 to x{}", dst_reg);
                if dst_reg != 0 {
                    self.emit_mov(dst_reg, 0);
                }

                self.store_spill(dst_reg, dest_spill);
                eprintln!("DEBUG: Call - done, dst_reg=x{}, spill={:?}", dst_reg, dest_spill);
            }

            Instruction::Ret(value) => {
                eprintln!("DEBUG: Compiling Ret instruction with value: {:?}", value);
                // Move result to x0 (return register)
                let src_reg = self.get_physical_reg_for_irvalue(value, false)?;
                eprintln!("DEBUG: Ret - src_reg={}, moving to x0", src_reg);
                if src_reg != 0 {
                    self.emit_mov(0, src_reg);
                }
                // Return directly - don't jump to epilogue!
                // Functions have their own stack frames and should return immediately.
                // Only the top-level code needs the epilogue to restore frame before returning to trampoline.
                eprintln!("DEBUG: Emitting ret instruction");
                self.emit_ret();
            }
        }

        // Clear temporary registers after each instruction (like Beagle does)
        self.clear_temp_registers();

        Ok(())
    }

    /// Get physical register for an IR value
    /// If is_dest=true and value is Spill, returns temp register without loading
    /// If is_dest=false and value is Spill, loads from stack into temp register
    fn get_physical_reg_for_irvalue(&mut self, value: &IrValue, is_dest: bool) -> Result<usize, String> {
        match value {
            IrValue::Register(vreg) => {
                Ok(self.get_physical_reg(vreg))
            }
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
            _ => Err(format!("Expected register or spill, got {:?}", value)),
        }
    }

    fn get_physical_reg(&mut self, vreg: &VirtualRegister) -> usize {
        // IMPORTANT: Don't look up Argument registers in the map!
        // The allocator maps Argument(n) -> Temp(n) to represent physical xn,
        // but Temp(n) might also exist as a virtual register with a different allocation.
        // Instead, get Argument registers' physical location directly from their variant.
        match vreg {
            VirtualRegister::Argument(n) => *n,  // Arguments are already physical (x0-x7)
            _ => {
                // After register allocation, the IR has been rewritten with physical registers
                // Physical registers are represented as VirtualRegister::Temp(X) where X is the physical register number.
                //
                // If the register is in the allocation map, it's an original virtual register
                // and we need to look up its physical register.
                // If not in the map, it's already a physical register, so just use its index.
                self.register_map.get(vreg)
                    .map(|physical| physical.index())
                    .unwrap_or_else(|| vreg.index())  // Already physical, use index directly
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
            eprintln!("DEBUG store_spill: slot {} -> offset {}", stack_offset, offset);
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
            let target_pos = self.label_positions.get(label)
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
                return Err(format!("Unknown branch instruction at {}: {:08x}", code_index, instruction));
            }
        }

        // Apply ADR fixups
        for (code_index, label) in &self.pending_adr_fixups {
            let target_pos = self.label_positions.get(label)
                .ok_or_else(|| format!("Undefined label: {}", label))?;

            // Calculate offset in instructions
            let offset_instructions = (*target_pos as isize) - (*code_index as isize);

            // ADR uses byte offsets, so multiply by 4
            let byte_offset = offset_instructions * 4;

            eprintln!("DEBUG ADR fixup: code_index={}, label={}, target_pos={}, offset_instructions={}, byte_offset={}",
                      code_index, label, target_pos, offset_instructions, byte_offset);

            // Check if offset fits in 21-bit signed immediate
            if !(-1048576..=1048575).contains(&byte_offset) {
                return Err(format!("ADR offset too large: {}", byte_offset));
            }

            // ADR encoding: immlo (2 bits) | immhi (19 bits)
            let immlo = (byte_offset & 0x3) as u32;  // Lower 2 bits
            let immhi = ((byte_offset >> 2) & 0x7FFFF) as u32;  // Upper 19 bits

            // Patch the instruction
            let instruction = self.code[*code_index];
            self.code[*code_index] = (instruction & 0x9F00001F) | (immlo << 29) | (immhi << 5);

            eprintln!("DEBUG ADR: patched instruction at {} from {:08x} to {:08x}",
                      code_index, instruction, self.code[*code_index]);
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
    fn emit_orr_imm(&mut self, dst: usize, src: usize, imm: u32) {
        // For small immediates like 0b100, we can use the logical immediate encoding
        // ORR Xd, Xn, #imm
        // This is a simplified version for our specific use case (imm < 256)
        // Full ARM64 immediate encoding is complex, but for small values we can use:
        // 0xB2400000 | (imms << 10) | (src << 5) | dst
        // For imm=0b100 (4), we use pattern: immr=0, imms=2 (encodes value 0b111 >> (64-3))
        // Simplified: just use 0xB2400000 as base with imm encoding
        let instruction = if imm == 0b100 {
            // ORR X, X, #0b100 - special case for Function tag
            0xB2400C00u32 | ((src as u32) << 5) | (dst as u32)
        } else {
            panic!("ORR immediate only implemented for 0b100");
        };
        self.code.push(instruction);
    }

    /// Generate AND with immediate (for extracting tag bits)
    /// AND Xd, Xn, #imm
    fn emit_and_imm(&mut self, dst: usize, src: usize, imm: u32) {
        // AND Xd, Xn, #0b111 to extract the last 3 bits (tag)
        // ARM64 logical immediate encoding for 0b111 (3 ones)
        // immr=0, imms=2 (encodes a pattern of 3 ones)
        let instruction = if imm == 0b111 {
            // AND X, X, #0b111
            0x92400C00u32 | ((src as u32) << 5) | (dst as u32)
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
        let imm = imm as u64;  // Treat as unsigned for bitwise ops

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

    fn emit_add_imm(&mut self, dst: usize, src: usize, imm: i64) {
        // ADD Xd, Xn, #imm
        let instruction = 0x91000000 | ((imm as u32 & 0xFFF) << 10) | ((src as u32) << 5) | (dst as u32);
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

    fn emit_lsl_imm(&mut self, dst: usize, src: usize, shift: u32) {
        // LSL Xd, Xn, #shift (logical shift left)
        // This is actually UBFM (Unsigned Bitfield Move)
        // LSL #shift is: UBFM Xd, Xn, #(-shift mod 64), #(63-shift)
        let shift = shift & 0x3F; // 6 bits
        let immr = (64 - shift) & 0x3F;
        let imms = 63 - shift;
        let instruction = 0xD3400000 | (immr << 16) | (imms << 10) | ((src as u32) << 5) | (dst as u32);
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

    fn emit_cmp(&mut self, src1: usize, src2: usize) {
        // CMP Xn, Xm (compare - this is SUBS XZR, Xn, Xm)
        let instruction = 0xEB00001F | ((src2 as u32) << 16) | ((src1 as u32) << 5);
        self.code.push(instruction);
    }

    fn emit_cmp_imm(&mut self, src: usize, imm: i64) {
        // CMP Xn, #imm (compare - this is SUBS XZR, Xn, #imm)
        let imm12 = (imm & 0xFFF) as u32;  // 12-bit immediate
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

    fn emit_jump(&mut self, label: Label) {
        // B label (unconditional branch)
        let fixup_index = self.code.len();
        self.pending_fixups.push((fixup_index, label));

        // Emit placeholder
        let instruction = 0x14000000;
        self.code.push(instruction);
    }

    fn emit_str_offset(&mut self, src: usize, base: usize, offset: i32) {
        // STR Xt, [Xn, #offset]
        // Offset is in bytes, needs to be divided by 8 for encoding (unsigned 12-bit)
        let offset_scaled = (offset / 8) as u32;
        let instruction = 0xF9000000 | (offset_scaled << 10) | ((base as u32) << 5) | (src as u32);
        self.code.push(instruction);
    }

    fn emit_load_from_fp(&mut self, dst: usize, offset: i32) {
        // LDR Xd, [x29, #offset] with signed offset
        // Using LDUR for signed 9-bit offset
        let offset_bits = (offset as u32) & 0x1FF; // 9-bit signed
        let instruction = 0xF8400000 | (offset_bits << 12) | (29 << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_store_to_fp(&mut self, src: usize, offset: i32) {
        // STR Xt, [x29, #offset] with signed offset
        // Using STUR for signed 9-bit offset
        let offset_bits = (offset as u32) & 0x1FF; // 9-bit signed
        let instruction = 0xF8000000 | (offset_bits << 12) | (29 << 5) | (src as u32);
        eprintln!("DEBUG emit_store_to_fp: offset={}, offset_bits={:03x}, instruction={:08x}",
                  offset, offset_bits, instruction);
        self.code.push(instruction);
    }

    fn emit_stp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        // STP Xt, Xt2, [Xn, #offset]! (pre-index)
        // offset is in 8-byte units for STP, range -512 to 504
        let offset_scaled = ((offset & 0x7F) as u32) << 15;  // 7-bit signed offset
        let instruction = 0xA9800000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ldp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        // LDP Xt, Xt2, [Xn], #offset (post-index)
        // offset is in 8-byte units for LDP, range -512 to 504
        let offset_scaled = ((offset & 0x7F) as u32) << 15;  // 7-bit signed offset
        let instruction = 0xA8C00000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
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
    }

    /// Emit an external function call with automatic X30 (link register) preservation
    ///
    /// This helper automates the common pattern of:
    /// 1. Save X30 to stack
    /// 2. Load function address into register
    /// 3. Call function via BLR
    /// 4. Restore X30 from stack
    ///
    /// # Parameters
    /// - `target_fn`: Address of the external function to call
    /// - `_description`: Human-readable description for debugging (currently unused)
    fn emit_external_call(&mut self, target_fn: usize, _description: &str) {
        // Save X30 (link register) to stack
        // sub sp, sp, #16
        self.emit_sub_sp_imm(16);
        // str x30, [sp]
        self.emit_str_offset(30, 31, 0);  // x31 = sp

        // Load function address and call
        self.emit_mov_imm(15, target_fn as i64);  // Use x15 as temp
        self.emit_blr(15);

        // Restore X30 from stack
        // ldr x30, [sp]
        self.emit_ldr_offset(30, 31, 0);
        // add sp, sp, #16
        self.emit_add_sp_imm(16);
    }

    fn emit_adr(&mut self, dst: usize, label: Label) {
        // ADR Xd, <label>
        // Loads PC-relative address of label into Xd
        // Encoding: 0x10000000 | (immlo << 29) | (immhi << 5) | rd
        // immlo: 2 bits (bits 30-29)
        // immhi: 19 bits (bits 23-5)
        // rd: 5 bits (bits 4-0)
        //
        // For now, emit placeholder (offset 0) and record for fixup
        let fixup_index = self.code.len();
        self.pending_adr_fixups.push((fixup_index, label));

        // ADR with offset 0 (placeholder)
        let instruction = 0x10000000 | (dst as u32);
        self.code.push(instruction);
    }

    /// Execute the compiled code (for testing)
    ///
    /// Uses a trampoline to safely execute JIT code with proper stack management
    pub fn execute(&self) -> Result<i64, String> {
        eprintln!("DEBUG: execute() called with {} instructions", self.code.len());
        eprintln!("DEBUG: JIT code:");
        for (i, inst) in self.code.iter().enumerate() {
            eprintln!("  {:04x}: {:08x}", i * 4, inst);
        }
        let code_size = self.code.len() * 4;

        unsafe {
            // Allocate memory with mmap
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err("mmap failed".to_string());
            }

            // Copy code to executable memory
            let code_bytes = std::slice::from_raw_parts(
                self.code.as_ptr() as *const u8,
                code_size,
            );
            std::ptr::copy_nonoverlapping(code_bytes.as_ptr(), ptr as *mut u8, code_size);

            // Make memory executable
            if libc::mprotect(ptr, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                libc::munmap(ptr, code_size);
                return Err("mprotect failed".to_string());
            }

            // Clear instruction cache (required on ARM64)
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(ptr, code_size);
            }

            // Execute through trampoline for safety
            eprintln!("DEBUG: Creating trampoline...");
            let trampoline = Trampoline::new(64 * 1024); // 64KB stack
            eprintln!("DEBUG: Calling trampoline.execute()...");
            let result = trampoline.execute(ptr as *const u8);
            eprintln!("DEBUG: Trampoline returned: {}", result);

            // Explicitly drop trampoline before cleaning up JIT code
            drop(trampoline);

            // NOTE: We intentionally DO NOT call munmap here!
            // The JIT code needs to stay alive because function objects may hold
            // pointers to code in this block. In a production system, we would need
            // a proper JIT code cache with reference counting or garbage collection.
            // For now, we leak the memory (acceptable for a REPL/demo).
            //
            // libc::munmap(ptr, code_size);  // DISABLED - would free function code!

            // Return the tagged result (caller is responsible for untagging if needed)
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read;
    use crate::clojure_ast::analyze;
    use crate::compiler::Compiler;
    use crate::gc_runtime::GCRuntime;
    use std::sync::Arc;
    use std::cell::UnsafeCell;

    #[test]
    fn test_arm64_codegen_add() {
        let code = "(+ 1 2)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let result_reg = compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ 1 2)", machine_code.len());
        for (i, inst) in machine_code.iter().enumerate() {
            println!("  {:04x}: {:08x}", i * 4, inst);
        }

        let result = codegen.execute().unwrap();
        // Result is tagged: 3 << 3 = 24
        assert_eq!(result, 24);
    }

    #[test]
    fn test_arm64_codegen_nested() {
        let code = "(+ (* 2 3) 4)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let result_reg = compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ (* 2 3) 4)", machine_code.len());

        let result = codegen.execute().unwrap();
        // Result is tagged: 10 << 3 = 80
        assert_eq!(result, 80);
    }
}
