// Our simplified IR based on Beagle's design
// We'll copy and adapt the parts we need

use std::cmp::Ordering;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum Condition {
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// Target for unified CallWithSaves instruction
/// Replaces separate ExternalCallWithSaves, CallDirectWithSaves, and dynamic CallWithSaves
#[derive(Debug, Clone, PartialEq)]
pub enum CallTarget {
    /// Known external function address (trampolines)
    /// Args go in x0-x7
    External(usize),

    /// Dynamic dispatch - must check tag at runtime
    /// Handles: raw functions (0b100), closures (0b101), IFn objects (other)
    Dynamic(IrValue),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum VirtualRegister {
    Temp(usize),     // Compiler-generated temporary registers
    Argument(usize), // Function argument registers (x0-x7)
}

impl VirtualRegister {
    /// Create a new temporary register (for backwards compatibility)
    #[allow(dead_code)]
    pub fn new(index: usize) -> Self {
        VirtualRegister::Temp(index)
    }

    /// Get the index (for display/debugging)
    pub fn index(&self) -> usize {
        match self {
            VirtualRegister::Temp(n) => *n,
            VirtualRegister::Argument(n) => *n,
        }
    }

    /// Check if this is an argument register
    #[allow(dead_code)]
    pub fn is_argument(&self) -> bool {
        matches!(self, VirtualRegister::Argument(_))
    }

    /// Display name for debugging
    #[allow(dead_code)]
    pub fn display_name(&self) -> String {
        match self {
            VirtualRegister::Temp(n) => format!("v{}", n),
            VirtualRegister::Argument(n) => format!("v_arg{}", n),
        }
    }
}

impl Ord for VirtualRegister {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by type first (Argument < Temp), then by index
        match (self, other) {
            (VirtualRegister::Argument(a), VirtualRegister::Argument(b)) => a.cmp(b),
            (VirtualRegister::Temp(a), VirtualRegister::Temp(b)) => a.cmp(b),
            (VirtualRegister::Argument(_), VirtualRegister::Temp(_)) => Ordering::Less,
            (VirtualRegister::Temp(_), VirtualRegister::Argument(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for VirtualRegister {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum IrValue {
    Register(VirtualRegister),
    TaggedConstant(isize), // For tagged integers
    RawConstant(i64),      // For untagged values (pointers, lengths, etc.)
    True,
    False,
    Null,
    Spill(VirtualRegister, usize), // Spilled register with stack offset
    /// FramePointer - represents x29 (FP) in ARM64
    /// Used for passing stack pointer to allocation trampolines for GC safety
    FramePointer,
}

pub type Label = String;

#[derive(Debug, Clone)]
pub enum Instruction {
    // Integer Arithmetic (work on untagged values)
    AddInt(IrValue, IrValue, IrValue), // dst, src1, src2
    Sub(IrValue, IrValue, IrValue),
    Mul(IrValue, IrValue, IrValue),
    Div(IrValue, IrValue, IrValue),

    // Bitwise operations (work on untagged values)
    BitAnd(IrValue, IrValue, IrValue),        // dst, src1, src2
    BitOr(IrValue, IrValue, IrValue),         // dst, src1, src2
    BitXor(IrValue, IrValue, IrValue),        // dst, src1, src2
    BitNot(IrValue, IrValue),                 // dst, src
    BitShiftLeft(IrValue, IrValue, IrValue),  // dst, src, amount
    BitShiftRight(IrValue, IrValue, IrValue), // dst, src, amount (arithmetic/signed)
    UnsignedBitShiftRight(IrValue, IrValue, IrValue), // dst, src, amount (logical/unsigned)

    // Memory operations (for heap access)
    /// HeapStore(ptr, offset, value) - *(ptr + offset*8) = value - store 64-bit value to heap
    HeapStore(IrValue, i32, IrValue),

    // Float Arithmetic (work on raw f64 bits in registers)
    AddFloat(IrValue, IrValue, IrValue), // dst, src1, src2 - f64 addition
    SubFloat(IrValue, IrValue, IrValue),
    MulFloat(IrValue, IrValue, IrValue),
    DivFloat(IrValue, IrValue, IrValue),
    IntToFloat(IrValue, IrValue), // dst, src - convert int to float

    // Float heap operations (floats are heap-allocated)
    LoadFloat(IrValue, IrValue), // dst, src - load f64 from heap float pointer

    // Type tag extraction
    GetTag(IrValue, IrValue), // dst, src - extract tag bits (last 3 bits)

    // Comparison (produces boolean in register)
    Compare(IrValue, IrValue, IrValue, Condition), // dst, src1, src2, condition

    // Type tagging/untagging
    Tag(IrValue, IrValue, IrValue), // dst, value, tag
    Untag(IrValue, IrValue),        // dst, tagged_value

    // Constants
    LoadConstant(IrValue, IrValue),

    // Note: LoadVarBySymbol, LoadVarBySymbolDynamic, StoreVarBySymbol, and EnsureVarBySymbol
    // have been converted to builtin function calls (runtime.builtin/load-var-by-symbol, etc.)
    // See builtins.rs for implementations.
    LoadTrue(IrValue),
    LoadFalse(IrValue),

    // Control flow
    Label(Label),
    Jump(Label),
    JumpIf(Label, Condition, IrValue, IrValue), // label, condition, val1, val2

    // NOTE: Recur and RecurWithSaves have been removed.
    // Recur is now lowered to Assign + Jump in the compiler (compile_recur).
    // The parallel assignment semantics are handled by:
    // 1. Assigning new values to temp registers
    // 2. Assigning temps to binding registers
    // 3. Jumping to the loop label

    // Assignment
    Assign(IrValue, IrValue), // dst, src

    // Function operations
    MakeFunctionPtr(IrValue, usize, Vec<IrValue>), // MakeFunctionPtr(dst, code_ptr, closure_values) - create function with raw code pointer
    LoadClosure(IrValue, IrValue, usize), // LoadClosure(dst, fn_obj, index) - load closure variable
    Call(IrValue, IrValue, Vec<IrValue>), // Call(dst, fn, args) - invoke function

    /// CallWithSaves(dst, target, args, saves) - unified call with register preservation
    /// target: CallTarget specifying how to call (External, Direct, or Dynamic)
    /// args: arguments to pass
    /// saves: registers to save/restore across the call
    CallWithSaves(IrValue, CallTarget, Vec<IrValue>, Vec<IrValue>),

    // Multi-arity function operations
    /// MakeMultiArityFn(dst, arities, variadic_min, variadic_index, closure_values)
    /// arities: Vec of (param_count, code_ptr) pairs
    /// variadic_min: If Some, the minimum arg count for variadic dispatch
    /// variadic_index: If Some, the index of the variadic arity in the arities table
    MakeMultiArityFn(IrValue, Vec<(usize, usize)>, Option<usize>, Option<usize>, Vec<IrValue>),

    /// LoadClosureMultiArity(dst, fn_obj, arity_count, index)
    /// Load closure variable from multi-arity function (needs arity_count to compute offset)
    LoadClosureMultiArity(IrValue, IrValue, usize, usize),

    // Variadic argument operations
    /// CollectRestArgs(dst, fixed_count, param_offset)
    /// Collects excess args into a list for variadic functions.
    /// - At runtime, x9 contains the total argument count
    /// - fixed_count: number of fixed parameters in this arity
    /// - param_offset: offset for user args (1 for closures, 0 for raw functions)
    ///
    /// The instruction computes: excess_count = x9 - fixed_count
    /// Then collects args from x(param_offset + fixed_count) onwards into a list.
    CollectRestArgs(IrValue, usize, usize),

    // DefType operations
    // NOTE: MakeType has been refactored out - deftype construction now uses:
    // 1. ExternalCall to trampoline_allocate_type_object_raw (returns untagged pointer)
    // 2. HeapStore instructions to write each field
    // 3. Tag instruction to add HeapObject tag (0b110)
    // This follows the Beagle pattern of decomposing complex operations into primitives.

    // NOTE: LoadTypeField and StoreTypeField have been refactored out.
    // Field access now uses ExternalCall to trampoline_load_type_field_by_symbol with
    // pre-interned symbol IDs, eliminating stack-based string passing.

    // Return
    Ret(IrValue),

    // Debug
    Breakpoint, // BRK #0 - trap for debugger

    // NOTE: Println has been refactored out - now uses ExternalCall to trampoline_println_regs
    // The trampoline takes (count, v0, v1, v2, v3, v4, v5, v6) in registers, up to 7 values.

    // Note: LoadKeyword has been converted to a builtin function call (runtime.builtin/load-keyword)
    // See builtins.rs for implementation.

    // Exception handling
    /// PushExceptionHandler(catch_label, exception_slot_index) - setup exception handler
    /// Saves SP/FP/LR and catch label; if exception occurs, jumps to catch_label
    /// with exception stored at the given stack slot index
    /// The slot index is pre-allocated by the compiler to ensure proper stack frame sizing
    PushExceptionHandler(Label, usize),

    /// PopExceptionHandler - remove exception handler (normal exit from try)
    PopExceptionHandler,

    /// Throw(exception_value) - throw exception, never returns
    /// Pops handler, stores exception, restores SP/FP/LR, jumps to catch
    Throw(IrValue),

    /// LoadExceptionLocal(dest, exception_slot_index) - load exception value from stack after catch
    /// Loads the exception from the pre-allocated stack slot into dest register
    LoadExceptionLocal(IrValue, usize),

    // Assertion checking (pre/post conditions)
    /// AssertPre(condition_value, condition_index)
    /// Checks if condition_value is truthy; if falsy, throws AssertionError
    /// condition_index is used in error message
    AssertPre(IrValue, usize),

    /// AssertPost(condition_value, condition_index)
    /// Like AssertPre but for post-conditions
    AssertPost(IrValue, usize),

    // External calls (for trampolines)
    /// ExternalCall(dst, func_addr, args) - call a known function address
    /// func_addr is an untagged constant (the trampoline address)
    /// Used for protocol lookup, var access, etc.
    /// NOTE: For calls needing register saves, use CallWithSaves with CallTarget::External
    ExternalCall(IrValue, usize, Vec<IrValue>),

    // Local variable operations (stack-based)
    /// StoreLocal(slot, value) - store value to local slot on stack
    /// Used at function entry to save arguments to stack
    /// slot is 0-based index, codegen converts to FP-relative offset
    StoreLocal(usize, IrValue),

    /// LoadLocal(dst, slot) - load from local slot to register
    /// Used when accessing function parameters that were stored as locals
    LoadLocal(IrValue, usize),
}

/// IR builder - helps construct IR instructions
pub struct IrBuilder {
    next_temp_register: usize,
    #[allow(dead_code)]
    next_argument_register: usize,
    next_label: usize,
    pub instructions: Vec<Instruction>,
    /// Number of stack slots reserved for exception handling
    /// These are allocated before register allocation runs, ensuring
    /// the stack frame is sized correctly in the prologue
    pub reserved_exception_slots: usize,
    /// Number of local variable slots (for storing arguments)
    /// Following Beagle's pattern: arguments are stored to locals at function entry
    pub num_locals: usize,
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IrBuilder {
    pub fn new() -> Self {
        IrBuilder {
            // Temp registers start at 0 (no more conflicts!)
            next_temp_register: 0,
            next_argument_register: 0,
            next_label: 0,
            instructions: Vec::new(),
            reserved_exception_slots: 0,
            num_locals: 0,
        }
    }

    /// Reset the builder for a new compilation unit
    /// Used when compiling multiple expressions in sequence (e.g., TopLevelDo)
    pub fn reset(&mut self) {
        self.next_temp_register = 0;
        self.next_argument_register = 0;
        self.next_label = 0;
        self.instructions.clear();
        self.reserved_exception_slots = 0;
        self.num_locals = 0;
    }

    /// Allocate a stack slot for exception handling
    /// Returns the slot index (0-based, will be converted to FP-relative offset in codegen)
    pub fn allocate_exception_slot(&mut self) -> usize {
        let slot = self.reserved_exception_slots;
        self.reserved_exception_slots += 1;
        slot
    }

    /// Allocate a local variable slot on the stack
    /// Used for storing function arguments at entry (Beagle pattern)
    /// Returns the slot index (0-based, codegen converts to FP-relative offset)
    pub fn allocate_local(&mut self) -> usize {
        let slot = self.num_locals;
        self.num_locals += 1;
        slot
    }

    /// Create a new temporary register
    pub fn new_register(&mut self) -> IrValue {
        let reg = VirtualRegister::Temp(self.next_temp_register);
        self.next_temp_register += 1;
        IrValue::Register(reg)
    }

    /// Assign a value to a register, reusing if already a register.
    /// This is a key optimization from Beagle that reduces register pressure.
    /// If the value is already in a register, returns that register.
    /// Otherwise allocates a new register and emits a LoadConstant instruction.
    pub fn assign_new(&mut self, val: IrValue) -> IrValue {
        if let IrValue::Register(_) = val {
            // Value is already a register - reuse it!
            return val;
        }
        // Need to load into a new register
        let register = self.new_register();
        self.emit(Instruction::LoadConstant(register, val));
        register
    }

    /// Create a new argument register (for function parameters)
    #[allow(dead_code)]
    pub fn new_argument_register(&mut self) -> IrValue {
        let reg = VirtualRegister::Argument(self.next_argument_register);
        self.next_argument_register += 1;
        IrValue::Register(reg)
    }

    pub fn new_label(&mut self) -> Label {
        let label = format!("L{}", self.next_label);
        self.next_label += 1;
        label
    }

    pub fn emit(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    /// Take the instructions without consuming the builder, clearing the buffer
    /// Also resets the reserved_exception_slots counter
    pub fn take_instructions(&mut self) -> Vec<Instruction> {
        self.reserved_exception_slots = 0;
        std::mem::take(&mut self.instructions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_builder() {
        let mut builder = IrBuilder::new();

        let r0 = builder.new_register();
        let r1 = builder.new_register();
        let r2 = builder.new_register();

        builder.emit(Instruction::LoadConstant(r0, IrValue::TaggedConstant(42)));
        builder.emit(Instruction::LoadConstant(r1, IrValue::TaggedConstant(10)));
        builder.emit(Instruction::AddInt(r2, r0, r1));

        let instructions = builder.take_instructions();
        assert_eq!(instructions.len(), 3);
    }
}
