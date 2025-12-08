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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum VirtualRegister {
    Temp(usize),      // Compiler-generated temporary registers
    Argument(usize),  // Function argument registers (x0-x7)
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
    TaggedConstant(isize),  // For tagged integers
    True,
    False,
    Null,
    Spill(VirtualRegister, usize),  // Spilled register with stack offset
}

pub type Label = String;

#[derive(Debug, Clone)]
pub enum Instruction {
    // Arithmetic (work on untagged values)
    AddInt(IrValue, IrValue, IrValue),  // dst, src1, src2
    Sub(IrValue, IrValue, IrValue),
    Mul(IrValue, IrValue, IrValue),
    Div(IrValue, IrValue, IrValue),

    // Comparison (produces boolean in register)
    Compare(IrValue, IrValue, IrValue, Condition),  // dst, src1, src2, condition

    // Type tagging/untagging
    Tag(IrValue, IrValue, IrValue),    // dst, value, tag
    Untag(IrValue, IrValue),           // dst, tagged_value

    // Constants
    LoadConstant(IrValue, IrValue),
    LoadVar(IrValue, IrValue),  // LoadVar(dest_reg, var_ptr) - direct load for non-dynamic vars
    LoadVarDynamic(IrValue, IrValue),  // LoadVarDynamic(dest_reg, var_ptr) - trampoline call for ^:dynamic vars
    StoreVar(IrValue, IrValue), // StoreVar(var_ptr, value_reg) - stores value into var at runtime
    LoadTrue(IrValue),
    LoadFalse(IrValue),

    // Dynamic var bindings
    PushBinding(IrValue, IrValue),  // PushBinding(var_ptr, value) - push thread-local binding
    PopBinding(IrValue),            // PopBinding(var_ptr) - pop thread-local binding
    SetVar(IrValue, IrValue),       // SetVar(var_ptr, value) - modify thread-local binding (for set!)

    // Control flow
    Label(Label),
    Jump(Label),
    JumpIf(Label, Condition, IrValue, IrValue),  // label, condition, val1, val2

    // Assignment
    Assign(IrValue, IrValue),  // dst, src

    // Function operations
    MakeFunctionPtr(IrValue, usize, Vec<IrValue>), // MakeFunctionPtr(dst, code_ptr, closure_values) - create function with raw code pointer
    LoadClosure(IrValue, IrValue, usize),   // LoadClosure(dst, fn_obj, index) - load closure variable
    Call(IrValue, IrValue, Vec<IrValue>),   // Call(dst, fn, args) - invoke function
    CallWithSaves(IrValue, IrValue, Vec<IrValue>, Vec<IrValue>),  // CallWithSaves(dst, fn, args, saves) - call with register preservation

    // DefType operations
    /// MakeType(dst, type_id, field_values) - create deftype instance
    MakeType(IrValue, usize, Vec<IrValue>),
    /// LoadTypeField(dst, obj, field_name) - load field from deftype instance
    /// Field name is used for runtime lookup (future: inline caching)
    LoadTypeField(IrValue, IrValue, String),

    // Return
    Ret(IrValue),
}

/// IR builder - helps construct IR instructions
pub struct IrBuilder {
    next_temp_register: usize,
    #[allow(dead_code)]
    next_argument_register: usize,
    next_label: usize,
    pub instructions: Vec<Instruction>,
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
        }
    }

    /// Create a new temporary register
    pub fn new_register(&mut self) -> IrValue {
        let reg = VirtualRegister::Temp(self.next_temp_register);
        self.next_temp_register += 1;
        IrValue::Register(reg)
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
        // if let Instruction::MakeFunction(_, _, ref closure_values) = instruction {
        //     eprintln!("DEBUG IrBuilder::emit - MakeFunction with closure_values = {:?}", closure_values);
        // }
        self.instructions.push(instruction);
    }

    /// Take the instructions without consuming the builder, clearing the buffer
    pub fn take_instructions(&mut self) -> Vec<Instruction> {
        let instructions = std::mem::take(&mut self.instructions);
        // for inst in &instructions {
        //     if let Instruction::MakeFunction(_, _, closure_values) = inst {
        //         eprintln!("DEBUG IrBuilder::take_instructions - MakeFunction with closure_values = {:?}", closure_values);
        //     }
        // }
        instructions
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
