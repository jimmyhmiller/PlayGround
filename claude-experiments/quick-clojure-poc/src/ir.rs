// Our simplified IR based on Beagle's design
// We'll copy and adapt the parts we need

use std::cmp::Ordering;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VirtualRegister {
    pub index: usize,
    pub is_argument: bool,
}

impl VirtualRegister {
    pub fn new(index: usize) -> Self {
        VirtualRegister {
            index,
            is_argument: false,
        }
    }

    pub fn arg(index: usize) -> Self {
        VirtualRegister {
            index,
            is_argument: true,
        }
    }
}

impl Ord for VirtualRegister {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

impl PartialOrd for VirtualRegister {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Copy, Clone)]
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
    LoadVar(IrValue, IrValue),  // LoadVar(dest_reg, var_ptr) - dereferences var at runtime, checks dynamic bindings
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

    // Return
    Ret(IrValue),
}

/// IR builder - helps construct IR instructions
pub struct IrBuilder {
    next_register: usize,
    next_label: usize,
    pub instructions: Vec<Instruction>,
}

impl IrBuilder {
    pub fn new() -> Self {
        IrBuilder {
            next_register: 0,
            next_label: 0,
            instructions: Vec::new(),
        }
    }

    pub fn new_register(&mut self) -> IrValue {
        let reg = VirtualRegister::new(self.next_register);
        self.next_register += 1;
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

    pub fn finish(self) -> Vec<Instruction> {
        self.instructions
    }

    /// Take the instructions without consuming the builder, clearing the buffer
    pub fn take_instructions(&mut self) -> Vec<Instruction> {
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
        builder.emit(Instruction::Ret(r2));

        let instructions = builder.finish();
        assert_eq!(instructions.len(), 4);
    }
}
