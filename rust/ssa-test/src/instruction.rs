use crate::ast::{BinaryOperator, UnaryOperator};

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct Variable(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub struct PhiId(pub usize);


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhiReference {
    Instruction {
        block_id: BlockId,
        instruction_offset: usize,
    },
    Phi(PhiId),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Phi {
    pub id: PhiId,
    pub block_id: BlockId,
    pub operands: Vec<Value>,
    pub uses: Vec<PhiReference>,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum Value {
    Literal(i32),
    Var(Variable),
    Phi(PhiId),
    Undefined,
}
impl Value {
    pub fn new_phi(phi_id: PhiId) -> Self {
        Value::Phi(phi_id)
    }

    pub fn get_phi_id(&self) -> PhiId {
        match self {
            Value::Phi(phi_id) => *phi_id,
            _ => panic!("Value is not a Phi"),
        }
    }

    pub fn is_phi(&self) -> bool {
        matches!(self, Value::Phi(_))
    }

    pub fn is_same_phi(&self, phi_id: PhiId) -> bool {
        match self {
            Value::Phi(id) => *id == phi_id,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Basic assignments
    Assign {
        dest: Variable,
        value: Value,
    },

    // Binary operations
    BinaryOp {
        dest: Variable,
        left: Value,
        op: BinaryOperator,
        right: Value,
    },

    // Unary operations
    UnaryOp {
        dest: Variable,
        op: UnaryOperator,
        operand: Value,
    },

    // Control flow
    Jump {
        target: BlockId,
    },

    ConditionalJump {
        condition: Value,
        true_target: BlockId,
        false_target: BlockId,
    },

    // Print instruction for debugging
    Print {
        value: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub predecessors: Vec<BlockId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub blocks: Vec<Block>,
    pub entry: BlockId,
}

impl Variable {
    pub fn new(name: &str) -> Self {
        Variable(name.to_string())
    }

    pub fn temp(id: usize) -> Self {
        Variable(format!("temp_{}", id))
    }
}

impl Value {
    pub fn literal(val: i32) -> Self {
        Value::Literal(val)
    }

    pub fn var(name: &str) -> Self {
        Value::Var(Variable::new(name))
    }

    pub fn temp(id: usize) -> Self {
        Value::Var(Variable::temp(id))
    }
}

impl Block {
    pub fn new(id: BlockId) -> Self {
        Block {
            id,
            instructions: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    pub fn add_instruction(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }
}

impl Function {
    pub fn new(entry: BlockId) -> Self {
        Function {
            blocks: Vec::new(),
            entry,
        }
    }

    pub fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }
}
