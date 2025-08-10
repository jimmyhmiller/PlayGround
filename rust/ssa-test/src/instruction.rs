use crate::ast::{BinaryOperator, UnaryOperator};

#[derive(Debug, Clone, PartialEq)]
pub struct Variable(pub String);

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Literal(i32),
    Var(Variable),
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
    
    // Phi function for SSA
    Phi {
        dest: Variable,
        sources: Vec<(BlockId, Value)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub terminator: Option<Instruction>, // Jump or ConditionalJump
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub blocks: Vec<BasicBlock>,
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

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        BasicBlock {
            id,
            instructions: Vec::new(),
            terminator: None,
        }
    }
    
    pub fn add_instruction(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }
    
    pub fn set_terminator(&mut self, instr: Instruction) {
        self.terminator = Some(instr);
    }
}

impl Function {
    pub fn new(entry: BlockId) -> Self {
        Function {
            blocks: Vec::new(),
            entry,
        }
    }
    
    pub fn add_block(&mut self, block: BasicBlock) {
        self.blocks.push(block);
    }
}