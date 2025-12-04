// LLVM Machine IR (MIR) schema
// Simplified representation of LLVM MIR for visualization
use serde::{Deserialize, Serialize};

/// LLVM MIR module (top-level container)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMModule {
    /// Module name
    pub name: String,

    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    #[serde(default)]
    pub target: String,

    /// Functions in this module
    pub functions: Vec<LLVMFunction>,
}

/// LLVM function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMFunction {
    /// Function name
    pub name: String,

    /// Function attributes (e.g., "nounwind", "readonly")
    #[serde(default)]
    pub attributes: Vec<String>,

    /// Basic blocks in this function
    pub blocks: Vec<LLVMBasicBlock>,
}

/// LLVM basic block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMBasicBlock {
    /// Block label/name
    pub label: String,

    /// Block attributes (e.g., "loop.header", "loop.latch")
    #[serde(default)]
    pub attributes: Vec<String>,

    /// Loop depth (0 = not in loop)
    #[serde(rename = "loopDepth", default)]
    pub loop_depth: u32,

    /// Predecessor block labels
    #[serde(default)]
    pub predecessors: Vec<String>,

    /// Successor block labels
    #[serde(default)]
    pub successors: Vec<String>,

    /// Instructions in this block
    pub instructions: Vec<LLVMInstruction>,
}

/// LLVM instruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMInstruction {
    /// Result register (e.g., "%3")
    #[serde(default)]
    pub result: Option<String>,

    /// Opcode (e.g., "add", "load", "br")
    pub opcode: String,

    /// Type (e.g., "i32", "i64*", "void")
    #[serde(rename = "type")]
    pub ty: String,

    /// Operands (e.g., ["%1", "%2"])
    #[serde(default)]
    pub operands: Vec<String>,

    /// Instruction attributes (e.g., ["nsw", "nuw"])
    #[serde(default)]
    pub attributes: Vec<String>,

    /// Debug metadata
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

impl LLVMModule {
    /// Create a simple example module for testing
    pub fn example_loop() -> Self {
        Self {
            name: "example".to_string(),
            target: "x86_64-unknown-linux-gnu".to_string(),
            functions: vec![LLVMFunction {
                name: "simple_loop".to_string(),
                attributes: vec!["nounwind".to_string()],
                blocks: vec![
                    LLVMBasicBlock {
                        label: "entry".to_string(),
                        attributes: vec![],
                        loop_depth: 0,
                        predecessors: vec![],
                        successors: vec!["loop.header".to_string()],
                        instructions: vec![
                            LLVMInstruction {
                                result: Some("%n".to_string()),
                                opcode: "load".to_string(),
                                ty: "i32".to_string(),
                                operands: vec!["%n.addr".to_string()],
                                attributes: vec![],
                                metadata: None,
                            },
                            LLVMInstruction {
                                result: None,
                                opcode: "br".to_string(),
                                ty: "void".to_string(),
                                operands: vec!["label %loop.header".to_string()],
                                attributes: vec![],
                                metadata: None,
                            },
                        ],
                    },
                    LLVMBasicBlock {
                        label: "loop.header".to_string(),
                        attributes: vec!["loop.header".to_string()],
                        loop_depth: 1,
                        predecessors: vec!["entry".to_string(), "loop.body".to_string()],
                        successors: vec!["loop.body".to_string(), "exit".to_string()],
                        instructions: vec![
                            LLVMInstruction {
                                result: Some("%i".to_string()),
                                opcode: "phi".to_string(),
                                ty: "i32".to_string(),
                                operands: vec![
                                    "[ 0, %entry ]".to_string(),
                                    "[ %i.next, %loop.body ]".to_string(),
                                ],
                                attributes: vec![],
                                metadata: None,
                            },
                            LLVMInstruction {
                                result: Some("%cmp".to_string()),
                                opcode: "icmp".to_string(),
                                ty: "i1".to_string(),
                                operands: vec!["slt".to_string(), "%i".to_string(), "%n".to_string()],
                                attributes: vec![],
                                metadata: None,
                            },
                            LLVMInstruction {
                                result: None,
                                opcode: "br".to_string(),
                                ty: "void".to_string(),
                                operands: vec![
                                    "%cmp".to_string(),
                                    "label %loop.body".to_string(),
                                    "label %exit".to_string(),
                                ],
                                attributes: vec![],
                                metadata: None,
                            },
                        ],
                    },
                    LLVMBasicBlock {
                        label: "loop.body".to_string(),
                        attributes: vec!["loop.latch".to_string()],
                        loop_depth: 1,
                        predecessors: vec!["loop.header".to_string()],
                        successors: vec!["loop.header".to_string()],
                        instructions: vec![
                            LLVMInstruction {
                                result: Some("%i.next".to_string()),
                                opcode: "add".to_string(),
                                ty: "i32".to_string(),
                                operands: vec!["%i".to_string(), "1".to_string()],
                                attributes: vec!["nsw".to_string()],
                                metadata: None,
                            },
                            LLVMInstruction {
                                result: None,
                                opcode: "br".to_string(),
                                ty: "void".to_string(),
                                operands: vec!["label %loop.header".to_string()],
                                attributes: vec![],
                                metadata: None,
                            },
                        ],
                    },
                    LLVMBasicBlock {
                        label: "exit".to_string(),
                        attributes: vec![],
                        loop_depth: 0,
                        predecessors: vec!["loop.header".to_string()],
                        successors: vec![],
                        instructions: vec![LLVMInstruction {
                            result: None,
                            opcode: "ret".to_string(),
                            ty: "void".to_string(),
                            operands: vec![],
                            attributes: vec![],
                            metadata: None,
                        }],
                    },
                ],
            }],
        }
    }
}
