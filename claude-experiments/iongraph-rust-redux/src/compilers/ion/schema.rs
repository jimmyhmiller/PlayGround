// Port of iongraph.ts - Ion-specific schema definitions
use serde::{Deserialize, Serialize};

pub const CURRENT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockPtr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockID(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InsPtr(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonJSON {
    pub version: u32,
    #[serde(default)]
    pub function: Option<String>,
    #[serde(default)]
    pub functions: Vec<Function>,
    #[serde(default)]
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pass {
    pub name: String,
    pub mir: Option<MIRPass>,
    pub lir: Option<LIRPass>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIRPass {
    pub blocks: Vec<MIRBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIRPass {
    pub blocks: Vec<LIRBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIRBlock {
    pub ptr: BlockPtr,
    pub id: u32,
    #[serde(rename = "loopDepth", default)]
    pub loop_depth: u32,
    pub attributes: Vec<String>,
    pub predecessors: Vec<BlockPtr>,
    pub successors: Vec<BlockPtr>,
    pub instructions: Vec<MIRInstruction>,
    #[serde(rename = "resumePoint", default)]
    pub resume_point: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIRBlock {
    pub ptr: BlockPtr,
    pub id: u32,
    #[serde(rename = "loopDepth", default)]
    pub loop_depth: u32,
    #[serde(default)]
    pub attributes: Vec<String>,
    #[serde(default)]
    pub predecessors: Vec<BlockPtr>,
    #[serde(default)]
    pub successors: Vec<BlockPtr>,
    pub instructions: Vec<LIRInstruction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIRInstruction {
    pub ptr: InsPtr,
    pub id: u32,
    pub opcode: String,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub attributes: Option<Vec<String>>,
    #[serde(default)]
    pub inputs: Vec<u32>,
    #[serde(default)]
    pub uses: Vec<u32>,
    #[serde(default)]
    pub mem_inputs: Vec<serde_json::Value>, // TODO: proper type
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIRInstruction {
    pub ptr: InsPtr,
    pub id: u32,
    pub mir_ptr: Option<u32>,
    pub opcode: String,
    pub type_: Option<String>,
    pub attributes: Option<Vec<String>>,
    #[serde(default)]
    pub defs: Vec<u32>,
}
