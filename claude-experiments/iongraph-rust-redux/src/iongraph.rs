// Port of iongraph.ts
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

pub fn migrate(mut ion_json: serde_json::Value) -> IonJSON {
    // Check if we need to migrate from version 0
    if ion_json.get("version").is_none() {
        // This is version 0, migrate to version 1
        ion_json["version"] = serde_json::Value::Number(1.into());

        // Migrate passes
        if let Some(passes) = ion_json.get_mut("passes") {
            if let Some(passes_array) = passes.as_array_mut() {
                for pass in passes_array {
                    // Migrate MIR blocks
                    if let Some(mir) = pass.get_mut("mir") {
                        if let Some(mir_array) = mir.as_array_mut() {
                            for block in mir_array {
                                migrate_block_v0_to_v1(block);
                            }
                        }
                    }

                    // Migrate LIR blocks
                    if let Some(lir) = pass.get_mut("lir") {
                        if let Some(lir_array) = lir.as_array_mut() {
                            for block in lir_array {
                                migrate_block_v0_to_v1(block);
                            }
                        }
                    }
                }
            }
        }
    }

    serde_json::from_value(ion_json).expect("Failed to deserialize migrated IonJSON")
}

fn migrate_block_v0_to_v1(block: &mut serde_json::Value) {
    // In version 0, block IDs were numbers
    // In version 1, we use block numbers directly as IDs
    // The main change is handling predecessors/successors as BlockPtrs

    if let Some(obj) = block.as_object_mut() {
        // Migrate predecessors
        if let Some(preds) = obj.get("predecessors") {
            if let Some(preds_array) = preds.as_array() {
                let migrated: Vec<_> = preds_array
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect();
                obj.insert("predecessors".to_string(), serde_json::json!(migrated));
            }
        }

        // Migrate successors
        if let Some(succs) = obj.get("successors") {
            if let Some(succs_array) = succs.as_array() {
                let migrated: Vec<_> = succs_array
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect();
                obj.insert("successors".to_string(), serde_json::json!(migrated));
            }
        }
    }
}
