// Port of iongraph.ts - Ion-specific schema definitions

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::json_compat::Value;

#[cfg(not(feature = "serde"))]
use crate::json::{FromJson, JsonObjectExt, ParseError, ToJson};

pub const CURRENT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BlockPtr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BlockID(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InsPtr(pub u32);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IonJSON {
    pub version: u32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub function: Option<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub functions: Vec<Function>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Function {
    pub name: String,
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pass {
    pub name: String,
    pub mir: Option<MIRPass>,
    pub lir: Option<LIRPass>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MIRPass {
    pub blocks: Vec<MIRBlock>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LIRPass {
    pub blocks: Vec<LIRBlock>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MIRBlock {
    pub ptr: BlockPtr,
    pub id: u32,
    #[cfg_attr(feature = "serde", serde(rename = "loopDepth", default))]
    pub loop_depth: u32,
    pub attributes: Vec<String>,
    pub predecessors: Vec<BlockPtr>,
    pub successors: Vec<BlockPtr>,
    pub instructions: Vec<MIRInstruction>,
    #[cfg_attr(feature = "serde", serde(rename = "resumePoint", default))]
    pub resume_point: Option<Value>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LIRBlock {
    pub ptr: BlockPtr,
    pub id: u32,
    #[cfg_attr(feature = "serde", serde(rename = "loopDepth", default))]
    pub loop_depth: u32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub attributes: Vec<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub predecessors: Vec<BlockPtr>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub successors: Vec<BlockPtr>,
    pub instructions: Vec<LIRInstruction>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MIRInstruction {
    pub ptr: InsPtr,
    pub id: u32,
    pub opcode: String,
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub type_: Option<String>,
    pub attributes: Option<Vec<String>>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub inputs: Vec<u32>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub uses: Vec<u32>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub mem_inputs: Vec<Value>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LIRInstruction {
    pub ptr: InsPtr,
    pub id: u32,
    pub mir_ptr: Option<u32>,
    pub opcode: String,
    pub type_: Option<String>,
    pub attributes: Option<Vec<String>>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub defs: Vec<u32>,
}

// ============================================================================
// FromJson implementations for non-serde builds
// ============================================================================

#[cfg(not(feature = "serde"))]
impl FromJson for BlockPtr {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(BlockPtr(u32::from_json(value)?))
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for BlockID {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(BlockID(u32::from_json(value)?))
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for InsPtr {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(InsPtr(u32::from_json(value)?))
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for IonJSON {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(IonJSON {
            version: value.get_field("version")?,
            function: value.get_field_opt("function")?,
            functions: value.get_field_or_default("functions")?,
            passes: value.get_field_or_default("passes")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for Function {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(Function {
            name: value.get_field("name")?,
            passes: value.get_field("passes")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for Pass {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(Pass {
            name: value.get_field("name")?,
            mir: value.get_field_opt("mir")?,
            lir: value.get_field_opt("lir")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for MIRPass {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(MIRPass {
            blocks: value.get_field("blocks")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for LIRPass {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(LIRPass {
            blocks: value.get_field("blocks")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for MIRBlock {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(MIRBlock {
            ptr: value.get_field("ptr")?,
            id: value.get_field("id")?,
            loop_depth: value.get_field_renamed_or_default(&["loopDepth", "loop_depth"])?,
            attributes: value.get_field_or_default("attributes")?,
            predecessors: value.get_field_or_default("predecessors")?,
            successors: value.get_field_or_default("successors")?,
            instructions: value.get_field("instructions")?,
            resume_point: value.get_field_renamed_opt(&["resumePoint", "resume_point"])?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for LIRBlock {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(LIRBlock {
            ptr: value.get_field("ptr")?,
            id: value.get_field("id")?,
            loop_depth: value.get_field_renamed_or_default(&["loopDepth", "loop_depth"])?,
            attributes: value.get_field_or_default("attributes")?,
            predecessors: value.get_field_or_default("predecessors")?,
            successors: value.get_field_or_default("successors")?,
            instructions: value.get_field("instructions")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for MIRInstruction {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(MIRInstruction {
            ptr: value.get_field("ptr")?,
            id: value.get_field("id")?,
            opcode: value.get_field("opcode")?,
            type_: value.get_field_opt("type")?,
            attributes: value.get_field_opt("attributes")?,
            inputs: value.get_field_or_default("inputs")?,
            uses: value.get_field_or_default("uses")?,
            mem_inputs: value.get_field_or_default("memInputs")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for LIRInstruction {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(LIRInstruction {
            ptr: value.get_field("ptr")?,
            id: value.get_field("id")?,
            mir_ptr: value.get_field_opt("mir_ptr")?,
            opcode: value.get_field("opcode")?,
            type_: value.get_field_opt("type_")?,
            attributes: value.get_field_opt("attributes")?,
            defs: value.get_field_or_default("defs")?,
        })
    }
}

// ============================================================================
// ToJson implementations for non-serde builds
// ============================================================================

#[cfg(not(feature = "serde"))]
impl ToJson for BlockPtr {
    fn to_json(&self) -> Value {
        self.0.to_json()
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for BlockID {
    fn to_json(&self) -> Value {
        self.0.to_json()
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for InsPtr {
    fn to_json(&self) -> Value {
        self.0.to_json()
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for IonJSON {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("version".to_string(), self.version.to_json());
        if let Some(ref func) = self.function {
            map.insert("function".to_string(), func.to_json());
        }
        map.insert("functions".to_string(), self.functions.to_json());
        map.insert("passes".to_string(), self.passes.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for Function {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("name".to_string(), self.name.to_json());
        map.insert("passes".to_string(), self.passes.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for Pass {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("name".to_string(), self.name.to_json());
        if let Some(ref mir) = self.mir {
            map.insert("mir".to_string(), mir.to_json());
        }
        if let Some(ref lir) = self.lir {
            map.insert("lir".to_string(), lir.to_json());
        }
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for MIRPass {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("blocks".to_string(), self.blocks.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for LIRPass {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("blocks".to_string(), self.blocks.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for MIRBlock {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("ptr".to_string(), self.ptr.to_json());
        map.insert("id".to_string(), self.id.to_json());
        map.insert("loopDepth".to_string(), self.loop_depth.to_json());
        map.insert("attributes".to_string(), self.attributes.to_json());
        map.insert("predecessors".to_string(), self.predecessors.to_json());
        map.insert("successors".to_string(), self.successors.to_json());
        map.insert("instructions".to_string(), self.instructions.to_json());
        if let Some(ref rp) = self.resume_point {
            map.insert("resumePoint".to_string(), rp.clone());
        }
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for LIRBlock {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("ptr".to_string(), self.ptr.to_json());
        map.insert("id".to_string(), self.id.to_json());
        map.insert("loopDepth".to_string(), self.loop_depth.to_json());
        map.insert("attributes".to_string(), self.attributes.to_json());
        map.insert("predecessors".to_string(), self.predecessors.to_json());
        map.insert("successors".to_string(), self.successors.to_json());
        map.insert("instructions".to_string(), self.instructions.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for MIRInstruction {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("ptr".to_string(), self.ptr.to_json());
        map.insert("id".to_string(), self.id.to_json());
        map.insert("opcode".to_string(), self.opcode.to_json());
        if let Some(ref t) = self.type_ {
            map.insert("type".to_string(), t.to_json());
        }
        if let Some(ref attrs) = self.attributes {
            map.insert("attributes".to_string(), attrs.to_json());
        }
        map.insert("inputs".to_string(), self.inputs.to_json());
        map.insert("uses".to_string(), self.uses.to_json());
        map.insert("memInputs".to_string(), self.mem_inputs.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for LIRInstruction {
    fn to_json(&self) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("ptr".to_string(), self.ptr.to_json());
        map.insert("id".to_string(), self.id.to_json());
        if let Some(ref ptr) = self.mir_ptr {
            map.insert("mir_ptr".to_string(), ptr.to_json());
        }
        map.insert("opcode".to_string(), self.opcode.to_json());
        if let Some(ref t) = self.type_ {
            map.insert("type_".to_string(), t.to_json());
        }
        if let Some(ref attrs) = self.attributes {
            map.insert("attributes".to_string(), attrs.to_json());
        }
        map.insert("defs".to_string(), self.defs.to_json());
        Value::Object(map)
    }
}
