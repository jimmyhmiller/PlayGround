// Universal CodeGraph JSON format - compiler-agnostic

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::json_compat::Value;
use std::collections::HashMap;

#[cfg(not(feature = "serde"))]
use crate::json::{FromJson, JsonObjectExt, ParseError, ToJson};

/// Universal CodeGraph format version
pub const UNIVERSAL_VERSION: &str = "codegraph-v1";

/// Top-level universal IR container
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UniversalIR {
    /// Format identifier - must be "codegraph-v1"
    pub format: String,

    /// Compiler identifier (e.g., "ion", "llvm-mir", "gcc-rtl")
    pub compiler: String,

    /// Optional metadata about the function/compilation unit
    #[cfg_attr(feature = "serde", serde(default))]
    pub metadata: HashMap<String, Value>,

    /// Flat list of basic blocks
    pub blocks: Vec<UniversalBlock>,
}

/// Universal basic block representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UniversalBlock {
    /// Unique block identifier (can be string or number)
    pub id: String,

    /// Block attributes (e.g., ["loopheader"], ["backedge"], etc.)
    #[cfg_attr(feature = "serde", serde(default))]
    pub attributes: Vec<String>,

    /// Loop nesting depth (0 = not in loop)
    #[cfg_attr(feature = "serde", serde(rename = "loopDepth", default))]
    pub loop_depth: u32,

    /// Predecessor block IDs
    #[cfg_attr(feature = "serde", serde(default))]
    pub predecessors: Vec<String>,

    /// Successor block IDs (forward edges only, used for layout)
    #[cfg_attr(feature = "serde", serde(default))]
    pub successors: Vec<String>,

    /// Back edge target IDs (for rendering only, not used in layout)
    #[cfg_attr(feature = "serde", serde(default))]
    pub back_edges: Vec<String>,

    /// Whether this block has a self-loop edge
    #[cfg_attr(feature = "serde", serde(default))]
    pub has_self_loop: bool,

    /// Instructions in this block
    pub instructions: Vec<UniversalInstruction>,

    /// Optional block-level metadata
    #[cfg_attr(feature = "serde", serde(default))]
    pub metadata: HashMap<String, Value>,
}

/// Universal instruction representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UniversalInstruction {
    /// Instruction opcode/name (required)
    pub opcode: String,

    /// Optional instruction attributes
    #[cfg_attr(feature = "serde", serde(default))]
    pub attributes: Vec<String>,

    /// Optional type annotation (e.g., "int32", "i64*", etc.)
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub type_: Option<String>,

    /// Optional profiling data
    #[cfg_attr(feature = "serde", serde(default))]
    pub profiling: Option<ProfilingData>,

    /// Optional instruction-level metadata (compiler-specific fields go here)
    #[cfg_attr(feature = "serde", serde(default))]
    pub metadata: HashMap<String, Value>,
}

/// Profiling/sample count data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProfilingData {
    /// Total sample count
    #[cfg_attr(feature = "serde", serde(default))]
    pub sample_count: u64,

    /// Hotness score (0.0 = cold, 1.0 = hot)
    #[cfg_attr(feature = "serde", serde(default))]
    pub hotness: f64,

    /// Additional profiling metrics
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub extra: HashMap<String, Value>,
}

impl UniversalIR {
    /// Validate that this is a universal format IR
    pub fn validate(&self) -> Result<(), String> {
        if self.format != UNIVERSAL_VERSION {
            return Err(format!(
                "Invalid format: expected '{}', got '{}'",
                UNIVERSAL_VERSION, self.format
            ));
        }
        Ok(())
    }

    /// Check if this JSON looks like universal format (without full deserialization)
    pub fn is_universal_format(json: &Value) -> bool {
        json.get("format")
            .and_then(|v| v.as_str())
            .map(|s| s == UNIVERSAL_VERSION)
            .unwrap_or(false)
    }
}

// ============================================================================
// FromJson implementations for non-serde builds
// ============================================================================

#[cfg(not(feature = "serde"))]
impl FromJson for UniversalIR {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(UniversalIR {
            format: value.get_field("format")?,
            compiler: value.get_field("compiler")?,
            metadata: value.get_field_or_default("metadata")?,
            blocks: value.get_field("blocks")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for UniversalBlock {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(UniversalBlock {
            id: value.get_field("id")?,
            attributes: value.get_field_or_default("attributes")?,
            loop_depth: value.get_field_renamed_or_default(&["loopDepth", "loop_depth"])?,
            predecessors: value.get_field_or_default("predecessors")?,
            successors: value.get_field_or_default("successors")?,
            back_edges: value.get_field_or_default("back_edges")?,
            has_self_loop: value.get_field_or_default("has_self_loop")?,
            instructions: value.get_field("instructions")?,
            metadata: value.get_field_or_default("metadata")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for UniversalInstruction {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        Ok(UniversalInstruction {
            opcode: value.get_field("opcode")?,
            attributes: value.get_field_or_default("attributes")?,
            type_: value.get_field_opt("type")?,
            profiling: value.get_field_opt("profiling")?,
            metadata: value.get_field_or_default("metadata")?,
        })
    }
}

#[cfg(not(feature = "serde"))]
impl FromJson for ProfilingData {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        // Handle flatten by collecting all other keys into extra
        let sample_count = value.get_field_or_default("sample_count")?;
        let hotness = value.get_field_or_default("hotness")?;

        let mut extra = HashMap::new();
        if let Some(obj) = value.as_object() {
            for (k, v) in obj {
                if k != "sample_count" && k != "hotness" {
                    extra.insert(k.clone(), v.clone());
                }
            }
        }

        Ok(ProfilingData {
            sample_count,
            hotness,
            extra,
        })
    }
}

// ============================================================================
// ToJson implementations for non-serde builds
// ============================================================================

#[cfg(not(feature = "serde"))]
impl ToJson for UniversalIR {
    fn to_json(&self) -> Value {
        let mut map = HashMap::new();
        map.insert("format".to_string(), self.format.to_json());
        map.insert("compiler".to_string(), self.compiler.to_json());
        map.insert("metadata".to_string(), self.metadata.to_json());
        map.insert("blocks".to_string(), self.blocks.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for UniversalBlock {
    fn to_json(&self) -> Value {
        let mut map = HashMap::new();
        map.insert("id".to_string(), self.id.to_json());
        map.insert("attributes".to_string(), self.attributes.to_json());
        map.insert("loopDepth".to_string(), self.loop_depth.to_json());
        map.insert("predecessors".to_string(), self.predecessors.to_json());
        map.insert("successors".to_string(), self.successors.to_json());
        map.insert("back_edges".to_string(), self.back_edges.to_json());
        map.insert("has_self_loop".to_string(), self.has_self_loop.to_json());
        map.insert("instructions".to_string(), self.instructions.to_json());
        map.insert("metadata".to_string(), self.metadata.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for UniversalInstruction {
    fn to_json(&self) -> Value {
        let mut map = HashMap::new();
        map.insert("opcode".to_string(), self.opcode.to_json());
        map.insert("attributes".to_string(), self.attributes.to_json());
        if let Some(ref t) = self.type_ {
            map.insert("type".to_string(), t.to_json());
        }
        if let Some(ref p) = self.profiling {
            map.insert("profiling".to_string(), p.to_json());
        }
        map.insert("metadata".to_string(), self.metadata.to_json());
        Value::Object(map)
    }
}

#[cfg(not(feature = "serde"))]
impl ToJson for ProfilingData {
    fn to_json(&self) -> Value {
        let mut map = HashMap::new();
        map.insert("sample_count".to_string(), self.sample_count.to_json());
        map.insert("hotness".to_string(), self.hotness.to_json());
        // Flatten extra fields
        for (k, v) in &self.extra {
            map.insert(k.clone(), v.clone());
        }
        Value::Object(map)
    }
}
