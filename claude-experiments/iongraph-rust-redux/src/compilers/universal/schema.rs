// Universal CodeGraph JSON format - compiler-agnostic
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Universal CodeGraph format version
pub const UNIVERSAL_VERSION: &str = "codegraph-v1";

/// Top-level universal IR container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIR {
    /// Format identifier - must be "codegraph-v1"
    pub format: String,

    /// Compiler identifier (e.g., "ion", "llvm-mir", "gcc-rtl")
    pub compiler: String,

    /// Optional metadata about the function/compilation unit
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Flat list of basic blocks
    pub blocks: Vec<UniversalBlock>,
}

/// Universal basic block representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalBlock {
    /// Unique block identifier (can be string or number)
    pub id: String,

    /// Block attributes (e.g., ["loopheader"], ["backedge"], etc.)
    #[serde(default)]
    pub attributes: Vec<String>,

    /// Loop nesting depth (0 = not in loop)
    #[serde(rename = "loopDepth", default)]
    pub loop_depth: u32,

    /// Predecessor block IDs
    #[serde(default)]
    pub predecessors: Vec<String>,

    /// Successor block IDs
    #[serde(default)]
    pub successors: Vec<String>,

    /// Instructions in this block
    pub instructions: Vec<UniversalInstruction>,

    /// Optional block-level metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Universal instruction representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalInstruction {
    /// Instruction opcode/name (required)
    pub opcode: String,

    /// Optional instruction attributes
    #[serde(default)]
    pub attributes: Vec<String>,

    /// Optional type annotation (e.g., "int32", "i64*", etc.)
    #[serde(rename = "type")]
    pub type_: Option<String>,

    /// Optional profiling data
    #[serde(default)]
    pub profiling: Option<ProfilingData>,

    /// Optional instruction-level metadata (compiler-specific fields go here)
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Profiling/sample count data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingData {
    /// Total sample count
    #[serde(default)]
    pub sample_count: u64,

    /// Hotness score (0.0 = cold, 1.0 = hot)
    #[serde(default)]
    pub hotness: f64,

    /// Additional profiling metrics
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
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
    pub fn is_universal_format(json: &serde_json::Value) -> bool {
        json.get("format")
            .and_then(|v| v.as_str())
            .map(|s| s == UNIVERSAL_VERSION)
            .unwrap_or(false)
    }
}
