use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockID(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockNumber(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonJSON {
    pub functions: Vec<Func>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Func {
    pub name: String,
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pass {
    pub name: String,
    pub mir: MIRData,
    pub lir: LIRData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIRData {
    pub blocks: Vec<MIRBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIRData {
    pub blocks: Vec<LIRBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIRBlock {
    pub id: BlockID,
    pub number: BlockNumber,
    #[serde(rename = "loopDepth")]
    pub loop_depth: u32,
    pub attributes: Vec<String>,
    pub predecessors: Vec<BlockNumber>,
    pub successors: Vec<BlockNumber>,
    pub instructions: Vec<MIRInstruction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIRBlock {
    pub id: BlockID,
    pub number: BlockNumber,
    pub instructions: Vec<LIRInstruction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MIRInstruction {
    pub id: u32,
    pub opcode: String,
    pub attributes: Vec<String>,
    pub inputs: Vec<u32>,
    pub uses: Vec<u32>,
    #[serde(rename = "memInputs")]
    pub mem_inputs: Vec<serde_json::Value>, // Unknown type, using generic JSON
    #[serde(rename = "type")]
    pub instruction_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIRInstruction {
    pub id: u32,
    pub opcode: String,
    pub defs: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

// Layout structures from the algorithm
pub type LayoutNodeID = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutNode {
    pub id: LayoutNodeID,
    pub pos: Vec2,
    pub size: Vec2,
    pub block: Option<Block>,
    pub src_nodes: Vec<LayoutNodeID>, // References to source nodes by ID
    pub dst_nodes: Vec<LayoutNodeID>, // References to destination nodes by ID
    pub joint_offsets: Vec<f64>,
    pub flags: u32, // NodeFlags
    pub dst_block: Option<Block>, // For dummy nodes
    pub layer: usize, // For layer assignment
}

pub type NodeFlags = u32;
pub const LEFTMOST_DUMMY: NodeFlags = 1 << 0;
pub const RIGHTMOST_DUMMY: NodeFlags = 1 << 1;
pub const IMMINENT_BACKEDGE_DUMMY: NodeFlags = 1 << 2;

impl LayoutNode {
    pub fn new_block_node(id: LayoutNodeID, block: Block, layer: usize) -> Self {
        Self {
            id,
            pos: Vec2::new(CONTENT_PADDING, CONTENT_PADDING),
            size: Vec2::new(100.0, 50.0), // Default size for now
            block: Some(block),
            src_nodes: vec![],
            dst_nodes: vec![],
            joint_offsets: vec![],
            flags: 0,
            dst_block: None,
            layer,
        }
    }

    pub fn new_dummy_node(id: LayoutNodeID, dst_block: Block, layer: usize) -> Self {
        Self {
            id,
            pos: Vec2::new(CONTENT_PADDING, CONTENT_PADDING),
            size: Vec2::new(0.0, 0.0), // Dummy nodes have no size
            block: None,
            src_nodes: vec![],
            dst_nodes: vec![],
            joint_offsets: vec![],
            flags: 0,
            dst_block: Some(dst_block),
            layer,
        }
    }

    pub fn is_dummy(&self) -> bool {
        self.block.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockID,
    pub number: BlockNumber,
    pub mir_block: Option<MIRBlock>,
    pub lir_block: Option<LIRBlock>,
    pub predecessors: Vec<BlockNumber>,
    pub successors: Vec<BlockNumber>,
    #[serde(rename = "loopDepth")]
    pub loop_depth: u32,
    #[serde(rename = "loopNum")]
    pub loop_num: u32,
    pub attributes: Vec<String>,
    pub layer: usize, // Layer assignment for layout
    pub size: Vec2, // Size of the block for rendering
    
    // Additional fields to match TypeScript Block IR
    #[serde(rename = "hasLayoutNode")]
    pub has_layout_node: bool,
    #[serde(rename = "instructionCount")]
    pub instruction_count: u32,
    #[serde(rename = "lirInstructionCount")]
    pub lir_instruction_count: u32,
    #[serde(rename = "isBranch")]
    pub is_branch: bool,
    #[serde(rename = "isEntry")]
    pub is_entry: bool,
    #[serde(rename = "isExit")]
    pub is_exit: bool,
    #[serde(rename = "isMerge")]
    pub is_merge: bool,
}

impl Block {
    // Computed properties to match TypeScript Block IR snapshots
    pub fn predecessor_count(&self) -> usize {
        self.predecessors.len()
    }
    
    pub fn predecessor_numbers(&self) -> &Vec<BlockNumber> {
        &self.predecessors
    }
    
    pub fn successor_count(&self) -> usize {
        self.successors.len()
    }
    
    pub fn successor_numbers(&self) -> &Vec<BlockNumber> {
        &self.successors
    }
    
    // Convert to the exact TypeScript Block IR format for snapshots
    pub fn to_block_ir(&self) -> BlockIR {
        BlockIR {
            attributes: self.attributes.clone(),
            has_layout_node: self.has_layout_node,
            id: self.id.0,
            instruction_count: self.instruction_count,
            is_branch: Some(self.is_branch),
            is_entry: Some(self.is_entry),
            is_exit: Some(self.is_exit),
            is_merge: Some(self.is_merge),
            is_backedge: None,
            is_loop_header: None,
            layer: self.layer,
            lir_instruction_count: self.lir_instruction_count,
            loop_depth: self.loop_depth,
            loop_height: None,
            loop_num: self.loop_num,
            number: self.number.0,
            outgoing_edge_count: None,
            parent_loop_number: None,
            predecessor_count: self.predecessor_count(),
            predecessor_numbers: self.predecessors.iter().map(|p| p.0).collect(),
            size: BlockIRSize {
                x: self.size.x,
                y: self.size.y,
            },
            successor_count: self.successor_count(),
            successor_numbers: self.successors.iter().map(|s| s.0).collect(),
        }
    }
}

// Exact TypeScript Block IR format for snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockIR {
    pub attributes: Vec<String>,
    #[serde(rename = "hasLayoutNode")]
    pub has_layout_node: bool,
    pub id: u32,
    #[serde(rename = "instructionCount")]
    pub instruction_count: u32,
    #[serde(rename = "isBranch", skip_serializing_if = "Option::is_none")]
    pub is_branch: Option<bool>,
    #[serde(rename = "isEntry", skip_serializing_if = "Option::is_none")]
    pub is_entry: Option<bool>,
    #[serde(rename = "isExit", skip_serializing_if = "Option::is_none")]
    pub is_exit: Option<bool>,
    #[serde(rename = "isMerge", skip_serializing_if = "Option::is_none")]
    pub is_merge: Option<bool>,
    #[serde(rename = "isBackedge", skip_serializing_if = "Option::is_none")]
    pub is_backedge: Option<bool>,
    #[serde(rename = "isLoopHeader", skip_serializing_if = "Option::is_none")]
    pub is_loop_header: Option<bool>,
    pub layer: usize,
    #[serde(rename = "lirInstructionCount")]
    pub lir_instruction_count: u32,
    #[serde(rename = "loopDepth")]
    pub loop_depth: u32,
    #[serde(rename = "loopHeight", skip_serializing_if = "Option::is_none")]
    pub loop_height: Option<u32>,
    #[serde(rename = "loopNum")]
    pub loop_num: u32,
    pub number: u32,
    #[serde(rename = "outgoingEdgeCount", skip_serializing_if = "Option::is_none")]
    pub outgoing_edge_count: Option<u32>,
    #[serde(rename = "parentLoopNumber", skip_serializing_if = "Option::is_none")]
    pub parent_loop_number: Option<u32>,
    #[serde(rename = "predecessorCount")]
    pub predecessor_count: usize,
    #[serde(rename = "predecessorNumbers")]
    pub predecessor_numbers: Vec<u32>,
    pub size: BlockIRSize,
    #[serde(rename = "successorCount")]
    pub successor_count: usize,
    #[serde(rename = "successorNumbers")]
    pub successor_numbers: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockIRSize {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Loop {
    pub header: BlockNumber,
    pub backedges: Vec<BlockNumber>,
    pub depth: u32,
}

// Constants from the original implementation
pub const ARROW_RADIUS: f64 = 12.0;
pub const BLOCK_PADDING: f64 = 16.0;
pub const PORT_START: f64 = 16.0;
pub const PORT_SPACING: f64 = 60.0;
pub const LAYER_SPACING: f64 = 120.0;
pub const TRANSLATION_CLAMP_AMOUNT: f64 = 40.0;
pub const THRESHOLD_T: f64 = 0.5;
pub const THRESHOLD_ZOOM: f64 = 0.01;

// Additional constants for straightenEdges
pub const BLOCK_GAP: f64 = 44.0;
pub const LAYOUT_ITERATIONS: usize = 2;
pub const NEARLY_STRAIGHT: f64 = 30.0;
pub const NEARLY_STRAIGHT_ITERATIONS: usize = 8;

// Additional constants for finagleJoints and verticalize
pub const JOINT_SPACING: f64 = 16.0;
pub const TRACK_PADDING: f64 = 36.0;
pub const CONTENT_PADDING: f64 = 20.0;