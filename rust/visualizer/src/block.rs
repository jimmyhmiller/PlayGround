use serde::{Serialize, Deserialize};

// Maximum number of temp value types we keep track of
pub const MAX_TEMP_TYPES: usize = 8;

// Maximum number of local variable types we keep track of
const MAX_LOCAL_TYPES: usize = 8;

// Represent the type of a value (local/stack/self) in YJIT
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Type {
    Unknown,
    UnknownImm,
    UnknownHeap,
    Nil,
    True,
    False,
    Fixnum,
    Flonum,
    Array,
    Hash,
    ImmSymbol,

    #[allow(unused)]
    HeapSymbol,

    TString, // An object with the T_STRING flag set, possibly an rb_cString
    CString, // An un-subclassed string of type rb_cString (can have instance vars in some cases)

    BlockParamProxy, // A special sentinel value indicating the block parameter should be read from
                     // the current surrounding cfp
}
// Default initialization
impl Default for Type {
    fn default() -> Self {
        Type::Unknown
    }
}


// Potential mapping of a value on the temporary stack to
// self, a local variable or constant so that we can track its type
#[derive(Copy, Clone, Eq, Debug, Serialize, Deserialize, Hash, PartialEq)]
pub enum TempMapping {
    MapToStack, // Normal stack value
    MapToSelf,  // Temp maps to the self operand
    MapToLocal(u8), // Temp maps to a local variable with index
                //ConstMapping,         // Small constant (0, 1, 2, Qnil, Qfalse, Qtrue)
}


impl Default for TempMapping {
    fn default() -> Self {
        TempMapping::MapToStack
    }
}


#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Context {
    // Number of values currently on the temporary stack
    stack_size: u16,

    // Offset of the JIT SP relative to the interpreter SP
    // This represents how far the JIT's SP is from the "real" SP
    sp_offset: i16,

    // Depth of this block in the sidechain (eg: inline-cache chain)
    chain_depth: u8,

    // Local variable types we keep track of
    local_types: [Type; MAX_LOCAL_TYPES],

    // Temporary variable types we keep track of
    temp_types: [Type; MAX_TEMP_TYPES],

    // Type we track for self
    self_type: Type,

    // Mapping of temp stack entries to types we track
    temp_mapping: [TempMapping; MAX_TEMP_TYPES],
}


#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct VALUE(pub usize);


#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CmeDependency {
    receiver_klass: VALUE,
    callee_cme: u32,
}


#[derive(Debug, Copy, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct BlockId {
    pub iseq: usize,
    pub idx: u32,
}

impl BlockId {
    fn _name(&self) -> String {
        format!("block_{}_{}", self.iseq, self.idx)
    }
}


#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Hash)]
pub enum BranchShape {
    Next0,   // Target 0 is next
    Next1,   // Target 1 is next
    Default, // Neither target is next
}




/// A place that a branch could jump to
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct BranchTarget {
    pub address: Option<usize>,
    pub id: BlockId,
    pub ctx: Context,
    pub block: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Branch {
    pub id: usize,
    pub block_id: usize,
    pub start_addr: Option<usize>,
    pub end_addr: Option<usize>,

    pub shape: BranchShape,
    #[serde(skip)]
    pub writable_areas: Vec<(usize, usize)>,
    pub disasm: String,

    // Branch target blocks and their contexts
    pub targets: [Option<BranchTarget>; 2],

    #[serde(skip)]
    pub bytes: Vec<u8>,
}


#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct CodeLocation {
    pub file: Option<String>,
    pub method_name: Option<String>,
    pub line_start: (i32, i32),
    pub line_end: (i32, i32),
}


#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Block {
    pub block_id: BlockId,

    pub id: usize,

    pub end_idx: u32,

    pub ctx: Context,

    pub start_addr: Option<usize>,
    pub end_addr: Option<usize>,

    pub incoming: Vec<Branch>,
    pub outgoing: Vec<Branch>,

    pub gc_object_offsets: Vec<u32>,

    pub entry_exit: Option<usize>,

    pub location: CodeLocation,

    pub disasm: String,

    pub epoch: usize,

    #[serde(default)]
    pub is_exit: bool,

    pub created_at: usize,

}

