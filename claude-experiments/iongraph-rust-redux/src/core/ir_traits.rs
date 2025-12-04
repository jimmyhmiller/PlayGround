use crate::layout_provider::LayoutProvider;
use serde::de::DeserializeOwned;
use std::collections::HashMap;

/// Block pointer type (opaque identifier)
pub type BlockPtr = String;

/// Trait representing a compiler's intermediate representation
///
/// This trait abstracts over different compiler IRs (Ion, LLVM, GCC, etc.)
/// allowing the same visualization engine to work with any compiler.
///
/// # Example: Ion Implementation
///
/// ```ignore
/// pub struct IonIR;
///
/// impl CompilerIR for IonIR {
///     fn format_id() -> &'static str { "ion" }
///     fn version() -> u32 { 1 }
///     type Instruction = MIRInstruction;
///     type Block = MIRBlock;
///     type Container = IonJSON;
///     // ... implement other methods
/// }
/// ```
pub trait CompilerIR: Sized {
    /// Unique identifier for this IR format (e.g., "ion", "llvm-mir", "gcc-rtl")
    fn format_id() -> &'static str;

    /// Version of the IR format (for migration support)
    fn version() -> u32;

    /// Instruction type for this IR
    type Instruction: IRInstruction;

    /// Block type for this IR
    type Block: IRBlock<Instruction = Self::Instruction>;

    /// Top-level container type (e.g., Ion's IonJSON with functions → passes → blocks)
    type Container: DeserializeOwned;

    /// Extract all blocks from the container
    ///
    /// This flattens the compiler-specific hierarchy into a simple block list.
    /// For example, Ion has functions → passes → blocks, which gets flattened
    /// to just a list of blocks.
    fn extract_blocks(container: &Self::Container) -> Vec<Self::Block>;

    /// Optional: Migrate from older format versions
    ///
    /// Default implementation does nothing (no migration needed).
    fn migrate(container: Self::Container) -> Result<Self::Container, String> {
        Ok(container)
    }

    /// Optional: Color scheme for instruction/block attributes
    ///
    /// Returns a map from attribute name to CSS color string.
    /// Default implementation returns empty map (no custom colors).
    fn attribute_colors() -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for individual instructions in an IR
///
/// Each compiler has different instruction formats:
/// - Ion: MIR/LIR with opcode, type, attributes
/// - LLVM: SSA form with %registers
/// - GCC: RTL instructions
///
/// This trait provides a common interface for all of them.
pub trait IRInstruction: Clone {
    /// Instruction opcode/name (e.g., "MLoadElement", "add i32", "mov rax, rbx")
    fn opcode(&self) -> &str;

    /// Instruction attributes (e.g., ["Movable", "Guard"] for Ion)
    fn attributes(&self) -> &[String];

    /// Optional type annotation (e.g., "int32" for Ion, "i64*" for LLVM)
    fn type_annotation(&self) -> Option<&str> {
        None
    }

    /// Optional profiling data (e.g., sample counts for Ion)
    ///
    /// Returns None if no profiling data available.
    fn profiling_data(&self) -> Option<Vec<u64>> {
        None
    }

    /// Render instruction as HTML table row
    ///
    /// This is where compiler-specific rendering happens. Each compiler
    /// can define its own table structure (columns, formatting, etc.).
    ///
    /// # Arguments
    /// * `provider` - The layout provider (mutable access for creating elements)
    /// * `id` - Sequential instruction ID within the block
    ///
    /// # Returns
    /// A `<tr>` element containing the rendered instruction
    fn render_row<P: LayoutProvider>(&self, provider: &mut P, id: usize) -> Box<P::Element>;
}

/// Trait for basic blocks in an IR
///
/// All compilers use some notion of basic blocks (sequential instructions
/// with control flow only at the end). This trait provides a common interface.
pub trait IRBlock: Clone {
    /// The instruction type for this block
    type Instruction: IRInstruction;

    /// Opaque block pointer/identifier
    ///
    /// This is typically a compiler-internal pointer or unique name.
    fn ptr(&self) -> BlockPtr;

    /// Block attributes (e.g., ["loopheader", "backedge", "splitedge"] for Ion)
    fn attributes(&self) -> &[String];

    /// Loop nesting depth (0 = not in loop, 1 = one level deep, etc.)
    ///
    /// Default implementation returns 0 (no loop information).
    fn loop_depth(&self) -> u32 {
        0
    }

    /// Predecessor block indices
    ///
    /// Returns indices into the flattened block array (not block IDs).
    fn predecessors(&self) -> &[usize];

    /// Successor block indices
    ///
    /// Returns indices into the flattened block array (not block IDs).
    fn successors(&self) -> &[usize];

    /// Instructions in this block
    fn instructions(&self) -> &[Self::Instruction];
}
