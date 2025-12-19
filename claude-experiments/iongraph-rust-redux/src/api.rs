//! High-level API for building and rendering graphs.
//!
//! # Examples
//!
//! ## Render from UniversalIR
//! ```
//! use iongraph_rust_redux::api::render_svg;
//! use iongraph_rust_redux::compilers::universal::{UniversalIR, UniversalBlock, UniversalInstruction};
//!
//! let ir = UniversalIR {
//!     format: "codegraph-v1".to_string(),
//!     compiler: "example".to_string(),
//!     metadata: Default::default(),
//!     blocks: vec![
//!         UniversalBlock {
//!             id: "0".to_string(),
//!             attributes: vec![],
//!             loop_depth: 0,
//!             predecessors: vec![],
//!             successors: vec![],
//!             back_edges: vec![],
//!             has_self_loop: false,
//!             instructions: vec![
//!                 UniversalInstruction {
//!                     opcode: "return".to_string(),
//!                     attributes: vec![],
//!                     type_: None,
//!                     profiling: None,
//!                     metadata: Default::default(),
//!                 },
//!             ],
//!             metadata: Default::default(),
//!         },
//!     ],
//! };
//!
//! let svg = render_svg(&ir);
//! assert!(svg.starts_with("<svg"));
//! ```
//!
//! ## Build graphs with GraphBuilder
//! ```
//! use iongraph_rust_redux::api::GraphBuilder;
//!
//! let svg = GraphBuilder::new("my-compiler")
//!     .block("entry")
//!         .instruction("param", Some("i32"))
//!         .instruction("const", Some("i32"))
//!     .block("body")
//!         .instruction("add", Some("i32"))
//!         .instruction("return", None)
//!     .edge("entry", "body")
//!     .render_svg();
//!
//! assert!(svg.starts_with("<svg"));
//! ```
//!
//! ## Implement traits for your own types
//! ```
//! use iongraph_rust_redux::api::{Render, ToGraph, GraphBlock, GraphInstruction};
//! use iongraph_rust_redux::compilers::universal::UniversalIR;
//!
//! // Your custom instruction type
//! struct MyInstr {
//!     name: String,
//!     ty: Option<String>,
//! }
//!
//! impl GraphInstruction for MyInstr {
//!     fn opcode(&self) -> &str { &self.name }
//!     fn type_annotation(&self) -> Option<&str> { self.ty.as_deref() }
//! }
//!
//! // Your custom block type
//! struct MyBlock {
//!     name: String,
//!     instrs: Vec<MyInstr>,
//!     succs: Vec<String>,
//! }
//!
//! impl GraphBlock for MyBlock {
//!     type Instruction = MyInstr;
//!     fn id(&self) -> &str { &self.name }
//!     fn successors(&self) -> Vec<String> { self.succs.clone() }
//!     fn instructions(&self) -> Vec<&Self::Instruction> { self.instrs.iter().collect() }
//! }
//!
//! // Your custom graph type
//! struct MyGraph {
//!     blocks: Vec<MyBlock>,
//! }
//!
//! impl ToGraph for MyGraph {
//!     type Block = MyBlock;
//!     fn compiler(&self) -> &str { "my-compiler" }
//!     fn blocks(&self) -> Vec<&Self::Block> { self.blocks.iter().collect() }
//! }
//!
//! // Now you can render it!
//! let graph = MyGraph {
//!     blocks: vec![
//!         MyBlock {
//!             name: "entry".to_string(),
//!             instrs: vec![MyInstr { name: "ret".to_string(), ty: None }],
//!             succs: vec![],
//!         },
//!     ],
//! };
//!
//! let svg = graph.render_svg();
//! assert!(svg.starts_with("<svg"));
//! ```

use crate::compilers::universal::{UniversalIR, UniversalBlock, UniversalInstruction, UNIVERSAL_VERSION};
use crate::graph::{Graph, GraphOptions};
use crate::layout_provider::LayoutProvider;
use crate::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use std::collections::HashMap;

// =============================================================================
// Trait-based API
// =============================================================================

/// Trait for types that can be rendered as SVG graphs.
///
/// This trait is automatically implemented for any type that implements `ToGraph`.
pub trait Render {
    /// Convert this to a UniversalIR representation.
    fn to_universal_ir(&self) -> UniversalIR;

    /// Render this graph to an SVG string.
    fn render_svg(&self) -> String {
        render_svg(&self.to_universal_ir())
    }
}

/// Trait for instruction-like types.
///
/// Implement this trait for your instruction type to enable graph rendering.
pub trait GraphInstruction {
    /// The opcode/name of this instruction (required).
    fn opcode(&self) -> &str;

    /// Optional type annotation (e.g., "i32", "void*").
    fn type_annotation(&self) -> Option<&str> {
        None
    }

    /// Optional attributes for this instruction.
    fn attributes(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Trait for block-like types.
///
/// Implement this trait for your block type to enable graph rendering.
pub trait GraphBlock {
    /// The instruction type used by this block.
    type Instruction: GraphInstruction;

    /// The unique identifier for this block.
    fn id(&self) -> &str;

    /// The successor block IDs (forward edges).
    fn successors(&self) -> Vec<String>;

    /// The back edge target block IDs (for loops).
    fn back_edges(&self) -> Vec<String> {
        Vec::new()
    }

    /// Optional attributes for this block (e.g., "loopheader", "entry").
    fn attributes(&self) -> Vec<String> {
        Vec::new()
    }

    /// The loop nesting depth (0 = not in a loop).
    fn loop_depth(&self) -> u32 {
        0
    }

    /// The instructions in this block.
    fn instructions(&self) -> Vec<&Self::Instruction>;
}

/// Trait for graph-like types.
///
/// Implement this trait to make your data structure renderable as a graph.
/// The `Render` trait is automatically implemented for any type implementing `ToGraph`.
pub trait ToGraph {
    /// The block type used by this graph.
    type Block: GraphBlock;

    /// The compiler/IR name for this graph.
    fn compiler(&self) -> &str;

    /// Get all blocks in this graph.
    fn blocks(&self) -> Vec<&Self::Block>;
}

// Blanket implementation: any ToGraph automatically gets Render
impl<T: ToGraph> Render for T {
    fn to_universal_ir(&self) -> UniversalIR {
        // First pass: collect all blocks and build predecessor map
        let blocks_ref = self.blocks();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();

        for block in &blocks_ref {
            for succ in block.successors() {
                predecessors.entry(succ).or_default().push(block.id().to_string());
            }
            for back_edge in block.back_edges() {
                predecessors.entry(back_edge).or_default().push(block.id().to_string());
            }
        }

        // Second pass: build UniversalBlocks
        let universal_blocks: Vec<UniversalBlock> = blocks_ref
            .iter()
            .map(|block| {
                let id = block.id().to_string();
                let back_edges = block.back_edges();
                let has_self_loop = back_edges.contains(&id);

                let instructions: Vec<UniversalInstruction> = block
                    .instructions()
                    .iter()
                    .map(|instr| UniversalInstruction {
                        opcode: instr.opcode().to_string(),
                        attributes: instr.attributes(),
                        type_: instr.type_annotation().map(|s| s.to_string()),
                        profiling: None,
                        metadata: HashMap::new(),
                    })
                    .collect();

                UniversalBlock {
                    id: id.clone(),
                    attributes: block.attributes(),
                    loop_depth: block.loop_depth(),
                    predecessors: predecessors.get(&id).cloned().unwrap_or_default(),
                    successors: block.successors(),
                    back_edges,
                    has_self_loop,
                    instructions,
                    metadata: HashMap::new(),
                }
            })
            .collect();

        UniversalIR {
            format: UNIVERSAL_VERSION.to_string(),
            compiler: self.compiler().to_string(),
            metadata: HashMap::new(),
            blocks: universal_blocks,
        }
    }
}

// Also implement Render for UniversalIR directly
impl Render for UniversalIR {
    fn to_universal_ir(&self) -> UniversalIR {
        self.clone()
    }
}

/// Render a UniversalIR to an SVG string.
///
/// This is the simplest way to convert a graph to SVG.
pub fn render_svg(ir: &UniversalIR) -> String {
    let mut layout_provider = PureSVGTextLayoutProvider::new();
    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    let mut graph = Graph::new(layout_provider, ir.clone(), options);
    let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
    graph.render(nodes_by_layer, layer_heights, track_heights);

    layout_provider = graph.layout_provider;
    let mut svg_root = layout_provider.create_svg_element("svg");
    layout_provider.set_attribute(&mut svg_root, "xmlns", "http://www.w3.org/2000/svg");

    let width = (graph.size.x + 40.0).ceil() as i32;
    let height = (graph.size.y + 40.0).ceil() as i32;

    layout_provider.set_attribute(&mut svg_root, "width", &width.to_string());
    layout_provider.set_attribute(&mut svg_root, "height", &height.to_string());
    layout_provider.set_attribute(
        &mut svg_root,
        "viewBox",
        &format!("0 0 {} {}", width, height),
    );
    layout_provider.append_child(&mut svg_root, graph.graph_container);

    layout_provider.to_svg_string(&svg_root)
}

/// Parse JSON and render to SVG.
///
/// Supports both IonJSON format and UniversalIR format.
pub fn render_svg_from_json(json_str: &str) -> Result<String, String> {
    // Try to detect format
    let value: crate::json_compat::Value = crate::json_compat::parse(json_str)
        .map_err(|e| format!("JSON parse error: {}", e))?;

    if UniversalIR::is_universal_format(&value) {
        // Parse as UniversalIR
        let ir: UniversalIR = crate::json_compat::parse_as(json_str)
            .map_err(|e| format!("UniversalIR parse error: {}", e))?;
        Ok(render_svg(&ir))
    } else {
        // Try as IonJSON
        use crate::compilers::ion::schema::IonJSON;
        use crate::compilers::universal::pass_to_universal;

        let ion: IonJSON = crate::json_compat::parse_as(json_str)
            .map_err(|e| format!("IonJSON parse error: {}", e))?;

        if ion.functions.is_empty() {
            return Err("No functions in IonJSON".to_string());
        }
        if ion.functions[0].passes.is_empty() {
            return Err("No passes in function".to_string());
        }

        let func = &ion.functions[0];
        let pass = &func.passes[0];
        let ir = pass_to_universal(pass, &func.name);

        Ok(render_svg(&ir))
    }
}

/// Parse IonJSON and render a specific function/pass to SVG.
pub fn render_ion_pass(json_str: &str, func_idx: usize, pass_idx: usize) -> Result<String, String> {
    use crate::compilers::ion::schema::IonJSON;
    use crate::compilers::universal::pass_to_universal;

    let ion: IonJSON = crate::json_compat::parse_as(json_str)
        .map_err(|e| format!("IonJSON parse error: {}", e))?;

    if func_idx >= ion.functions.len() {
        return Err(format!("Function index {} out of range (max: {})",
            func_idx, ion.functions.len().saturating_sub(1)));
    }

    let func = &ion.functions[func_idx];
    if pass_idx >= func.passes.len() {
        return Err(format!("Pass index {} out of range (max: {})",
            pass_idx, func.passes.len().saturating_sub(1)));
    }

    let pass = &func.passes[pass_idx];
    let ir = pass_to_universal(pass, &func.name);

    Ok(render_svg(&ir))
}

/// Builder for constructing graphs programmatically.
///
/// # Example
/// ```
/// use iongraph_rust_redux::api::GraphBuilder;
///
/// let svg = GraphBuilder::new("my-ir")
///     .block("b0")
///         .attr("entry")
///         .instruction("load", Some("i32"))
///         .instruction("add", Some("i32"))
///     .block("b1")
///         .instruction("store", None)
///         .instruction("return", None)
///     .edge("b0", "b1")
///     .render_svg();
/// ```
pub struct GraphBuilder {
    compiler: String,
    blocks: Vec<BlockBuilder>,
    edges: Vec<(String, String)>,
    back_edges: Vec<(String, String)>,
    current_block: Option<usize>,
}

struct BlockBuilder {
    id: String,
    attributes: Vec<String>,
    loop_depth: u32,
    instructions: Vec<UniversalInstruction>,
}

impl GraphBuilder {
    /// Create a new graph builder with the given compiler name.
    pub fn new(compiler: &str) -> Self {
        GraphBuilder {
            compiler: compiler.to_string(),
            blocks: Vec::new(),
            edges: Vec::new(),
            back_edges: Vec::new(),
            current_block: None,
        }
    }

    /// Add a new block and make it the current block.
    pub fn block(mut self, id: &str) -> Self {
        let idx = self.blocks.len();
        self.blocks.push(BlockBuilder {
            id: id.to_string(),
            attributes: Vec::new(),
            loop_depth: 0,
            instructions: Vec::new(),
        });
        self.current_block = Some(idx);
        self
    }

    /// Add an attribute to the current block.
    pub fn attr(mut self, attribute: &str) -> Self {
        if let Some(idx) = self.current_block {
            self.blocks[idx].attributes.push(attribute.to_string());
        }
        self
    }

    /// Set the loop depth of the current block.
    pub fn loop_depth(mut self, depth: u32) -> Self {
        if let Some(idx) = self.current_block {
            self.blocks[idx].loop_depth = depth;
        }
        self
    }

    /// Add an instruction to the current block.
    pub fn instruction(mut self, opcode: &str, type_: Option<&str>) -> Self {
        if let Some(idx) = self.current_block {
            self.blocks[idx].instructions.push(UniversalInstruction {
                opcode: opcode.to_string(),
                attributes: Vec::new(),
                type_: type_.map(|s| s.to_string()),
                profiling: None,
                metadata: HashMap::new(),
            });
        }
        self
    }

    /// Add an instruction with attributes to the current block.
    pub fn instruction_with_attrs(mut self, opcode: &str, type_: Option<&str>, attrs: &[&str]) -> Self {
        if let Some(idx) = self.current_block {
            self.blocks[idx].instructions.push(UniversalInstruction {
                opcode: opcode.to_string(),
                attributes: attrs.iter().map(|s| s.to_string()).collect(),
                type_: type_.map(|s| s.to_string()),
                profiling: None,
                metadata: HashMap::new(),
            });
        }
        self
    }

    /// Add a forward edge between two blocks.
    pub fn edge(mut self, from: &str, to: &str) -> Self {
        self.edges.push((from.to_string(), to.to_string()));
        self
    }

    /// Add a back edge (loop edge) between two blocks.
    pub fn back_edge(mut self, from: &str, to: &str) -> Self {
        self.back_edges.push((from.to_string(), to.to_string()));
        self
    }

    /// Build the UniversalIR from this builder.
    pub fn build(self) -> UniversalIR {
        // Build successor/predecessor maps
        let mut successors: HashMap<String, Vec<String>> = HashMap::new();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
        let mut back_edge_map: HashMap<String, Vec<String>> = HashMap::new();

        for (from, to) in &self.edges {
            successors.entry(from.clone()).or_default().push(to.clone());
            predecessors.entry(to.clone()).or_default().push(from.clone());
        }

        for (from, to) in &self.back_edges {
            back_edge_map.entry(from.clone()).or_default().push(to.clone());
            predecessors.entry(to.clone()).or_default().push(from.clone());
        }

        let blocks: Vec<UniversalBlock> = self.blocks.into_iter().map(|b| {
            let has_self_loop = back_edge_map.get(&b.id)
                .map(|targets| targets.contains(&b.id))
                .unwrap_or(false);

            UniversalBlock {
                id: b.id.clone(),
                attributes: b.attributes,
                loop_depth: b.loop_depth,
                predecessors: predecessors.get(&b.id).cloned().unwrap_or_default(),
                successors: successors.get(&b.id).cloned().unwrap_or_default(),
                back_edges: back_edge_map.get(&b.id).cloned().unwrap_or_default(),
                has_self_loop,
                instructions: b.instructions,
                metadata: HashMap::new(),
            }
        }).collect();

        UniversalIR {
            format: UNIVERSAL_VERSION.to_string(),
            compiler: self.compiler,
            metadata: HashMap::new(),
            blocks,
        }
    }

    /// Build and render to SVG.
    pub fn render_svg(self) -> String {
        let ir = self.build();
        render_svg(&ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_svg_simple() {
        let ir = UniversalIR {
            format: UNIVERSAL_VERSION.to_string(),
            compiler: "test".to_string(),
            metadata: HashMap::new(),
            blocks: vec![
                UniversalBlock {
                    id: "0".to_string(),
                    attributes: vec![],
                    loop_depth: 0,
                    predecessors: vec![],
                    successors: vec![],
                    back_edges: vec![],
                    has_self_loop: false,
                    instructions: vec![
                        UniversalInstruction {
                            opcode: "return".to_string(),
                            attributes: vec![],
                            type_: None,
                            profiling: None,
                            metadata: HashMap::new(),
                        },
                    ],
                    metadata: HashMap::new(),
                },
            ],
        };

        let svg = render_svg(&ir);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("return"));
    }

    #[test]
    fn test_builder_simple() {
        let svg = GraphBuilder::new("test")
            .block("entry")
                .instruction("const", Some("i32"))
                .instruction("return", None)
            .render_svg();

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("const"));
        assert!(svg.contains("return"));
    }

    #[test]
    fn test_builder_with_edges() {
        let svg = GraphBuilder::new("test")
            .block("entry")
                .instruction("branch", None)
            .block("left")
                .instruction("const", Some("i32"))
            .block("right")
                .instruction("const", Some("i32"))
            .block("exit")
                .instruction("phi", Some("i32"))
                .instruction("return", None)
            .edge("entry", "left")
            .edge("entry", "right")
            .edge("left", "exit")
            .edge("right", "exit")
            .render_svg();

        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_builder_with_loop() {
        let svg = GraphBuilder::new("test")
            .block("entry")
                .instruction("const", Some("i32"))
            .block("header")
                .attr("loopheader")
                .loop_depth(1)
                .instruction("phi", Some("i32"))
                .instruction("cmp", Some("bool"))
            .block("body")
                .loop_depth(1)
                .instruction("add", Some("i32"))
            .block("exit")
                .instruction("return", None)
            .edge("entry", "header")
            .edge("header", "body")
            .edge("header", "exit")
            .back_edge("body", "header")
            .render_svg();

        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_builder_build_ir() {
        let ir = GraphBuilder::new("my-compiler")
            .block("b0")
                .attr("entry")
                .instruction("load", Some("i32"))
            .block("b1")
                .instruction("store", None)
            .edge("b0", "b1")
            .build();

        assert_eq!(ir.format, UNIVERSAL_VERSION);
        assert_eq!(ir.compiler, "my-compiler");
        assert_eq!(ir.blocks.len(), 2);
        assert_eq!(ir.blocks[0].id, "b0");
        assert_eq!(ir.blocks[0].successors, vec!["b1"]);
        assert_eq!(ir.blocks[1].predecessors, vec!["b0"]);
    }

    // =========================================================================
    // Trait-based API tests
    // =========================================================================

    // Custom instruction type for testing
    struct TestInstr {
        name: String,
        ty: Option<String>,
    }

    impl GraphInstruction for TestInstr {
        fn opcode(&self) -> &str {
            &self.name
        }
        fn type_annotation(&self) -> Option<&str> {
            self.ty.as_deref()
        }
    }

    // Custom block type for testing
    struct TestBlock {
        id: String,
        instrs: Vec<TestInstr>,
        succs: Vec<String>,
        back: Vec<String>,
        attrs: Vec<String>,
        depth: u32,
    }

    impl GraphBlock for TestBlock {
        type Instruction = TestInstr;

        fn id(&self) -> &str {
            &self.id
        }

        fn successors(&self) -> Vec<String> {
            self.succs.clone()
        }

        fn back_edges(&self) -> Vec<String> {
            self.back.clone()
        }

        fn attributes(&self) -> Vec<String> {
            self.attrs.clone()
        }

        fn loop_depth(&self) -> u32 {
            self.depth
        }

        fn instructions(&self) -> Vec<&Self::Instruction> {
            self.instrs.iter().collect()
        }
    }

    // Custom graph type for testing
    struct TestGraph {
        name: String,
        blocks: Vec<TestBlock>,
    }

    impl ToGraph for TestGraph {
        type Block = TestBlock;

        fn compiler(&self) -> &str {
            &self.name
        }

        fn blocks(&self) -> Vec<&Self::Block> {
            self.blocks.iter().collect()
        }
    }

    #[test]
    fn test_trait_simple_graph() {
        let graph = TestGraph {
            name: "my-custom-ir".to_string(),
            blocks: vec![
                TestBlock {
                    id: "entry".to_string(),
                    instrs: vec![
                        TestInstr { name: "load".to_string(), ty: Some("i32".to_string()) },
                        TestInstr { name: "ret".to_string(), ty: None },
                    ],
                    succs: vec![],
                    back: vec![],
                    attrs: vec![],
                    depth: 0,
                },
            ],
        };

        let svg = graph.render_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("load"));
        assert!(svg.contains("ret"));
    }

    #[test]
    fn test_trait_graph_with_edges() {
        let graph = TestGraph {
            name: "test-ir".to_string(),
            blocks: vec![
                TestBlock {
                    id: "entry".to_string(),
                    instrs: vec![TestInstr { name: "br".to_string(), ty: None }],
                    succs: vec!["left".to_string(), "right".to_string()],
                    back: vec![],
                    attrs: vec![],
                    depth: 0,
                },
                TestBlock {
                    id: "left".to_string(),
                    instrs: vec![TestInstr { name: "add".to_string(), ty: Some("i32".to_string()) }],
                    succs: vec!["exit".to_string()],
                    back: vec![],
                    attrs: vec![],
                    depth: 0,
                },
                TestBlock {
                    id: "right".to_string(),
                    instrs: vec![TestInstr { name: "sub".to_string(), ty: Some("i32".to_string()) }],
                    succs: vec!["exit".to_string()],
                    back: vec![],
                    attrs: vec![],
                    depth: 0,
                },
                TestBlock {
                    id: "exit".to_string(),
                    instrs: vec![TestInstr { name: "ret".to_string(), ty: None }],
                    succs: vec![],
                    back: vec![],
                    attrs: vec![],
                    depth: 0,
                },
            ],
        };

        let svg = graph.render_svg();
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_trait_to_universal_ir() {
        let graph = TestGraph {
            name: "test".to_string(),
            blocks: vec![
                TestBlock {
                    id: "a".to_string(),
                    instrs: vec![TestInstr { name: "jmp".to_string(), ty: None }],
                    succs: vec!["b".to_string()],
                    back: vec![],
                    attrs: vec!["entry".to_string()],
                    depth: 0,
                },
                TestBlock {
                    id: "b".to_string(),
                    instrs: vec![TestInstr { name: "ret".to_string(), ty: None }],
                    succs: vec![],
                    back: vec![],
                    attrs: vec![],
                    depth: 0,
                },
            ],
        };

        let ir = graph.to_universal_ir();
        assert_eq!(ir.compiler, "test");
        assert_eq!(ir.blocks.len(), 2);
        assert_eq!(ir.blocks[0].id, "a");
        assert_eq!(ir.blocks[0].successors, vec!["b"]);
        assert_eq!(ir.blocks[0].attributes, vec!["entry"]);
        assert_eq!(ir.blocks[1].predecessors, vec!["a"]);
    }

    #[test]
    fn test_universal_ir_render_trait() {
        // UniversalIR itself implements Render
        let ir = UniversalIR {
            format: UNIVERSAL_VERSION.to_string(),
            compiler: "test".to_string(),
            metadata: HashMap::new(),
            blocks: vec![
                UniversalBlock {
                    id: "0".to_string(),
                    attributes: vec![],
                    loop_depth: 0,
                    predecessors: vec![],
                    successors: vec![],
                    back_edges: vec![],
                    has_self_loop: false,
                    instructions: vec![
                        UniversalInstruction {
                            opcode: "nop".to_string(),
                            attributes: vec![],
                            type_: None,
                            profiling: None,
                            metadata: HashMap::new(),
                        },
                    ],
                    metadata: HashMap::new(),
                },
            ],
        };

        // Use the Render trait method
        let svg = Render::render_svg(&ir);
        assert!(svg.starts_with("<svg"));
    }
}
