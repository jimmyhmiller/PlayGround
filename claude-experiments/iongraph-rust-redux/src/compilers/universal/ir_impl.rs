// Universal IR implementation of CompilerIR traits
use crate::core::ir_traits::{CompilerIR, IRInstruction, IRBlock, BlockPtr};
use crate::core::semantic_attrs::{AttributeSemantics, SemanticAttribute};
use crate::layout_provider::LayoutProvider;
use super::schema::{UniversalIR, UniversalBlock, UniversalInstruction};
use std::collections::HashMap;

/// Universal IR implementation (compiler-agnostic)
pub struct UniversalCompilerIR;

impl CompilerIR for UniversalCompilerIR {
    fn format_id() -> &'static str {
        "universal"
    }

    fn version() -> u32 {
        1
    }

    type Instruction = UniversalInstruction;
    type Block = UniversalBlockWithIndices;
    type Container = UniversalIR;

    fn extract_blocks(container: &Self::Container) -> Vec<Self::Block> {
        // Convert string IDs to indices
        let id_to_index: HashMap<String, usize> = container
            .blocks
            .iter()
            .enumerate()
            .map(|(idx, block)| (block.id.clone(), idx))
            .collect();

        container
            .blocks
            .iter()
            .map(|block| {
                // Convert predecessor/successor IDs to indices
                let predecessors: Vec<usize> = block
                    .predecessors
                    .iter()
                    .filter_map(|id| id_to_index.get(id).copied())
                    .collect();

                let successors: Vec<usize> = block
                    .successors
                    .iter()
                    .filter_map(|id| id_to_index.get(id).copied())
                    .collect();

                UniversalBlockWithIndices {
                    inner: block.clone(),
                    predecessor_indices: predecessors,
                    successor_indices: successors,
                }
            })
            .collect()
    }

    fn migrate(container: Self::Container) -> Result<Self::Container, String> {
        container.validate()?;
        Ok(container)
    }

    fn attribute_colors() -> HashMap<String, String> {
        // Default colors for universal format
        // These can be overridden by compiler-specific implementations
        HashMap::from([
            ("hot".into(), "#ff849e".into()),
            ("cold".into(), "#ffe546".into()),
        ])
    }
}

/// Universal block with resolved predecessor/successor indices
///
/// The universal format uses string IDs for blocks, but our graph structure
/// needs integer indices. This wrapper holds both.
#[derive(Debug, Clone)]
pub struct UniversalBlockWithIndices {
    inner: UniversalBlock,
    predecessor_indices: Vec<usize>,
    successor_indices: Vec<usize>,
}

impl IRInstruction for UniversalInstruction {
    fn opcode(&self) -> &str {
        &self.opcode
    }

    fn attributes(&self) -> &[String] {
        &self.attributes
    }

    fn type_annotation(&self) -> Option<&str> {
        self.type_.as_deref()
    }

    fn profiling_data(&self) -> Option<Vec<u64>> {
        self.profiling.as_ref().map(|p| vec![p.sample_count])
    }

    fn render_row<P: LayoutProvider>(&self, provider: &mut P, _id: usize) -> Box<P::Element> {
        let mut row = provider.create_element("tr");
        provider.add_class(&mut row, "ig-ins");

        // Add instruction attributes as classes
        for attr in &self.attributes {
            let class_name = format!("ig-ins-att-{}", attr);
            provider.add_class(&mut row, &class_name);
        }

        // Opcode column
        let mut opcode_cell = provider.create_element("td");
        provider.set_inner_text(&mut opcode_cell, &self.opcode);
        provider.append_child(&mut row, opcode_cell);

        // Type column (if present)
        if let Some(ref type_) = self.type_ {
            let mut type_cell = provider.create_element("td");
            provider.add_class(&mut type_cell, "ig-ins-type");
            provider.set_inner_text(&mut type_cell, type_);
            provider.append_child(&mut row, type_cell);
        }

        // Profiling data column (if present)
        if let Some(ref profiling) = self.profiling {
            if profiling.sample_count > 0 {
                let mut sample_cell = provider.create_element("td");
                provider.add_class(&mut sample_cell, "ig-ins-samples");
                provider.set_inner_text(&mut sample_cell, &profiling.sample_count.to_string());
                provider.append_child(&mut row, sample_cell);
            }
        }

        row
    }
}

impl IRBlock for UniversalBlockWithIndices {
    type Instruction = UniversalInstruction;

    fn ptr(&self) -> BlockPtr {
        self.inner.id.clone()
    }

    fn attributes(&self) -> &[String] {
        &self.inner.attributes
    }

    fn loop_depth(&self) -> u32 {
        self.inner.loop_depth
    }

    fn predecessors(&self) -> &[usize] {
        &self.predecessor_indices
    }

    fn successors(&self) -> &[usize] {
        &self.successor_indices
    }

    fn instructions(&self) -> &[Self::Instruction] {
        &self.inner.instructions
    }
}

/// Universal attribute semantics mapping
///
/// By default, we use the same attribute names as Ion, but this can be
/// customized per-compiler via the metadata field.
impl AttributeSemantics for UniversalCompilerIR {
    fn parse_attribute(attr: &str) -> SemanticAttribute {
        match attr {
            "loopheader" | "loop.header" | "loop_header" => SemanticAttribute::LoopHeader,
            "backedge" | "loop.latch" | "loop_backedge" => SemanticAttribute::Backedge,
            "splitedge" | "split_edge" => SemanticAttribute::SplitEdge,
            "entry" => SemanticAttribute::Entry,
            "unreachable" => SemanticAttribute::Unreachable,
            _ => SemanticAttribute::Custom,
        }
    }
}
