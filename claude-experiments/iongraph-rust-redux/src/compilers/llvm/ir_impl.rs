// LLVM MIR implementation of CompilerIR traits
use crate::core::ir_traits::{CompilerIR, IRInstruction, IRBlock, BlockPtr};
use crate::core::semantic_attrs::{AttributeSemantics, SemanticAttribute};
use crate::layout_provider::LayoutProvider;
use super::schema::{LLVMModule, LLVMBasicBlock, LLVMInstruction};
use std::collections::HashMap;

/// LLVM MIR compiler implementation
pub struct LLVMIR;

impl CompilerIR for LLVMIR {
    fn format_id() -> &'static str {
        "llvm-mir"
    }

    fn version() -> u32 {
        1
    }

    type Instruction = LLVMInstruction;
    type Block = LLVMBlockWithIndices;
    type Container = LLVMModule;

    fn extract_blocks(container: &Self::Container) -> Vec<Self::Block> {
        // Flatten all functions and convert block labels to indices
        let mut all_blocks = Vec::new();

        for func in &container.functions {
            // Create label -> index mapping for this function
            let label_to_index: HashMap<String, usize> = func
                .blocks
                .iter()
                .enumerate()
                .map(|(idx, block)| (block.label.clone(), all_blocks.len() + idx))
                .collect();

            // Convert blocks with resolved indices
            for block in &func.blocks {
                let predecessors: Vec<usize> = block
                    .predecessors
                    .iter()
                    .filter_map(|label| label_to_index.get(label).copied())
                    .collect();

                let successors: Vec<usize> = block
                    .successors
                    .iter()
                    .filter_map(|label| label_to_index.get(label).copied())
                    .collect();

                all_blocks.push(LLVMBlockWithIndices {
                    inner: block.clone(),
                    predecessor_indices: predecessors,
                    successor_indices: successors,
                });
            }
        }

        all_blocks
    }

    fn attribute_colors() -> HashMap<String, String> {
        HashMap::from([
            ("nounwind".into(), "#4ec9b0".into()),
            ("readonly".into(), "#569cd6".into()),
            ("noalias".into(), "#c586c0".into()),
            ("inbounds".into(), "#6a9955".into()),
            ("nsw".into(), "#ce9178".into()),
            ("nuw".into(), "#dcdcaa".into()),
        ])
    }
}

/// LLVM basic block with resolved predecessor/successor indices
#[derive(Debug, Clone)]
pub struct LLVMBlockWithIndices {
    inner: LLVMBasicBlock,
    predecessor_indices: Vec<usize>,
    successor_indices: Vec<usize>,
}

impl IRInstruction for LLVMInstruction {
    fn opcode(&self) -> &str {
        &self.opcode
    }

    fn attributes(&self) -> &[String] {
        &self.attributes
    }

    fn type_annotation(&self) -> Option<&str> {
        Some(&self.ty)
    }

    fn profiling_data(&self) -> Option<Vec<u64>> {
        None // LLVM MIR doesn't include profiling data by default
    }

    fn render_row<P: LayoutProvider>(&self, provider: &mut P, _id: usize) -> Box<P::Element> {
        let mut row = provider.create_element("tr");
        provider.add_class(&mut row, "ig-ins");

        // Add instruction attributes as classes
        for attr in &self.attributes {
            let class_name = format!("ig-ins-att-{}", attr);
            provider.add_class(&mut row, &class_name);
        }

        // LLVM instruction format: %result = opcode type operands

        // Result column (if present)
        if let Some(ref result) = self.result {
            let mut result_cell = provider.create_element("td");
            provider.add_class(&mut result_cell, "ig-ins-result");
            provider.set_inner_text(&mut result_cell, result);
            provider.append_child(&mut row, result_cell);

            // Add "=" separator
            let mut eq_cell = provider.create_element("td");
            provider.add_class(&mut eq_cell, "ig-ins-eq");
            provider.set_inner_text(&mut eq_cell, "=");
            provider.append_child(&mut row, eq_cell);
        }

        // Opcode column
        let mut opcode_cell = provider.create_element("td");
        provider.add_class(&mut opcode_cell, "ig-ins-opcode");
        provider.set_inner_text(&mut opcode_cell, &self.opcode);
        provider.append_child(&mut row, opcode_cell);

        // Type column
        let mut type_cell = provider.create_element("td");
        provider.add_class(&mut type_cell, "ig-ins-type");
        provider.set_inner_text(&mut type_cell, &self.ty);
        provider.append_child(&mut row, type_cell);

        // Operands column
        if !self.operands.is_empty() {
            let mut operands_cell = provider.create_element("td");
            provider.add_class(&mut operands_cell, "ig-ins-operands");
            let operands_str = self.operands.join(", ");
            provider.set_inner_text(&mut operands_cell, &operands_str);
            provider.append_child(&mut row, operands_cell);
        }

        row
    }
}

impl IRBlock for LLVMBlockWithIndices {
    type Instruction = LLVMInstruction;

    fn ptr(&self) -> BlockPtr {
        self.inner.label.clone()
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

/// LLVM attribute semantics mapping
impl AttributeSemantics for LLVMIR {
    fn parse_attribute(attr: &str) -> SemanticAttribute {
        match attr {
            "loop.header" => SemanticAttribute::LoopHeader,
            "loop.latch" => SemanticAttribute::Backedge,
            _ => SemanticAttribute::Custom,
        }
    }
}
