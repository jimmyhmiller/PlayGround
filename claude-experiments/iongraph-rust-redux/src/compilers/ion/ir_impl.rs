// Ion implementation of the CompilerIR traits
use crate::core::ir_traits::{CompilerIR, IRInstruction, IRBlock};
use crate::core::semantic_attrs::{AttributeSemantics, SemanticAttribute};
use crate::layout_provider::LayoutProvider;
use super::schema::{IonJSON, MIRInstruction, LIRInstruction, MIRBlock};
use std::collections::HashMap;

/// Ion IR implementation
pub struct IonIR;

impl CompilerIR for IonIR {
    fn format_id() -> &'static str {
        "ion"
    }

    fn version() -> u32 {
        super::schema::CURRENT_VERSION
    }

    type Instruction = MIRInstruction;
    type Block = MIRBlock;
    type Container = IonJSON;

    fn extract_blocks(container: &Self::Container) -> Vec<Self::Block> {
        // Flatten the Ion hierarchy: functions → passes → blocks
        // For now, we just extract MIR blocks from the first pass
        // TODO: Handle multiple functions/passes and LIR
        if !container.functions.is_empty() {
            // New format: multiple functions
            container.functions.iter()
                .flat_map(|f| &f.passes)
                .filter_map(|p| p.mir.as_ref())
                .flat_map(|mir| &mir.blocks)
                .cloned()
                .collect()
        } else if !container.passes.is_empty() {
            // Old format: single function
            container.passes.iter()
                .filter_map(|p| p.mir.as_ref())
                .flat_map(|mir| &mir.blocks)
                .cloned()
                .collect()
        } else {
            vec![]
        }
    }

    fn migrate(container: Self::Container) -> Result<Self::Container, String> {
        // Ion doesn't use the serde_json::Value-based migration in the new trait
        // Migration is handled externally when loading JSON
        Ok(container)
    }

    fn attribute_colors() -> HashMap<String, String> {
        HashMap::from([
            ("Movable".into(), "#1048af".into()),
            ("Guard".into(), "#000000".into()),
            ("RecoveredOnBailout".into(), "#444444".into()),
            ("InWorklist".into(), "red".into()),
        ])
    }
}

impl IRInstruction for MIRInstruction {
    fn opcode(&self) -> &str {
        &self.opcode
    }

    fn attributes(&self) -> &[String] {
        self.attributes.as_ref().map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn type_annotation(&self) -> Option<&str> {
        self.type_.as_deref()
    }

    fn profiling_data(&self) -> Option<Vec<u64>> {
        None // MIR doesn't have profiling data
    }

    fn render_row<P: LayoutProvider>(&self, provider: &mut P, _id: usize) -> Box<P::Element> {
        let mut row = provider.create_element("tr");
        provider.add_class(&mut row, "ig-ins");

        // Add instruction attributes as classes
        if let Some(ref attrs) = self.attributes {
            for attr in attrs {
                let class_name = format!("ig-ins-att-{}", attr);
                provider.add_class(&mut row, &class_name);
            }
        }

        // ID column
        let mut id_cell = provider.create_element("td");
        provider.add_class(&mut id_cell, "ig-ins-num");
        provider.set_inner_text(&mut id_cell, &format!("{}", self.id));
        provider.append_child(&mut row, id_cell);

        // Opcode column
        let mut opcode_cell = provider.create_element("td");
        provider.set_inner_text(&mut opcode_cell, &self.opcode);
        provider.append_child(&mut row, opcode_cell);

        // Type column
        let mut type_cell = provider.create_element("td");
        provider.add_class(&mut type_cell, "ig-ins-type");
        if let Some(ref type_) = self.type_ {
            provider.set_inner_text(&mut type_cell, type_);
        }
        provider.append_child(&mut row, type_cell);

        row
    }
}

impl IRInstruction for LIRInstruction {
    fn opcode(&self) -> &str {
        &self.opcode
    }

    fn attributes(&self) -> &[String] {
        self.attributes.as_ref().map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn type_annotation(&self) -> Option<&str> {
        self.type_.as_deref()
    }

    fn profiling_data(&self) -> Option<Vec<u64>> {
        None // LIR sample counts not implemented yet
    }

    fn render_row<P: LayoutProvider>(&self, provider: &mut P, _id: usize) -> Box<P::Element> {
        let mut row = provider.create_element("tr");
        provider.add_class(&mut row, "ig-ins");

        // Add instruction attributes as classes
        if let Some(ref attrs) = self.attributes {
            for attr in attrs {
                let class_name = format!("ig-ins-att-{}", attr);
                provider.add_class(&mut row, &class_name);
            }
        }

        // ID column
        let mut id_cell = provider.create_element("td");
        provider.add_class(&mut id_cell, "ig-ins-num");
        provider.set_inner_text(&mut id_cell, &format!("{}", self.id));
        provider.append_child(&mut row, id_cell);

        // Opcode column
        let mut opcode_cell = provider.create_element("td");
        provider.set_inner_text(&mut opcode_cell, &self.opcode);
        provider.append_child(&mut row, opcode_cell);

        // TODO: Add sample count columns when sample_counts is implemented
        // if self.sample_counts.is_some() { ... }

        row
    }
}

impl IRBlock for MIRBlock {
    type Instruction = MIRInstruction;

    fn ptr(&self) -> crate::core::ir_traits::BlockPtr {
        format!("{}", self.ptr.0)
    }

    fn attributes(&self) -> &[String] {
        &self.attributes
    }

    fn loop_depth(&self) -> u32 {
        self.loop_depth
    }

    fn predecessors(&self) -> &[usize] {
        // NOTE: This is a temporary implementation
        // The actual predecessor indices need to be computed during graph construction
        // For now, we return empty to satisfy the trait
        &[]
    }

    fn successors(&self) -> &[usize] {
        // NOTE: This is a temporary implementation
        // The actual successor indices need to be computed during graph construction
        // For now, we return empty to satisfy the trait
        &[]
    }

    fn instructions(&self) -> &[Self::Instruction] {
        &self.instructions
    }
}

/// Ion attribute semantics mapping
impl AttributeSemantics for IonIR {
    fn parse_attribute(attr: &str) -> SemanticAttribute {
        match attr {
            "loopheader" => SemanticAttribute::LoopHeader,
            "backedge" => SemanticAttribute::Backedge,
            "splitedge" => SemanticAttribute::SplitEdge,
            _ => SemanticAttribute::Custom,
        }
    }
}
