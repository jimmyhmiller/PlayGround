//! Generic SSA visualizer for Graphviz DOT output.

use std::fs;
use std::process::Command;

use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
use crate::translator::SSATranslator;
use crate::types::Block;

/// Trait for formatting values in visualization
pub trait FormatValue {
    fn format_for_display(&self) -> String;
}

/// Trait for formatting instructions in visualization
pub trait FormatInstruction {
    fn format_for_display(&self) -> String;
}

/// Generic SSA visualizer
pub struct SSAVisualizer<'a, V, I, F>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    translator: &'a SSATranslator<V, I, F>,
}

impl<'a, V, I, F> SSAVisualizer<'a, V, I, F>
where
    V: SsaValue + FormatValue,
    I: SsaInstruction<Value = V> + FormatInstruction,
    F: InstructionFactory<Instr = I>,
{
    pub fn new(translator: &'a SSATranslator<V, I, F>) -> Self {
        SSAVisualizer { translator }
    }

    pub fn generate_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph SSA {\n");
        dot.push_str("    rankdir=TB;\n");
        dot.push_str("    node [shape=box, style=rounded];\n");
        dot.push_str("    \n");

        // Generate nodes for each block
        for block in &self.translator.blocks {
            dot.push_str(&self.generate_block_node(block));
        }

        // Generate edges between blocks
        for block in &self.translator.blocks {
            for pred in &block.predecessors {
                dot.push_str(&format!("    block_{} -> block_{};\n", pred.0, block.id.0));
            }
        }

        // Add legend for phi nodes if any exist
        if self.has_phi_nodes() {
            dot.push_str("\n    // Legend\n");
            dot.push_str("    subgraph cluster_legend {\n");
            dot.push_str("        label=\"Legend\";\n");
            dot.push_str("        style=dotted;\n");
            dot.push_str("        \"Φ = Phi function\" [shape=plaintext];\n");
            dot.push_str("    }\n");
        }

        dot.push_str("}\n");
        dot
    }

    fn generate_block_node(&self, block: &Block<I>) -> String {
        let mut node = String::new();

        node.push_str(&format!("    block_{} [label=\"", block.id.0));
        node.push_str(&format!("Block {}\\n", block.id.0));
        node.push_str("─────────────\\n");

        // Add phi nodes for this block if any
        if let Some(phis) = self.translator.incomplete_phis.get(&block.id) {
            for (var, phi_id) in phis {
                if let Some(phi) = self.translator.phis.get(phi_id) {
                    node.push_str(&format!("Φ({}) = ", var));
                    let operands: Vec<String> = phi
                        .operands
                        .iter()
                        .map(|op| op.format_for_display())
                        .collect();
                    node.push_str(&operands.join(", "));
                    node.push_str("\\n");
                }
            }
            if !phis.is_empty() {
                node.push_str("─────────────\\n");
            }
        }

        // Add instructions
        for instr in &block.instructions {
            node.push_str(&instr.format_for_display());
            node.push_str("\\n");
        }

        // Check if block is sealed
        if self.translator.sealed_blocks.contains(&block.id) {
            node.push_str("─────────────\\n");
            node.push_str("[SEALED]\\n");
        }

        node.push_str("\"];\n");
        node
    }

    fn has_phi_nodes(&self) -> bool {
        !self.translator.incomplete_phis.is_empty()
    }

    pub fn render_to_file(&self, filepath: &str) -> std::io::Result<()> {
        let dot_content = self.generate_dot();
        fs::write(filepath, dot_content)?;
        Ok(())
    }

    pub fn render_to_png(&self, png_path: &str) -> std::io::Result<()> {
        let dot_path = format!("{}.dot", png_path.trim_end_matches(".png"));
        self.render_to_file(&dot_path)?;

        let output = Command::new("dot")
            .arg("-Tpng")
            .arg("-o")
            .arg(png_path)
            .arg(&dot_path)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            eprintln!("Graphviz error: {}", error);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Graphviz failed: {}", error),
            ));
        }

        println!("SSA graph rendered to: {}", png_path);
        Ok(())
    }

    pub fn render_and_open(&self, png_path: &str) -> std::io::Result<()> {
        self.render_to_png(png_path)?;

        #[cfg(target_os = "macos")]
        {
            Command::new("open").arg(png_path).spawn()?;
        }

        #[cfg(target_os = "linux")]
        {
            Command::new("xdg-open").arg(png_path).spawn()?;
        }

        #[cfg(target_os = "windows")]
        {
            Command::new("cmd")
                .args(&["/C", "start", "", png_path])
                .spawn()?;
        }

        Ok(())
    }
}

/// Helper function to quickly visualize an SSATranslator
pub fn visualize_ssa<V, I, F>(translator: &SSATranslator<V, I, F>, name: &str) -> std::io::Result<()>
where
    V: SsaValue + FormatValue,
    I: SsaInstruction<Value = V> + FormatInstruction,
    F: InstructionFactory<Instr = I>,
{
    let visualizer = SSAVisualizer::new(translator);
    let png_path = format!("{}.png", name);
    visualizer.render_and_open(&png_path)
}
