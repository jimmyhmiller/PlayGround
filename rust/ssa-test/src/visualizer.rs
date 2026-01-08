//! Generic SSA visualizer for Graphviz DOT output.

use std::collections::HashSet;
use std::fs;
use std::process::Command;

use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
use crate::translator::SSATranslator;
use crate::types::{Block, BlockId};

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

        // Find reachable blocks (skip unreachable/empty blocks)
        let reachable = self.find_reachable_blocks();

        // Generate nodes for each reachable block
        for block in &self.translator.blocks {
            if reachable.contains(&block.id) {
                dot.push_str(&self.generate_block_node(block));
            }
        }

        // Generate edges between reachable blocks
        for block in &self.translator.blocks {
            if reachable.contains(&block.id) {
                for pred in &block.predecessors {
                    if reachable.contains(pred) {
                        dot.push_str(&format!("    block_{} -> block_{};\n", pred.0, block.id.0));
                    }
                }
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

    /// Find all blocks reachable from entry.
    /// A block is considered reachable if:
    /// - It's the entry block (BlockId(0)), OR
    /// - It has non-empty instructions AND has predecessors
    fn find_reachable_blocks(&self) -> HashSet<BlockId> {
        let mut reachable = HashSet::new();
        let entry = BlockId(0);

        // Entry is always reachable (even if empty, we show it)
        reachable.insert(entry);

        // BFS from entry following predecessor relationships backwards
        // Actually, we need to follow successor relationships, but we only have predecessors
        // So instead: a block is reachable if it has non-empty instructions OR is entry
        // Then we refine: only include blocks that are actually connected

        // Simple approach: include blocks that have instructions
        // (CfgCleanup clears instructions of unreachable blocks)
        for block in &self.translator.blocks {
            if !block.instructions.is_empty() {
                reachable.insert(block.id);
            }
        }

        reachable
    }

    fn generate_block_node(&self, block: &Block<I>) -> String {
        let mut node = String::new();

        node.push_str(&format!("    block_{} [label=\"", block.id.0));
        node.push_str(&format!("Block {}\\n", block.id.0));
        node.push_str("─────────────\\n");

        // Collect phis that belong to this block
        let block_phis: Vec<_> = self.translator.phis
            .values()
            .filter(|phi| phi.block_id == block.id)
            .collect();

        // Add phi nodes for this block
        for phi in &block_phis {
            // Use phi.dest if available, otherwise show Φ{id}
            let dest_name = phi.dest.as_ref()
                .map(|v| v.name().to_string())
                .unwrap_or_else(|| format!("Φ{}", phi.id.0));

            let operands: Vec<String> = phi
                .operands
                .iter()
                .map(|op| op.format_for_display())
                .collect();

            node.push_str(&format!("{} = φ({})\\n", dest_name, operands.join(", ")));
        }

        if !block_phis.is_empty() {
            node.push_str("─────────────\\n");
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
        !self.translator.phis.is_empty()
    }

    pub fn render_to_file(&self, filepath: &str) -> std::io::Result<()> {
        let dot_content = self.generate_dot();
        fs::write(filepath, dot_content)?;
        Ok(())
    }

    pub fn render_to_png(&self, png_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        use std::process::Stdio;

        let dot_content = self.generate_dot();

        // Pipe DOT content via stdin to avoid creating intermediate files
        let mut child = Command::new("dot")
            .arg("-Tpng")
            .arg("-o")
            .arg(png_path)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(dot_content.as_bytes())?;
        }

        let output = child.wait_with_output()?;

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
