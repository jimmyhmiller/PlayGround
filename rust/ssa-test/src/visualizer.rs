use crate::SSATranslator;
use crate::ast::{BinaryOperator, UnaryOperator};
use crate::instruction::{Block, Instruction, Value, Variable};
use std::fs;
use std::process::Command;

pub struct SSAVisualizer<'a> {
    translator: &'a SSATranslator,
}

impl<'a> SSAVisualizer<'a> {
    pub fn new(translator: &'a SSATranslator) -> Self {
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

    fn generate_block_node(&self, block: &Block) -> String {
        let mut node = String::new();

        // Start the block node
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
                        .map(|op| self.format_value(op))
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
            node.push_str(&self.format_instruction(instr));
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

    fn format_instruction(&self, instr: &Instruction) -> String {
        match instr {
            Instruction::Assign { dest, value } => {
                format!(
                    "{} := {}",
                    self.format_variable(dest),
                    self.format_value(value)
                )
            }
            Instruction::BinaryOp {
                dest,
                left,
                op,
                right,
            } => {
                format!(
                    "{} := {} {} {}",
                    self.format_variable(dest),
                    self.format_value(left),
                    self.format_binop(op),
                    self.format_value(right)
                )
            }
            Instruction::UnaryOp { dest, op, operand } => {
                format!(
                    "{} := {} {}",
                    self.format_variable(dest),
                    self.format_unaryop(op),
                    self.format_value(operand)
                )
            }
            Instruction::Jump { target } => {
                format!("jump block_{}", target.0)
            }
            Instruction::ConditionalJump {
                condition,
                true_target,
                false_target,
            } => {
                format!(
                    "if {} then block_{} else block_{}",
                    self.format_value(condition),
                    true_target.0,
                    false_target.0
                )
            }
            Instruction::Print { value } => {
                format!("print {}", self.format_value(value))
            }
        }
    }

    fn format_value(&self, value: &Value) -> String {
        match value {
            Value::Literal(n) => n.to_string(),
            Value::Var(var) => self.format_variable(var),
            Value::Phi(phi_id) => {
                if let Some(phi) = self.translator.phis.get(phi_id) {
                    if phi.operands.is_empty() {
                        format!("Φ{}", phi_id.0)
                    } else {
                        let operands: Vec<String> = phi
                            .operands
                            .iter()
                            .map(|op| self.format_value(op))
                            .collect();
                        format!("Φ{}({})", phi_id.0, operands.join(","))
                    }
                } else {
                    format!("Φ{}", phi_id.0)
                }
            }
            Value::Undefined => "⊥".to_string(),
        }
    }

    fn format_variable(&self, var: &Variable) -> String {
        var.0.clone()
    }

    fn format_binop(&self, op: &BinaryOperator) -> String {
        match op {
            BinaryOperator::Add => "+".to_string(),
            BinaryOperator::Subtract => "-".to_string(),
            BinaryOperator::Multiply => "*".to_string(),
            BinaryOperator::Divide => "/".to_string(),
            BinaryOperator::Equal => "==".to_string(),
            BinaryOperator::NotEqual => "!=".to_string(),
            BinaryOperator::LessThan => "<".to_string(),
            BinaryOperator::LessThanOrEqual => "<=".to_string(),
            BinaryOperator::GreaterThan => ">".to_string(),
            BinaryOperator::GreaterThanOrEqual => ">=".to_string(),
        }
    }

    fn format_unaryop(&self, op: &UnaryOperator) -> String {
        match op {
            UnaryOperator::Negate => "-".to_string(),
            UnaryOperator::Not => "!".to_string(),
        }
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
        // Create a temporary dot file
        let dot_path = format!("{}.dot", png_path.trim_end_matches(".png"));
        self.render_to_file(&dot_path)?;

        // Run graphviz to convert dot to png
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

        // Keep the dot file for inspection (don't delete it)
        // fs::remove_file(dot_path).ok();

        println!("SSA graph rendered to: {}", png_path);
        Ok(())
    }

    pub fn render_and_open(&self, png_path: &str) -> std::io::Result<()> {
        // First render to PNG
        self.render_to_png(png_path)?;

        // Open the PNG file with the default viewer
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

// Helper function to quickly visualize an SSATranslator
pub fn visualize_ssa(translator: &SSATranslator, name: &str) -> std::io::Result<()> {
    let visualizer = SSAVisualizer::new(translator);
    let png_path = format!("{}.png", name);
    visualizer.render_and_open(&png_path)
}
