//! Translation from input IR to SSA form.
//!
//! This module translates `InputInstr` to `SsaInstr` via:
//! 1. Building a CFG from the input
//! 2. Running the SSA construction algorithm
//! 3. Validating the result

use std::collections::{HashMap, HashSet};

use crate::cfg::{Cfg, CfgBlockId, CfgBuilder};
use crate::cfg_validation::{validate_cfg, CfgViolation};
use crate::linear::input::{InputInstr, InputValue, Label};
use crate::linear::ssa::{SsaInstr, SsaInstrFactory, SsaValue};
use crate::translator::SSATranslator;
use crate::types::BlockId;
use crate::validation::{validate_ssa, SSAViolation};

/// Type alias for the concrete translator used in linear IR translation.
pub type LinearTranslator = SSATranslator<SsaValue, SsaInstr, SsaInstrFactory>;

/// Result of successful SSA translation.
#[derive(Debug)]
pub struct SsaResult {
    /// The original CFG (for reference)
    pub cfg: Cfg<InputInstr>,
    /// The SSA translator containing blocks and phis
    pub translator: LinearTranslator,
}

impl SsaResult {
    /// Print the SSA form for debugging.
    pub fn print(&self) {
        println!("=== SSA Form ===");
        for block in &self.translator.blocks {
            let preds: Vec<_> = block.predecessors.iter().map(|b| format!("B{}", b.0)).collect();
            println!("\nBlock {} (preds: [{}]):", block.id.0, preds.join(", "));

            // Print phi nodes for this block
            for phi in self.translator.phis.values() {
                if phi.block_id == block.id {
                    let dest_str = phi.dest.as_ref()
                        .map(|v| v.name().to_string())
                        .unwrap_or_else(|| format!("Φ{}", phi.id.0));
                    let operands: Vec<_> = phi.operands.iter().map(|v| format!("{}", v)).collect();
                    println!("  {} = φ({})", dest_str, operands.join(", "));
                }
            }

            // Print instructions
            for instr in &block.instructions {
                println!("  {}", instr);
            }
        }
    }
}

/// Error type for translation failures
#[derive(Debug)]
pub enum TranslationError {
    /// CFG validation failed
    CfgViolations(Vec<CfgViolation<Label>>),
    /// SSA validation failed
    SsaViolations(Vec<SSAViolation<SsaValue>>),
}

/// Translate input IR to SSA form.
///
/// Returns `Ok(SsaResult)` on success, or `Err(violations)` if the result
/// fails CFG or SSA validation (which indicates a bug in the translator).
pub fn translate_to_ssa(
    instructions: Vec<InputInstr>,
) -> Result<SsaResult, TranslationError> {
    // Step 1: Build CFG from input
    let cfg = CfgBuilder::build(instructions);

    // Step 1.5: Validate CFG (skip informational violations like CriticalEdge)
    let cfg_violations: Vec<_> = validate_cfg(&cfg)
        .into_iter()
        .filter(|v| !matches!(v, CfgViolation::CriticalEdge { .. }))
        .filter(|v| !matches!(v, CfgViolation::EmptyBlock { .. })) // Labels create empty blocks
        .collect();
    if !cfg_violations.is_empty() {
        return Err(TranslationError::CfgViolations(cfg_violations));
    }

    // Step 2: Create SSA translator
    type Translator = SSATranslator<SsaValue, SsaInstr, SsaInstrFactory>;
    let mut translator = Translator::new();

    // Step 3: Map CFG blocks to SSA translator blocks
    let mut cfg_to_ssa: HashMap<CfgBlockId, BlockId> = HashMap::new();
    cfg_to_ssa.insert(CfgBlockId(0), BlockId(0)); // Entry block already exists

    for i in 1..cfg.blocks.len() {
        let ssa_block = translator.create_block();
        cfg_to_ssa.insert(CfgBlockId(i), ssa_block);
    }

    // Step 4: Add predecessors to SSA blocks
    for cfg_block in &cfg.blocks {
        let ssa_block_id = cfg_to_ssa[&cfg_block.id];
        for &pred in &cfg_block.predecessors {
            let ssa_pred = cfg_to_ssa[&pred];
            translator.add_predecessor(ssa_block_id, ssa_pred);
        }
    }

    // Step 5: Identify loop headers (blocks with back edges)
    let mut loop_headers: HashSet<CfgBlockId> = HashSet::new();
    for cfg_block in &cfg.blocks {
        for &pred in &cfg_block.predecessors {
            if pred.0 >= cfg_block.id.0 {
                loop_headers.insert(cfg_block.id);
            }
        }
    }

    // Step 6: Process each block
    for cfg_block in &cfg.blocks {
        let ssa_block_id = cfg_to_ssa[&cfg_block.id];
        translator.current_block = ssa_block_id;

        // Seal non-loop-header blocks if all predecessors processed
        let should_seal = cfg_block.id.0 == 0
            || (!loop_headers.contains(&cfg_block.id)
                && cfg_block.predecessors.iter().all(|p| p.0 < cfg_block.id.0));

        if should_seal {
            translator.seal_block(ssa_block_id);
        }

        // Translate each instruction
        for instr in &cfg_block.instructions {
            if let Some(ssa_instr) =
                translate_instruction(instr, ssa_block_id, &mut translator, &cfg_to_ssa, &cfg)
            {
                translator.emit(ssa_instr);
            }
        }
    }

    // Step 7: Seal remaining blocks (loop headers)
    for cfg_block in &cfg.blocks {
        let ssa_block_id = cfg_to_ssa[&cfg_block.id];
        if !translator.sealed_blocks.contains(&ssa_block_id) {
            translator.seal_block(ssa_block_id);
        }
    }

    // Step 8: Materialize remaining phis to variables
    translator.materialize_all_phis();

    // Step 9: Validate SSA
    let violations = validate_ssa(&translator);
    if !violations.is_empty() {
        return Err(TranslationError::SsaViolations(violations));
    }

    Ok(SsaResult {
        cfg,
        translator,
    })
}

/// Translate a single input instruction to SSA form.
fn translate_instruction(
    instr: &InputInstr,
    block_id: BlockId,
    translator: &mut SSATranslator<SsaValue, SsaInstr, SsaInstrFactory>,
    cfg_to_ssa: &HashMap<CfgBlockId, BlockId>,
    cfg: &Cfg<InputInstr>,
) -> Option<SsaInstr> {
    match instr {
        InputInstr::Assign { dest, value } => {
            // Resolve the value to SSA
            let ssa_value = resolve_input_value(value, block_id, translator);

            // Create new SSA variable for destination
            let ssa_dest = translator.get_temp_variable(dest);

            // Record the definition
            translator.write_variable(dest.clone(), block_id, SsaValue::Var(ssa_dest.clone()));

            Some(SsaInstr::Assign {
                dest: ssa_dest,
                value: ssa_value,
            })
        }

        InputInstr::BinOp { dest, left, op, right } => {
            // Resolve operands to SSA
            let ssa_left = resolve_input_value(left, block_id, translator);
            let ssa_right = resolve_input_value(right, block_id, translator);

            // Create new SSA variable for destination
            let ssa_dest = translator.get_temp_variable(dest);

            // Record the definition
            translator.write_variable(dest.clone(), block_id, SsaValue::Var(ssa_dest.clone()));

            Some(SsaInstr::BinOp {
                dest: ssa_dest,
                left: ssa_left,
                op: *op,
                right: ssa_right,
            })
        }

        InputInstr::JumpIf { cond, target } => {
            // Resolve condition to SSA
            let ssa_cond = resolve_input_value(cond, block_id, translator);

            // Look up target block from CFG's label mapping
            let then_block = cfg_to_ssa
                .get(&cfg.label_to_block.get(target).copied().unwrap_or(CfgBlockId(0)))
                .copied()
                .unwrap_or(BlockId(0));
            let else_block = BlockId(block_id.0 + 1); // Fall-through

            Some(SsaInstr::Branch {
                cond: ssa_cond,
                then_block,
                else_block,
            })
        }

        InputInstr::Jump(target) => {
            let target_block = cfg_to_ssa
                .get(&cfg.label_to_block.get(target).copied().unwrap_or(CfgBlockId(0)))
                .copied()
                .unwrap_or(BlockId(0));
            Some(SsaInstr::Jump { target: target_block })
        }

        InputInstr::Return(value) => {
            let ssa_value = resolve_input_value(value, block_id, translator);
            Some(SsaInstr::Return { value: ssa_value })
        }

        // Labels are metadata, don't emit an instruction
        InputInstr::Label(_) => None,
    }
}

/// Resolve an input value to an SSA value.
fn resolve_input_value(
    value: &InputValue,
    block_id: BlockId,
    translator: &mut SSATranslator<SsaValue, SsaInstr, SsaInstrFactory>,
) -> SsaValue {
    match value {
        InputValue::Var(name) => {
            // Look up the current SSA value for this variable
            translator.read_variable(name.clone(), block_id)
        }
        InputValue::Const(n) => SsaValue::Const(*n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::input::{BinOp, InputInstr, InputValue, Label};

    #[test]
    fn test_simple_assign() {
        let program = vec![
            InputInstr::Assign {
                dest: "x".into(),
                value: InputValue::Const(42),
            },
            InputInstr::Return(InputValue::Var("x".into())),
        ];

        let result = translate_to_ssa(program);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());

        let ssa = result.unwrap();
        ssa.print();

        // Verify: one block, no phis
        assert_eq!(ssa.translator.blocks.len(), 1);
        assert!(ssa.translator.phis.is_empty());
    }

    #[test]
    fn test_multiple_assigns_same_var() {
        // x = 1; x = 2; return x
        // Should create v0 := 1, v1 := 2, return v1
        let program = vec![
            InputInstr::Assign {
                dest: "x".into(),
                value: InputValue::Const(1),
            },
            InputInstr::Assign {
                dest: "x".into(),
                value: InputValue::Const(2),
            },
            InputInstr::Return(InputValue::Var("x".into())),
        ];

        let result = translate_to_ssa(program);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());

        let ssa = result.unwrap();
        ssa.print();

        // Should have 2 assign instructions (v0 := 1, v1 := 2) + return
        assert_eq!(ssa.translator.blocks[0].instructions.len(), 3);
    }

    #[test]
    fn test_if_else_creates_phi() {
        // x = 1
        // if x jump then
        // y = 10
        // jump end
        // then: y = 20
        // end: return y
        let program = vec![
            InputInstr::Assign {
                dest: "x".into(),
                value: InputValue::Const(1),
            },
            InputInstr::JumpIf {
                cond: InputValue::Var("x".into()),
                target: Label::new("then"),
            },
            InputInstr::Assign {
                dest: "y".into(),
                value: InputValue::Const(10),
            },
            InputInstr::Jump(Label::new("end")),
            InputInstr::Label(Label::new("then")),
            InputInstr::Assign {
                dest: "y".into(),
                value: InputValue::Const(20),
            },
            InputInstr::Label(Label::new("end")),
            InputInstr::Return(InputValue::Var("y".into())),
        ];

        let result = translate_to_ssa(program);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());

        let ssa = result.unwrap();
        ssa.print();

        // Should have phi for y at merge point
        assert!(!ssa.translator.phis.is_empty(), "Expected phi node for y");
    }

    #[test]
    fn test_loop_creates_phi() {
        // sum = 0; i = 0
        // loop: sum = sum + i; i = i + 1
        // if i < 10 jump loop
        // return sum
        let program = vec![
            InputInstr::Assign {
                dest: "sum".into(),
                value: InputValue::Const(0),
            },
            InputInstr::Assign {
                dest: "i".into(),
                value: InputValue::Const(0),
            },
            InputInstr::Label(Label::new("loop")),
            InputInstr::BinOp {
                dest: "sum".into(),
                left: InputValue::Var("sum".into()),
                op: BinOp::Add,
                right: InputValue::Var("i".into()),
            },
            InputInstr::BinOp {
                dest: "i".into(),
                left: InputValue::Var("i".into()),
                op: BinOp::Add,
                right: InputValue::Const(1),
            },
            InputInstr::BinOp {
                dest: "cond".into(),
                left: InputValue::Var("i".into()),
                op: BinOp::Lt,
                right: InputValue::Const(10),
            },
            InputInstr::JumpIf {
                cond: InputValue::Var("cond".into()),
                target: Label::new("loop"),
            },
            InputInstr::Return(InputValue::Var("sum".into())),
        ];

        let result = translate_to_ssa(program);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());

        let ssa = result.unwrap();
        ssa.print();

        // Should have phis for sum and i at loop header
        assert!(
            ssa.translator.phis.len() >= 2,
            "Expected phi nodes for sum and i, got {}",
            ssa.translator.phis.len()
        );
    }
}
