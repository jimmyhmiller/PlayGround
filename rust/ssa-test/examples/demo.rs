//! Demo: Linear IR → CFG → SSA with visualization
//!
//! Run with: cargo run --example demo

use ssa_test::linear::{translate_to_ssa, BinOp, InputInstr, InputValue, Label, SsaResult, TranslationError};
use ssa_test::visualizer::SSAVisualizer;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("          Linear IR → CFG → SSA Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Example: Compute factorial of n
    //
    // Pseudocode:
    //   n = 5
    //   result = 1
    //   i = 1
    //   loop:
    //     if i > n goto end
    //     result = result * i
    //     i = i + 1
    //     goto loop
    //   end:
    //     return result

    let program = vec![
        // Initialize
        InputInstr::Assign {
            dest: "n".into(),
            value: InputValue::Const(5),
        },
        InputInstr::Assign {
            dest: "result".into(),
            value: InputValue::Const(1),
        },
        InputInstr::Assign {
            dest: "i".into(),
            value: InputValue::Const(1),
        },
        // Loop header
        InputInstr::Label(Label::new("loop")),
        InputInstr::BinOp {
            dest: "done".into(),
            left: InputValue::Var("i".into()),
            op: BinOp::Gt,
            right: InputValue::Var("n".into()),
        },
        InputInstr::JumpIf {
            cond: InputValue::Var("done".into()),
            target: Label::new("end"),
        },
        // Loop body
        InputInstr::BinOp {
            dest: "result".into(),
            left: InputValue::Var("result".into()),
            op: BinOp::Mul,
            right: InputValue::Var("i".into()),
        },
        InputInstr::BinOp {
            dest: "i".into(),
            left: InputValue::Var("i".into()),
            op: BinOp::Add,
            right: InputValue::Const(1),
        },
        InputInstr::Jump(Label::new("loop")),
        // Exit
        InputInstr::Label(Label::new("end")),
        InputInstr::Return(InputValue::Var("result".into())),
    ];

    println!("┌─────────────────────────────────────────┐");
    println!("│           SOURCE PROGRAM                │");
    println!("└─────────────────────────────────────────┘\n");
    println!("  n = 5");
    println!("  result = 1");
    println!("  i = 1");
    println!("  loop:");
    println!("    if i > n goto end");
    println!("    result = result * i");
    println!("    i = i + 1");
    println!("    goto loop");
    println!("  end:");
    println!("    return result\n");

    println!("┌─────────────────────────────────────────┐");
    println!("│           LINEAR IR                     │");
    println!("└─────────────────────────────────────────┘\n");
    for (i, instr) in program.iter().enumerate() {
        println!("  {:2}: {}", i, instr);
    }

    // Translate to SSA
    let result = translate_to_ssa(program);

    match result {
        Ok(ssa) => {
            print_ssa_result(&ssa);
        }
        Err(error) => {
            println!("\n┌─────────────────────────────────────────┐");
            println!("│           VALIDATION FAILED!            │");
            println!("└─────────────────────────────────────────┘\n");
            match error {
                TranslationError::CfgViolations(violations) => {
                    println!("  CFG Validation Errors:");
                    for v in &violations {
                        println!("    - {}", v);
                    }
                }
                TranslationError::SsaViolations(violations) => {
                    println!("  SSA Validation Errors:");
                    for v in &violations {
                        println!("    - {}", v);
                    }
                }
            }
            std::process::exit(1);
        }
    }
}

fn print_ssa_result(ssa: &SsaResult) {
    println!("\n┌─────────────────────────────────────────┐");
    println!("│           CONTROL FLOW GRAPH            │");
    println!("└─────────────────────────────────────────┘\n");
    for block in &ssa.cfg.blocks {
        println!(
            "  Block {} (preds: {:?}, succs: {:?})",
            block.id.0,
            block.predecessors.iter().map(|b| b.0).collect::<Vec<_>>(),
            block.successors.iter().map(|b| b.0).collect::<Vec<_>>()
        );
        for instr in &block.instructions {
            println!("    {}", instr);
        }
        println!();
    }

    println!("┌─────────────────────────────────────────┐");
    println!("│           SSA FORM                      │");
    println!("└─────────────────────────────────────────┘\n");
    for block in &ssa.translator.blocks {
        let preds: Vec<_> = block
            .predecessors
            .iter()
            .map(|b| format!("B{}", b.0))
            .collect();
        println!("  Block {} (preds: [{}])", block.id.0, preds.join(", "));

        // Print phi nodes for this block
        for phi in ssa.translator.phis.values() {
            if phi.block_id == block.id {
                let dest_str = phi.dest.as_ref()
                    .map(|v| v.name().to_string())
                    .unwrap_or_else(|| format!("Φ{}", phi.id.0));
                let operands: Vec<_> = phi.operands.iter().map(|v| format!("{}", v)).collect();
                println!("    {} = φ({})", dest_str, operands.join(", "));
            }
        }

        // Print instructions
        for instr in &block.instructions {
            println!("    {}", instr);
        }
        println!();
    }

    // Use SSAVisualizer for DOT output
    println!("┌─────────────────────────────────────────┐");
    println!("│           GRAPHVIZ DOT OUTPUT           │");
    println!("└─────────────────────────────────────────┘\n");

    let visualizer = SSAVisualizer::new(&ssa.translator);
    let dot = visualizer.generate_dot();
    println!("{}", dot);

    // Save to file and render
    visualizer.render_to_file("ssa_graph.dot").expect("Failed to write DOT file");
    println!("\nDOT file saved to: ssa_graph.dot");
    println!("To render: dot -Tpng ssa_graph.dot -o ssa_graph.png");
}
