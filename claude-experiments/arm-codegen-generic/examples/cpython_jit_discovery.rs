use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;

    println!("🔍 CPython JIT Instruction Discovery");
    println!("=====================================\n");

    // Instructions we need for CPython JIT:
    let needed = vec![
        ("ADD (register)", vec!["add", "register"]),
        ("SUB (register)", vec!["sub", "register"]),
        ("CMP (register)", vec!["cmp", "register"]),
        ("CBNZ", vec!["cbnz"]),
        ("B.LT", vec!["b.lt", "branch", "less"]),
        ("B.GE", vec!["b.ge", "branch", "greater equal"]),
        ("B.LE", vec!["b.le", "branch", "less equal"]),
        ("B.GT", vec!["b.gt", "branch", "greater"]),
        ("B.EQ", vec!["b.eq", "branch", "equal"]),
        ("B.NE", vec!["b.ne", "branch", "not equal"]),
    ];

    for (name, patterns) in needed {
        println!("🔎 Searching for: {}", name);

        for pattern in patterns {
            let found = arm.find_instructions(pattern);
            if !found.is_empty() {
                println!("  Pattern '{}' found {} matches:", pattern, found.len());
                for instr in found.iter().take(5) {
                    if let Some((_, title)) = arm.instruction_info(instr) {
                        println!("    {} - {}", instr, title);
                    }
                }
                if found.len() > 5 {
                    println!("    ... and {} more", found.len() - 5);
                }
            }
        }
        println!();
    }

    Ok(())
}
