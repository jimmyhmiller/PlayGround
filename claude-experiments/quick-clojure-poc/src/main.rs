mod value;
mod reader;
mod clojure_ast;

// Our own IR and compiler
mod ir;
mod compiler;
mod arm_codegen;
mod gc;
mod gc_runtime;
mod register_allocation;
mod trampoline;

use std::io::{self, Write};
use std::sync::Arc;
use std::cell::UnsafeCell;
use crate::reader::read;
use crate::clojure_ast::{analyze, Expr};
use crate::compiler::Compiler;
use crate::arm_codegen::Arm64CodeGen;
use crate::gc_runtime::GCRuntime;

/// Print a tagged value, matching Clojure's behavior
/// nil prints nothing, other values print their untagged representation
fn print_tagged_value(tagged_value: i64) {
    let tag = tagged_value & 0b111;

    match tagged_value {
        7 => {}, // nil - print nothing, matching Clojure
        11 => println!("true"),
        3 => println!("false"),
        _ => {
            match tag {
                0b000 => {
                    // Integer - untag and print
                    let untagged = tagged_value >> 3;
                    println!("{}", untagged);
                }
                0b100 => {
                    // Function pointer
                    println!("#<fn@{:x}>", tagged_value as u64 >> 3);
                }
                0b101 => {
                    // Closure
                    println!("#<closure@{:x}>", tagged_value as u64 >> 3);
                }
                0b110 => {
                    // HeapObject (deftype instance, etc.)
                    // TODO: Look up type name from registry
                    println!("#<object@{:x}>", tagged_value as u64 >> 3);
                }
                _ => {
                    // Unknown tag - print raw value
                    println!("#{:x} (tag={})", tagged_value, tag);
                }
            }
        }
    }
}

fn print_help() {
    println!("\nClojure REPL Commands:");
    println!("  (+ 1 2)           - Execute expression");
    println!("  :ast (+ 1 2)      - Show AST");
    println!("  :ir (+ 1 2)       - Show IR instructions");
    println!("  :asm (+ 1 2)      - Show ARM64 machine code");
    println!("  :gc               - Run garbage collection");
    println!("  :gc-always        - Enable GC before every allocation (stress test)");
    println!("  :gc-always off    - Disable gc-always mode");
    println!("  :heap             - Show heap statistics");
    println!("  :namespaces       - List all namespaces");
    println!("  :inspect <ns>     - Inspect namespace bindings");
    println!("  :help             - Show this help");
    println!("  :quit             - Exit REPL");
    println!();
}

fn print_ast(ast: &Expr, indent: usize) {
    let prefix = "  ".repeat(indent);
    match ast {
        Expr::Literal(v) => println!("{}Literal({:?})", prefix, v),
        Expr::Var { namespace, name } => {
            if let Some(ns) = namespace {
                println!("{}Var({}/{})", prefix, ns, name)
            } else {
                println!("{}Var({})", prefix, name)
            }
        }
        Expr::Ns { name } => println!("{}Ns({})", prefix, name),
        Expr::Use { namespace } => println!("{}Use({})", prefix, namespace),
        Expr::Quote(v) => println!("{}Quote({:?})", prefix, v),
        Expr::Def { name, value, metadata } => {
            println!("{}Def", prefix);
            println!("{}  name: {}", prefix, name);
            if let Some(meta) = metadata {
                println!("{}  metadata: {:?}", prefix, meta);
            }
            println!("{}  value:", prefix);
            print_ast(value, indent + 2);
        }
        Expr::Set { var, value } => {
            println!("{}Set", prefix);
            println!("{}  var:", prefix);
            print_ast(var, indent + 2);
            println!("{}  value:", prefix);
            print_ast(value, indent + 2);
        }
        Expr::If { test, then, else_ } => {
            println!("{}If", prefix);
            println!("{}  test:", prefix);
            print_ast(test, indent + 2);
            println!("{}  then:", prefix);
            print_ast(then, indent + 2);
            if let Some(e) = else_ {
                println!("{}  else:", prefix);
                print_ast(e, indent + 2);
            }
        }
        Expr::Do { exprs } => {
            println!("{}Do", prefix);
            for (i, expr) in exprs.iter().enumerate() {
                println!("{}  [{}]:", prefix, i);
                print_ast(expr, indent + 2);
            }
        }
        Expr::Let { bindings, body } => {
            println!("{}Let", prefix);
            println!("{}  bindings:", prefix);
            for (name, value) in bindings {
                println!("{}    {} =", prefix, name);
                print_ast(value, indent + 3);
            }
            println!("{}  body:", prefix);
            for (i, expr) in body.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(expr, indent + 3);
            }
        }
        Expr::Loop { bindings, body } => {
            println!("{}Loop", prefix);
            println!("{}  bindings:", prefix);
            for (name, value) in bindings {
                println!("{}    {} =", prefix, name);
                print_ast(value, indent + 3);
            }
            println!("{}  body:", prefix);
            for (i, expr) in body.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(expr, indent + 3);
            }
        }
        Expr::Recur { args } => {
            println!("{}Recur", prefix);
            for (i, arg) in args.iter().enumerate() {
                println!("{}  [{}]:", prefix, i);
                print_ast(arg, indent + 2);
            }
        }
        Expr::Binding { bindings, body } => {
            println!("{}Binding", prefix);
            println!("{}  bindings:", prefix);
            for (var_name, value_expr) in bindings {
                println!("{}    {}:", prefix, var_name);
                print_ast(value_expr, indent + 3);
            }
            println!("{}  body:", prefix);
            for (i, expr) in body.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(expr, indent + 3);
            }
        }
        Expr::Fn { name, arities } => {
            println!("{}Fn", prefix);
            if let Some(n) = name {
                println!("{}  name: {}", prefix, n);
            }
            println!("{}  arities: {}", prefix, arities.len());
            for (i, arity) in arities.iter().enumerate() {
                println!("{}    arity {}:", prefix, i);
                println!("{}      params: {:?}", prefix, arity.params);
                if let Some(rest) = &arity.rest_param {
                    println!("{}      rest: {}", prefix, rest);
                }
                println!("{}      body ({} exprs)", prefix, arity.body.len());
            }
        }
        Expr::Call { func, args } => {
            println!("{}Call", prefix);
            println!("{}  func:", prefix);
            print_ast(func, indent + 2);
            println!("{}  args:", prefix);
            for (i, arg) in args.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(arg, indent + 3);
            }
        }
        Expr::VarRef { namespace, name } => {
            if let Some(ns) = namespace {
                println!("{}VarRef(#'{}/{})", prefix, ns, name)
            } else {
                println!("{}VarRef(#'{})", prefix, name)
            }
        }
        Expr::DefType { name, fields } => {
            println!("{}DefType", prefix);
            println!("{}  name: {}", prefix, name);
            println!("{}  fields: {:?}", prefix, fields);
        }
        Expr::TypeConstruct { type_name, args } => {
            println!("{}TypeConstruct({})", prefix, type_name);
            println!("{}  args:", prefix);
            for (i, arg) in args.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(arg, indent + 3);
            }
        }
        Expr::FieldAccess { field, object } => {
            println!("{}FieldAccess(.-{})", prefix, field);
            println!("{}  object:", prefix);
            print_ast(object, indent + 2);
        }
    }
}

fn print_machine_code(code: &[u32]) {
    println!("\nMachine Code ({} instructions, {} bytes):", code.len(), code.len() * 4);
    for (i, instruction) in code.iter().enumerate() {
        println!("  {:04x}: {:08x}  ; {}", i * 4, instruction, disassemble_arm64(*instruction));
    }
}

fn disassemble_arm64(inst: u32) -> String {
    // Simple ARM64 disassembly
    match inst {
        0xD65F03C0 => "ret".to_string(),
        i if (i & 0xFFE00000) == 0xD2800000 => {
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            format!("movz x{}, #{}", rd, imm)
        }
        i if (i & 0xFFE00000) == 0xF2A00000 => {
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            format!("movk x{}, #{}, lsl #16", rd, imm)
        }
        i if (i & 0xFFE00000) == 0xF2C00000 => {
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            format!("movk x{}, #{}, lsl #32", rd, imm)
        }
        i if (i & 0xFFE00000) == 0xF2E00000 => {
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            format!("movk x{}, #{}, lsl #48", rd, imm)
        }
        i if (i & 0xFFC00000) == 0xF9400000 => {
            // LDR Xd, [Xn, #offset]
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let offset = ((i >> 10) & 0xFFF) * 8; // Scaled offset
            format!("ldr x{}, [x{}, #{}]", rd, rn, offset)
        }
        i if (i & 0xFFC00000) == 0xF9000000 => {
            // STR Xt, [Xn, #offset]
            let rt = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let offset = ((i >> 10) & 0xFFF) * 8; // Scaled offset
            format!("str x{}, [x{}, #{}]", rt, rn, offset)
        }
        i if (i & 0xFFE00000) == 0x8B000000 => {
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("add x{}, x{}, x{}", rd, rn, rm)
        }
        i if (i & 0xFFE00000) == 0xCB000000 => {
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("sub x{}, x{}, x{}", rd, rn, rm)
        }
        i if (i & 0xFFE00000) == 0x9B000000 => {
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("mul x{}, x{}, x{}", rd, rn, rm)
        }
        i if (i & 0xFFC00000) == 0x91000000 => {
            // ADD (immediate) 64-bit
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let imm = (i >> 10) & 0xFFF;
            let shift = if (i >> 22) & 1 == 1 { 12 } else { 0 };
            let actual_imm = imm << shift;
            if rn == 31 && actual_imm == 0 {
                format!("mov x{}, sp", rd)
            } else if rn == 31 {
                format!("add x{}, sp, #{}", rd, actual_imm)
            } else {
                format!("add x{}, x{}, #{}", rd, rn, actual_imm)
            }
        }
        i if (i & 0xFFC00000) == 0xD1000000 => {
            // SUB (immediate) 64-bit
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let imm = (i >> 10) & 0xFFF;
            let shift = if (i >> 22) & 1 == 1 { 12 } else { 0 };
            let actual_imm = imm << shift;
            if rd == 31 {
                format!("sub sp, x{}, #{}", rn, actual_imm)
            } else if rn == 31 {
                format!("sub x{}, sp, #{}", rd, actual_imm)
            } else {
                format!("sub x{}, x{}, #{}", rd, rn, actual_imm)
            }
        }
        i if (i & 0xFF800000) == 0x92800000 => {
            // MOVN (move wide with NOT) 64-bit
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            let hw = (i >> 21) & 0x3;
            format!("movn x{}, #{}, lsl #{}", rd, imm, hw * 16)
        }
        i if (i & 0xFFC00000) == 0x92400000 => {
            // AND (immediate) 64-bit
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            // Simplified - just show the raw immediate fields
            let imms = (i >> 10) & 0x3F;
            format!("and x{}, x{}, #<imm:{}>", rd, rn, imms)
        }
        i if (i & 0xFFC00000) == 0xF1000000 => {
            // SUBS (immediate) - CMP when Rd=XZR
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let imm = (i >> 10) & 0xFFF;
            if rd == 31 {
                format!("cmp x{}, #{}", rn, imm)
            } else {
                format!("subs x{}, x{}, #{}", rd, rn, imm)
            }
        }
        i if (i & 0xFFE0FC00) == 0xD3400000 => {
            // LSR (logical shift right) - UBFM alias
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let immr = (i >> 16) & 0x3F;
            format!("lsr x{}, x{}, #{}", rd, rn, immr)
        }
        i if (i & 0xFFE00000) == 0xAA000000 => {
            let rd = i & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("mov x{}, x{}", rd, rm)
        }
        i if (i & 0xFFFFFC1F) == 0xD63F0000 => {
            // BLR Xn - Branch with Link to Register
            let rn = (i >> 5) & 0x1F;
            format!("blr x{}", rn)
        }
        i if (i & 0xFFC00000) == 0x93400000 => {
            // ASR (arithmetic shift right) - SBFM alias for 64-bit
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let shift = (i >> 16) & 0x3F;
            format!("asr x{}, x{}, #{}", rd, rn, shift)
        }
        i if (i & 0xFFC00000) == 0xD3400000 => {
            // LSL (logical shift left) - UBFM alias for 64-bit
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let immr = (i >> 16) & 0x3F;
            let shift = (64 - immr) & 0x3F;
            format!("lsl x{}, x{}, #{}", rd, rn, shift)
        }
        i if (i & 0xFF000000) == 0x54000000 => {
            // B.cond (conditional branch)
            let cond = i & 0xF;
            let offset = ((i >> 5) & 0x7FFFF) as i32;
            // Sign extend 19-bit offset
            let offset = if offset & 0x40000 != 0 {
                offset | !0x7FFFF
            } else {
                offset
            };
            let cond_name = match cond {
                0 => "eq",
                1 => "ne",
                10 => "ge",
                11 => "lt",
                12 => "gt",
                13 => "le",
                _ => "??",
            };
            format!("b.{} #{}", cond_name, offset * 4)
        }
        i if (i & 0xFC000000) == 0x14000000 => {
            // B (unconditional branch)
            let offset = (i & 0x03FFFFFF) as i32;
            // Sign extend 26-bit offset
            let offset = if offset & 0x02000000 != 0 {
                offset | !0x03FFFFFF
            } else {
                offset
            };
            format!("b #{}", offset * 4)
        }
        i if (i & 0xFF20FC1F) == 0xEB00001F => {
            // CMP (compare - SUBS XZR, Xn, Xm)
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("cmp x{}, x{}", rn, rm)
        }
        i if (i & 0xFFE00000) == 0x9A800000 => {
            // CSINC/CSET (conditional select increment)
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            let cond = (i >> 12) & 0xF;
            // Check if it's CSET (CSINC Xd, XZR, XZR, invert(cond))
            if rn == 31 && rm == 31 {
                let inverted_cond = cond;
                let actual_cond = inverted_cond ^ 1;
                let cond_name = match actual_cond {
                    0 => "eq",
                    1 => "ne",
                    10 => "ge",
                    11 => "lt",
                    12 => "gt",
                    13 => "le",
                    _ => "??",
                };
                format!("cset x{}, {}", rd, cond_name)
            } else {
                format!("csinc x{}, x{}, x{}, #{}", rd, rn, rm, cond)
            }
        }
        i if (i & 0xFFC00000) == 0xA9800000 => {
            // STP (Store Pair, pre-indexed)
            let rt = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rt2 = (i >> 10) & 0x1F;
            let offset = ((i >> 15) & 0x7F) as i32;
            // Sign extend 7-bit offset and multiply by 8
            let offset = if offset & 0x40 != 0 {
                (offset | !0x7F) * 8
            } else {
                offset * 8
            };
            format!("stp x{}, x{}, [x{}, #{}]!", rt, rt2, rn, offset)
        }
        i if (i & 0xFFC00000) == 0xA8C00000 => {
            // LDP (Load Pair, post-indexed)
            let rt = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rt2 = (i >> 10) & 0x1F;
            let offset = ((i >> 15) & 0x7F) as i32;
            // Sign extend 7-bit offset and multiply by 8
            let offset = if offset & 0x40 != 0 {
                (offset | !0x7F) * 8
            } else {
                offset * 8
            };
            format!("ldp x{}, x{}, [x{}], #{}", rt, rt2, rn, offset)
        }
        _ => format!("<unknown: {:08x}>", inst),
    }
}

/// Execute a Clojure script file (like `clojure script.clj`)
/// Prints results of top-level expressions, but not def/ns/use
fn run_script(filename: &str, gc_always: bool) {
    use std::fs;
    use std::io::BufRead;

    // Read file
    let file = match fs::File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            std::process::exit(1);
        }
    };

    // Create runtime with GC
    let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());

    // Enable gc-always mode if requested
    if gc_always {
        unsafe {
            let rt = &mut *runtime.get();
            rt.set_gc_always(true);
        }
    }

    // Create compiler
    let mut compiler = Compiler::new(runtime.clone());

    // Read and accumulate lines until we have a complete expression
    let reader = std::io::BufReader::new(file);
    let mut accumulated = String::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading line: {}", e);
                std::process::exit(1);
            }
        };

        // Skip comments and empty lines
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }

        accumulated.push_str(&line);
        accumulated.push('\n');

        // Try to read an expression
        match read(&accumulated) {
            Ok(val) => {
                // We got a complete expression, analyze and execute it
                match analyze(&val) {
                    Ok(ast) => {
                        // Compile and execute
                        match compiler.compile(&ast) {
                            Ok(result_reg) => {
                                let instructions = compiler.take_instructions();

                                // DEBUG: Print ALL IR
                                if instructions.len() > 2 {
                                    eprintln!("\n===== IR for {:?} ({} instructions) =====",
                                        match &ast {
                                            Expr::Def { name, .. } => format!("def {}", name),
                                            _ => "expression".to_string()
                                        },
                                        instructions.len());
                                    for (i, inst) in instructions.iter().enumerate() {
                                        eprintln!("{:3}: {:?}", i, inst);
                                    }
                                    eprintln!("Result register: {:?}\n", result_reg);
                                }

                                let mut codegen = Arm64CodeGen::new();

                                match codegen.compile(&instructions, &result_reg, 0) {
                                    Ok(_) => {
                                        match codegen.execute() {
                                            Ok(result) => {
                                                // Only print result for non-def/ns/use expressions
                                                // This matches Clojure's behavior in script mode
                                                if !matches!(ast,
                                                    Expr::Def { .. } |
                                                    Expr::Ns { .. } |
                                                    Expr::Use { .. }
                                                ) {
                                                    print_tagged_value(result);
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("Execution error: {}", e);
                                                std::process::exit(1);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Codegen error: {}", e);
                                        std::process::exit(1);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Compile error: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Analysis error: {}", e);
                        std::process::exit(1);
                    }
                }

                // Clear accumulated for next expression
                accumulated.clear();
            }
            Err(_) => {
                // Not a complete expression yet, continue accumulating
                continue;
            }
        }
    }

    // Check if there's any remaining incomplete expression
    if !accumulated.trim().is_empty() {
        eprintln!("Error: Incomplete expression at end of file");
        std::process::exit(1);
    }
}

fn main() {
    // Check for file argument (script mode)
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        // Parse arguments: [binary] [--gc-always] <file>
        let mut gc_always = false;
        let mut filename_idx = 1;

        for (i, arg) in args.iter().enumerate().skip(1) {
            if arg == "--gc-always" {
                gc_always = true;
            } else {
                filename_idx = i;
                break;
            }
        }

        if filename_idx < args.len() {
            // Script mode: execute file without REPL interface
            run_script(&args[filename_idx], gc_always);
        }
        return;
    }

    // REPL mode
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Clojure → ARM64 JIT Compiler                               ║");
    println!("║  Multi-stage compilation: Reader → AST → IR → Machine Code  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    print_help();

    // Create runtime with GC
    // Using UnsafeCell to avoid deadlock during compilation
    // SAFETY: Single-threaded REPL - no concurrent access
    let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));

    // Set global runtime for trampolines
    // SAFETY: Must be called before any JIT code runs
    trampoline::set_runtime(runtime.clone());

    // Create a persistent compiler to maintain global environment across REPL iterations
    let mut repl_compiler = Compiler::new(runtime.clone());

    loop {
        // Show current namespace in prompt
        let ns = repl_compiler.get_current_namespace();
        print!("{}=> ", ns);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                // Check for REPL commands
                if input == ":help" {
                    print_help();
                    continue;
                }

                if input == ":quit" || input == ":exit" {
                    break;
                }

                if input == ":gc" {
                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &mut *runtime.get();
                        match rt.run_gc() {
                            Ok(_) => println!("✓ Garbage collection completed"),
                            Err(e) => eprintln!("GC error: {}", e),
                        }
                    }
                    continue;
                }

                if input == ":gc-always" || input == ":gc-always on" {
                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &mut *runtime.get();
                        rt.set_gc_always(true);
                    }
                    println!("✓ gc-always mode ENABLED (GC runs before every allocation)");
                    continue;
                }

                if input == ":gc-always off" {
                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &mut *runtime.get();
                        rt.set_gc_always(false);
                    }
                    println!("✓ gc-always mode DISABLED");
                    continue;
                }

                if input == ":heap" {
                    // SAFETY: REPL command, not during compilation
                    let stats = unsafe {
                        let rt = &*runtime.get();
                        rt.heap_stats()
                    };
                    println!("\n╔════════════════════ Heap Statistics ════════════════════╗");
                    println!("║ GC Algorithm:    {:>20}                 ║", stats.gc_algorithm);
                    println!("║ Namespaces:      {:>8}                               ║", stats.namespace_count);
                    println!("║ Types:           {:>8}                               ║", stats.type_count);
                    println!("║ Stack Map:       {:>8} entries                      ║", stats.stack_map_entries);
                    println!("╚═════════════════════════════════════════════════════════╝");
                    println!();
                    continue;
                }

                if input == ":namespaces" {
                    // SAFETY: REPL command, not during compilation
                    let namespaces = unsafe {
                        let rt = &*runtime.get();
                        rt.list_namespaces()
                    };
                    println!("\n╔════════════════════ Namespaces ════════════════════╗");
                    println!("  Name                     Pointer      Bindings");
                    println!("  ─────────────────────────────────────────────────");
                    for (name, ptr, bindings) in namespaces {
                        println!("  {:20}  0x{:08x}      {}", name, ptr, bindings);
                    }
                    println!("╚════════════════════════════════════════════════════╝\n");
                    continue;
                }

                if input.starts_with(":inspect ") {
                    let ns_name = input.trim_start_matches(":inspect ").trim();

                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &*runtime.get();

                        // Find namespace pointer
                        if let Some(ns_ptr) = rt.list_namespaces()
                            .iter()
                            .find(|(name, _, _)| name == ns_name)
                            .map(|(_, ptr, _)| *ptr)
                        {
                            let bindings = rt.namespace_bindings(ns_ptr);
                            println!("\n╔════════════ Namespace: {} ════════════╗", ns_name);
                            println!("  Symbol         Var Pointer    Value (untagged)");
                            println!("  ──────────────────────────────────────────────────");
                            for (name, var_ptr) in bindings {
                                let value = rt.var_get_value(var_ptr);
                                let untagged = value >> 3;
                                println!("  {:12}  0x{:08x}      {}", name, var_ptr, untagged);
                            }
                            println!("╚═══════════════════════════════════════════════════╝\n");
                        } else {
                            eprintln!("Namespace '{}' not found", ns_name);
                        }
                    }
                    continue;
                }

                // Parse command and code
                let (command, code) = if input.starts_with(':') {
                    let parts: Vec<&str> = input.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        (parts[0], parts[1])
                    } else {
                        eprintln!("Usage: {} <expression>", parts[0]);
                        continue;
                    }
                } else {
                    ("", input)
                };

                // Read and analyze
                match read(code) {
                    Ok(value) => {
                        match analyze(&value) {
                            Ok(ast) => {
                                match command {
                                    ":ast" => {
                                        println!("\nAST:");
                                        print_ast(&ast, 0);
                                        println!();
                                    }
                                    ":ir" => {
                                        // Use the REPL compiler so we can access defined vars
                                        match repl_compiler.compile(&ast) {
                                            Ok(_) => {
                                                // First show any nested function IRs
                                                let fn_irs = repl_compiler.take_compiled_function_irs();
                                                for (fn_name, fn_instructions) in &fn_irs {
                                                    let name_str = fn_name.as_ref().map(|s| s.as_str()).unwrap_or("<anonymous>");
                                                    println!("\nFunction '{}' IR ({} instructions):", name_str, fn_instructions.len());
                                                    for (i, inst) in fn_instructions.iter().enumerate() {
                                                        println!("  {:3}: {:?}", i, inst);
                                                    }
                                                }

                                                // Then show top-level IR
                                                let instructions = repl_compiler.take_instructions();
                                                println!("\nTop-level IR ({} instructions):", instructions.len());
                                                for (i, inst) in instructions.iter().enumerate() {
                                                    println!("  {:3}: {:?}", i, inst);
                                                }
                                                println!();
                                            }
                                            Err(e) => eprintln!("Compile error: {}", e),
                                        }
                                    }
                                    ":asm" | ":machine" => {
                                        // Use the REPL compiler so we can access defined vars
                                        match repl_compiler.compile(&ast) {
                                            Ok(result_reg) => {
                                                let instructions = repl_compiler.take_instructions();
                                                let mut codegen = Arm64CodeGen::new();
                                                match codegen.compile(&instructions, &result_reg, 0) {
                                                    Ok(code) => {
                                                        print_machine_code(&code);
                                                        println!();
                                                    }
                                                    Err(e) => eprintln!("Codegen error: {}", e),
                                                }
                                            }
                                            Err(e) => eprintln!("Compile error: {}", e),
                                        }
                                    }
                                    "" => {
                                        // Normal execution using IR-based compilation with persistent compiler
                                        match repl_compiler.compile(&ast) {
                                            Ok(result_reg) => {
                                                let instructions = repl_compiler.take_instructions();
                                                let mut codegen = Arm64CodeGen::new();
                                                match codegen.compile(&instructions, &result_reg, 0) {
                                                    Ok(_) => {
                                                        match codegen.execute() {
                                                            Ok(result) => {
                                                                // If this was a top-level def, print the var instead of the value
                                                                if let Expr::Def { name, .. } = &ast {
                                                                    // Look up the var that was just stored
                                                                    // SAFETY: After successful execution, not during compilation
                                                                    unsafe {
                                                                        let rt = &*runtime.get();
                                                                        let ns_name = repl_compiler.get_current_namespace();
                                                                        let ns_ptr = rt.list_namespaces()
                                                                            .iter()
                                                                            .find(|(n, _, _)| n == &ns_name)
                                                                            .map(|(_, ptr, _)| *ptr)
                                                                            .unwrap();

                                                                        if let Some(var_ptr) = rt.namespace_lookup(ns_ptr, name) {
                                                                            let (ns_name, symbol_name) = rt.var_info(var_ptr);
                                                                            println!("#'{}/{}", ns_name, symbol_name);
                                                                        }
                                                                    }
                                                                } else {
                                                                    // For other expressions, print the result value
                                                                    print_tagged_value(result);
                                                                }
                                                            }
                                                            Err(e) => eprintln!("Execution error: {}", e),
                                                        }
                                                    }
                                                    Err(e) => eprintln!("Codegen error: {}", e),
                                                }
                                            }
                                            Err(e) => eprintln!("Compile error: {}", e),
                                        }
                                    }
                                    _ => {
                                        eprintln!("Unknown command: {}", command);
                                        eprintln!("Type :help for available commands");
                                    }
                                }
                            }
                            Err(e) => eprintln!("Parse error: {}", e),
                        }
                    }
                    Err(e) => eprintln!("Read error: {}", e),
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    println!("\n👋 Goodbye!");
}
