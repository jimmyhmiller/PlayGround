mod value;
mod reader;
mod clojure_ast;
mod eval;

// Our own IR and compiler
mod ir;
mod compiler;
mod arm_codegen;
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

fn print_help() {
    println!("\nClojure REPL Commands:");
    println!("  (+ 1 2)           - Execute expression");
    println!("  :ast (+ 1 2)      - Show AST");
    println!("  :ir (+ 1 2)       - Show IR instructions");
    println!("  :asm (+ 1 2)      - Show ARM64 machine code");
    println!("  :gc               - Run garbage collection");
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

fn main() {
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

                if input == ":heap" {
                    // SAFETY: REPL command, not during compilation
                    let stats = unsafe {
                        let rt = &*runtime.get();
                        rt.heap_stats()
                    };
                    println!("\n╔════════════════════ Heap Statistics ════════════════════╗");
                    println!("║ Heap Size:       {:>8} bytes ({:.1} KB)              ║",
                        stats.heap_size, stats.heap_size as f64 / 1024.0);
                    println!("║ Used:            {:>8} bytes ({:.1} KB)              ║",
                        stats.used_bytes, stats.used_bytes as f64 / 1024.0);
                    println!("║ Free:            {:>8} bytes ({:.1} KB)              ║",
                        stats.free_bytes, stats.free_bytes as f64 / 1024.0);
                    println!("║ Objects:         {:>8}                               ║", stats.object_count);
                    println!("║ Namespaces:      {:>8}                               ║", stats.namespace_count);
                    println!("╚═════════════════════════════════════════════════════════╝");

                    println!("\n  Address      Type         Size  Marked  Name");
                    println!("  ──────────────────────────────────────────────────────");
                    for obj in &stats.objects {
                        let marked = if obj.marked { "✓" } else { " " };
                        let name = obj.name.as_deref().unwrap_or("-");
                        println!("  0x{:08x}  {:10}  {:5}b    {}     {}",
                            obj.address, obj.obj_type, obj.size_bytes, marked, name);
                    }
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
                                                let instructions = repl_compiler.take_instructions();
                                                println!("\nIR ({} instructions):", instructions.len());
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
                                                match codegen.compile(&instructions, &result_reg) {
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
                                                match codegen.compile(&instructions, &result_reg) {
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
                                                                    println!("{}", result);
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
