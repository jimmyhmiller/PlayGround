// Arc<UnsafeCell<GCRuntime>> is intentional: Arc for reference counting,
// UnsafeCell for interior mutability in a single-threaded runtime context
#![allow(clippy::arc_with_non_send_sync)]

mod clojure_ast;
mod reader;
mod value;

// Our own IR and compiler
mod arm_codegen;
mod arm_instructions;
mod builtins;
mod compiler;
mod gc;
mod gc_runtime;
mod ir;
mod register_allocation;
mod trampoline;

use crate::arm_codegen::Arm64CodeGen;
use crate::clojure_ast::{Expr, analyze_toplevel_tagged};
use crate::compiler::Compiler;
use crate::gc_runtime::GCRuntime;
use crate::reader::{read, read_to_tagged};
use crate::trampoline::Trampoline;
use std::cell::UnsafeCell;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

/// Read input from stdin until delimiters are balanced.
/// Returns None on EOF, Some(input) when we have a balanced expression.
fn read_balanced_input(prompt: &str, continuation_prompt: &str) -> Option<String> {
    let stdin = io::stdin();
    let mut accumulated = String::new();
    let mut is_first_line = true;

    loop {
        // Print appropriate prompt
        if is_first_line {
            print!("{}", prompt);
        } else {
            print!("{}", continuation_prompt);
        }
        io::stdout().flush().unwrap();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF
                if accumulated.is_empty() {
                    return None;
                } else {
                    // Return what we have, even if incomplete
                    return Some(accumulated);
                }
            }
            Ok(_) => {
                // Check for REPL commands (only on first line)
                if is_first_line {
                    let trimmed = line.trim();
                    if trimmed.starts_with('/') {
                        return Some(trimmed.to_string());
                    }
                }

                accumulated.push_str(&line);

                // Check if delimiters are balanced
                if is_balanced(&accumulated) {
                    return Some(accumulated);
                }

                is_first_line = false;
            }
            Err(_) => return None,
        }
    }
}

/// Check if parentheses, brackets, and braces are balanced in the input.
/// Also handles strings (ignoring delimiters inside strings).
fn is_balanced(input: &str) -> bool {
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for ch in input.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }

        if ch == '"' {
            in_string = !in_string;
            continue;
        }

        if in_string {
            continue;
        }

        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            _ => {}
        }
    }

    // Input is balanced if all depths are zero and we're not in a string
    !in_string && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0
}

/// Print a tagged value, matching Clojure's behavior
/// nil prints nothing, other values print their untagged representation
fn print_tagged_value(tagged_value: i64, runtime: &GCRuntime) {
    let tag = tagged_value & 0b111;

    match tagged_value {
        7 => println!("nil"),
        11 => println!("true"),
        3 => println!("false"),
        _ => {
            match tag {
                0b000 => {
                    // Integer - untag and print
                    let untagged = tagged_value >> 3;
                    println!("{}", untagged);
                }
                0b001 => {
                    // Float - heap-allocated, read from heap
                    let float_val = runtime.read_float(tagged_value as usize);
                    println!("{}", float_val);
                }
                0b100 => {
                    // Function pointer
                    println!("#<fn@{:x}>", tagged_value as u64 >> 3);
                }
                0b101 => {
                    // Closure
                    println!("#<closure@{:x}>", tagged_value as u64 >> 3);
                }
                0b010 => {
                    // String - use runtime's format_value
                    let formatted = runtime.format_value(tagged_value as usize);
                    println!("{}", formatted);
                }
                0b110 => {
                    // HeapObject - use runtime's format_value for proper type handling
                    let formatted = runtime.format_value(tagged_value as usize);
                    println!("{}", formatted);
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
    println!("  /ast (+ 1 2)      - Show AST");
    println!("  /ir (+ 1 2)       - Show IR instructions");
    println!("  /asm (+ 1 2)      - Show ARM64 machine code");
    println!();
    println!("GC Commands:");
    println!("  /gc               - Run garbage collection");
    println!("  /gc-always        - Enable GC before every allocation (stress test)");
    println!("  /gc-always off    - Disable gc-always mode");
    println!();
    println!("Heap Inspection:");
    println!("  /heap             - Show basic heap statistics");
    println!("  /stats            - Show detailed heap stats (objects by type, free list)");
    println!("  /objects          - List all live objects");
    println!("  /objects <Type>   - List objects by type (String, Var, Namespace, etc.)");
    println!("  /inspect-addr 0x..- Inspect object at address");
    println!("  /refs 0x...       - Find references to an object");
    println!("  /roots            - List all GC roots");
    println!();
    println!("Namespace:");
    println!("  /namespaces       - List all namespaces");
    println!("  /inspect <ns>     - Inspect namespace bindings");
    println!();
    println!("  /help             - Show this help");
    println!("  /quit             - Exit REPL");
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
        Expr::Def {
            name,
            value,
            metadata,
        } => {
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
        Expr::TopLevelDo { forms } => {
            println!("{}TopLevelDo ({} forms, tagged pointers)", prefix, forms.len());
            for (i, ptr) in forms.iter().enumerate() {
                println!("{}  [{}]: 0x{:x}", prefix, i, ptr);
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
        Expr::FieldSet {
            field,
            object,
            value,
        } => {
            println!("{}FieldSet(.-{})", prefix, field);
            println!("{}  object:", prefix);
            print_ast(object, indent + 2);
            println!("{}  value:", prefix);
            print_ast(value, indent + 2);
        }
        Expr::Throw { exception } => {
            println!("{}Throw", prefix);
            println!("{}  exception:", prefix);
            print_ast(exception, indent + 2);
        }
        Expr::Try {
            body,
            catches,
            finally,
        } => {
            println!("{}Try", prefix);
            println!("{}  body:", prefix);
            for (i, expr) in body.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(expr, indent + 3);
            }
            if !catches.is_empty() {
                println!("{}  catches:", prefix);
                for (i, catch) in catches.iter().enumerate() {
                    println!(
                        "{}    [{}] {} {}:",
                        prefix, i, catch.exception_type, catch.binding
                    );
                    for (j, expr) in catch.body.iter().enumerate() {
                        println!("{}      [{}]:", prefix, j);
                        print_ast(expr, indent + 4);
                    }
                }
            }
            if let Some(finally_body) = finally {
                println!("{}  finally:", prefix);
                for (i, expr) in finally_body.iter().enumerate() {
                    println!("{}    [{}]:", prefix, i);
                    print_ast(expr, indent + 3);
                }
            }
        }
        // Protocol system
        Expr::DefProtocol { name, methods } => {
            println!("{}DefProtocol({})", prefix, name);
            for method in methods {
                println!(
                    "{}  method: {} (arities: {:?})",
                    prefix, method.name, method.arities
                );
            }
        }
        Expr::ExtendType {
            type_name,
            implementations,
        } => {
            println!("{}ExtendType({})", prefix, type_name);
            for impl_ in implementations {
                println!("{}  implements: {}", prefix, impl_.protocol_name);
                for method in &impl_.methods {
                    println!("{}    {} {:?}", prefix, method.name, method.params);
                    for (i, expr) in method.body.iter().enumerate() {
                        println!("{}      body[{}]:", prefix, i);
                        print_ast(expr, indent + 4);
                    }
                }
            }
        }
        Expr::ProtocolCall { method_name, args } => {
            println!("{}ProtocolCall({})", prefix, method_name);
            for (i, arg) in args.iter().enumerate() {
                println!("{}  arg[{}]:", prefix, i);
                print_ast(arg, indent + 2);
            }
        }
        Expr::Debugger { expr } => {
            println!("{}Debugger", prefix);
            print_ast(expr, indent + 1);
        }
    }
}

fn print_machine_code(code: &[u32]) {
    println!(
        "\nMachine Code ({} instructions, {} bytes):",
        code.len(),
        code.len() * 4
    );
    for (i, instruction) in code.iter().enumerate() {
        println!(
            "  {:04x}: {:08x}  ; {}",
            i * 4,
            instruction,
            disassemble_arm64(*instruction)
        );
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

/// Load and execute a Clojure file, used for loading clojure.core
fn load_clojure_file(
    filename: &str,
    compiler: &mut Compiler,
    runtime: &Arc<UnsafeCell<GCRuntime>>,
    print_results: bool,
) -> Result<(), String> {
    use std::fs;
    use std::io::BufRead;

    let file = match fs::File::open(filename) {
        Ok(f) => f,
        Err(e) => return Err(format!("Error reading file '{}': {}", filename, e)),
    };

    let reader = std::io::BufReader::new(file);
    let mut accumulated = String::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => return Err(format!("Error reading line: {}", e)),
        };

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }

        accumulated.push_str(&line);
        accumulated.push('\n');

        // Try to read using tagged pointer reader
        let ast = unsafe {
            let rt = &mut *runtime.get();
            match read_to_tagged(&accumulated, rt) {
                Ok(tagged) => {
                    // Register the parsed form as a temporary GC root during analysis
                    let root_id = rt.register_temporary_root(tagged);
                    let result = analyze_toplevel_tagged(rt, tagged);
                    rt.unregister_temporary_root(root_id);
                    match result {
                        Ok(ast) => ast,
                        Err(e) => {
                            // Analysis error - but might be incomplete expression
                            // Try the old reader to distinguish
                            match read(&accumulated) {
                                Ok(_) => return Err(format!("Analysis error: {}", e)),
                                Err(_) => continue, // Incomplete expression
                            }
                        }
                    }
                }
                Err(_) => continue, // Incomplete expression, keep accumulating
            }
        };

        match compiler.compile_toplevel(&ast) {
            Ok(_) => {
                let instructions = compiler.take_instructions();
                let num_locals = compiler.builder.num_locals;

                match Arm64CodeGen::compile_function(&instructions, num_locals, 0) {
                    Ok(compiled) => {
                        // Execute as 0-argument function via trampoline
                        let trampoline = Trampoline::new(64 * 1024);
                        let result = unsafe {
                            trampoline.execute(compiled.code_ptr as *const u8)
                        };

                        if print_results
                            && !matches!(
                                ast,
                                Expr::Def { .. }
                                    | Expr::Ns { .. }
                                    | Expr::Use { .. }
                            )
                        {
                            unsafe {
                                let rt = &*runtime.get();
                                print_tagged_value(result, rt);
                            }
                        }
                    }
                    Err(e) => return Err(format!("Codegen error: {}", e)),
                }
            }
            Err(e) => return Err(format!("Compile error: {}", e)),
        }
        accumulated.clear();
    }

    if !accumulated.trim().is_empty() {
        return Err("Incomplete expression at end of file".to_string());
    }

    Ok(())
}

/// Execute a single Clojure expression (like `clj -e "(+ 1 2)"`)
/// Prints the result of the expression
fn run_expr(expr: &str, gc_always: bool) {
    // Create runtime with GC
    let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());

    // Initialize builtins
    unsafe {
        builtins::initialize_builtins(runtime.clone());
        let rt = &mut *runtime.get();
        builtins::register_builtins(rt);
    }

    // Enable gc-always mode if requested
    if gc_always {
        unsafe {
            let rt = &mut *runtime.get();
            rt.set_gc_always(true);
        }
    }

    // Create compiler
    let mut compiler = Compiler::new(runtime.clone());

    // Load clojure.core first (silently)
    if let Err(e) = load_clojure_file("src/clojure/core.clj", &mut compiler, &runtime, false) {
        eprintln!("Error loading clojure.core: {}", e);
        std::process::exit(1);
    }

    // Set up user namespace (refer clojure.core and switch to it)
    compiler.setup_user_namespace();

    // Parse and analyze the expression using tagged pointers
    let ast = unsafe {
        let rt = &mut *runtime.get();
        let tagged = match read_to_tagged(expr, rt) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Read error: {}", e);
                std::process::exit(1);
            }
        };
        // Register parsed form as temporary root during analysis
        let root_id = rt.register_temporary_root(tagged);
        let result = analyze_toplevel_tagged(rt, tagged);
        rt.unregister_temporary_root(root_id);
        match result {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Analysis error: {}", e);
                std::process::exit(1);
            }
        }
    };

    // Compile the expression
    if let Err(e) = compiler.compile_toplevel(&ast) {
        eprintln!("Compile error: {}", e);
        std::process::exit(1);
    }

    let instructions = compiler.take_instructions();
    let num_locals = compiler.builder.num_locals;

    // Generate machine code
    let compiled = match Arm64CodeGen::compile_function(&instructions, num_locals, 0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Codegen error: {}", e);
            std::process::exit(1);
        }
    };

    // Execute as 0-argument function via trampoline
    let trampoline = Trampoline::new(64 * 1024);
    let result = unsafe { trampoline.execute(compiled.code_ptr as *const u8) };

    unsafe {
        let rt = &*runtime.get();
        print_tagged_value(result, rt);
    }
}

/// Execute a Clojure script file (like `clojure script.clj`)
/// Like Clojure, scripts don't print results - only explicit print calls produce output
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

    // Initialize builtins
    unsafe {
        builtins::initialize_builtins(runtime.clone());
        let rt = &mut *runtime.get();
        builtins::register_builtins(rt);
    }

    // Enable gc-always mode if requested
    if gc_always {
        unsafe {
            let rt = &mut *runtime.get();
            rt.set_gc_always(true);
        }
    }

    // Create compiler
    let mut compiler = Compiler::new(runtime.clone());

    // Load clojure.core first (silently)
    if let Err(e) = load_clojure_file("src/clojure/core.clj", &mut compiler, &runtime, false) {
        eprintln!("Error loading clojure.core: {}", e);
        std::process::exit(1);
    }

    // Set up user namespace (refer clojure.core and switch to it)
    compiler.setup_user_namespace();

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

        // Try to read and analyze using tagged pointers
        let ast = unsafe {
            let rt = &mut *runtime.get();
            match read_to_tagged(&accumulated, rt) {
                Ok(tagged) => {
                    // Register the parsed form as a temporary GC root during analysis
                    // This prevents it from being collected if GC runs during analysis
                    let root_id = rt.register_temporary_root(tagged);
                    let result = analyze_toplevel_tagged(rt, tagged);
                    rt.unregister_temporary_root(root_id);
                    match result {
                        Ok(ast) => ast,
                        Err(e) => {
                            // Analysis error - but might be incomplete expression
                            // Try the old reader to distinguish
                            match read(&accumulated) {
                                Ok(_) => {
                                    eprintln!("Analysis error: {}", e);
                                    std::process::exit(1);
                                }
                                Err(_) => continue, // Incomplete expression
                            }
                        }
                    }
                }
                Err(_) => continue, // Incomplete expression, keep accumulating
            }
        };

        // Compile and execute
        match compiler.compile_toplevel(&ast) {
            Ok(_) => {
                let instructions = compiler.take_instructions();
                let num_locals = compiler.builder.num_locals;

                match Arm64CodeGen::compile_function(&instructions, num_locals, 0) {
                    Ok(compiled) => {
                        // Execute as 0-argument function via trampoline
                        // Like Clojure, scripts don't print results automatically
                        // Only explicit print/println calls produce output
                        let trampoline = Trampoline::new(64 * 1024);
                        let _result = unsafe {
                            trampoline.execute(compiled.code_ptr as *const u8)
                        };
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

        // Clear accumulated for next expression
        accumulated.clear();
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
        // Parse arguments: [binary] [--gc-always] [-e <expr> | <file>]
        let mut gc_always = false;
        let mut expr_mode = false;
        let mut expr_or_file: Option<String> = None;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--gc-always" => gc_always = true,
                "-e" => {
                    expr_mode = true;
                    if i + 1 < args.len() {
                        i += 1;
                        expr_or_file = Some(args[i].clone());
                    }
                }
                _ => {
                    expr_or_file = Some(args[i].clone());
                }
            }
            i += 1;
        }

        if let Some(value) = expr_or_file {
            if expr_mode {
                // Expression mode: evaluate inline expression
                run_expr(&value, gc_always);
            } else {
                // Script mode: execute file without REPL interface
                run_script(&value, gc_always);
            }
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

    // Initialize builtins
    // SAFETY: Called once during initialization
    unsafe {
        builtins::initialize_builtins(runtime.clone());
    }

    // Register builtin functions in the runtime
    // SAFETY: Single-threaded initialization
    unsafe {
        let rt = &mut *runtime.get();
        builtins::register_builtins(rt);
    }

    // Create a persistent compiler to maintain global environment across REPL iterations
    let mut repl_compiler = Compiler::new(runtime.clone());

    // Load clojure.core first (silently)
    if let Err(e) = load_clojure_file("src/clojure/core.clj", &mut repl_compiler, &runtime, false) {
        eprintln!("Warning: Could not load clojure.core: {}", e);
    }

    // Set up user namespace (refer clojure.core and switch to it)
    repl_compiler.setup_user_namespace();

    loop {
        // Show current namespace in prompt
        let ns = repl_compiler.get_current_namespace();
        let prompt = format!("{}=> ", ns);
        let continuation_prompt = format!("{}   ", " ".repeat(ns.len()));

        let input = match read_balanced_input(&prompt, &continuation_prompt) {
            Some(s) => s,
            None => break, // EOF
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Check for REPL commands
        if input == "/help" {
                    print_help();
                    continue;
                }

                if input == "/quit" || input == "/exit" {
                    break;
                }

                if input == "/gc" {
                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &mut *runtime.get();
                        match rt.run_gc() {
                            Ok(_) => {
                                // Sync compiler's namespace registry after GC relocations
                                repl_compiler.sync_namespace_registry();
                                println!("✓ Garbage collection completed");
                            }
                            Err(e) => eprintln!("GC error: {}", e),
                        }
                    }
                    continue;
                }

                if input == "/gc-always" || input == "/gc-always on" {
                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &mut *runtime.get();
                        rt.set_gc_always(true);
                    }
                    println!("✓ gc-always mode ENABLED (GC runs before every allocation)");
                    continue;
                }

                if input == "/gc-always off" {
                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &mut *runtime.get();
                        rt.set_gc_always(false);
                    }
                    println!("✓ gc-always mode DISABLED");
                    continue;
                }

                if input == "/heap" {
                    // SAFETY: REPL command, not during compilation
                    let stats = unsafe {
                        let rt = &*runtime.get();
                        rt.heap_stats()
                    };
                    println!("\n╔════════════════════ Heap Statistics ════════════════════╗");
                    println!(
                        "║ GC Algorithm:    {:>20}                 ║",
                        stats.gc_algorithm
                    );
                    println!(
                        "║ Namespaces:      {:>8}                               ║",
                        stats.namespace_count
                    );
                    println!(
                        "║ Types:           {:>8}                               ║",
                        stats.type_count
                    );
                    println!(
                        "║ Stack Map:       {:>8} entries                      ║",
                        stats.stack_map_entries
                    );
                    println!("╚═════════════════════════════════════════════════════════╝");
                    println!();
                    continue;
                }

                if input == "/namespaces" {
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

                if input.starts_with("/inspect ") {
                    let ns_name = input.trim_start_matches("/inspect ").trim();

                    // SAFETY: REPL command, not during compilation
                    unsafe {
                        let rt = &*runtime.get();

                        // Find namespace pointer
                        if let Some(ns_ptr) = rt
                            .list_namespaces()
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
                                let tag = value & 7;
                                println!(
                                    "  {:12}  0x{:08x}      {} (tag={})",
                                    name, var_ptr, untagged, tag
                                );
                            }
                            println!("╚═══════════════════════════════════════════════════╝\n");
                        } else {
                            eprintln!("Namespace '{}' not found", ns_name);
                        }
                    }
                    continue;
                }

                // ========== Heap Inspection Commands ==========

                if input == "/stats" {
                    // Detailed heap statistics
                    let stats = unsafe {
                        let rt = &*runtime.get();
                        rt.detailed_heap_stats()
                    };
                    println!("\n╔════════════════════ Detailed Heap Stats ════════════════════╗");
                    println!("  GC Algorithm:       {}", stats.gc_algorithm);
                    println!(
                        "  Total Heap:         {} bytes ({:.2} KB)",
                        stats.total_bytes,
                        stats.total_bytes as f64 / 1024.0
                    );
                    println!(
                        "  Used:               {} bytes ({:.2} KB)",
                        stats.used_bytes,
                        stats.used_bytes as f64 / 1024.0
                    );
                    println!("  Live Objects:       {}", stats.object_count);
                    println!();
                    println!("  Objects by Type:");
                    println!("  ─────────────────────────────────────────────────────────────");
                    println!("  Type         Count       Bytes");
                    for (_, type_name, count, bytes) in &stats.objects_by_type {
                        println!("  {:12} {:>6}      {:>8}", type_name, count, bytes);
                    }
                    if let Some(free_entries) = stats.free_list_entries {
                        println!();
                        println!("  Free List (mark-and-sweep):");
                        println!("  ─────────────────────────────────────────────────────────────");
                        println!("  Free Entries:       {}", free_entries);
                        if let Some(free_bytes) = stats.free_bytes {
                            println!("  Free Bytes:         {}", free_bytes);
                        }
                        if let Some(largest) = stats.largest_free_block {
                            println!("  Largest Block:      {} bytes", largest);
                        }
                    }
                    println!("╚═════════════════════════════════════════════════════════════╝\n");
                    continue;
                }

                if input == "/objects" || input.starts_with("/objects ") {
                    // List objects (optionally filtered by type)
                    let type_filter = if input.starts_with("/objects ") {
                        Some(input.trim_start_matches("/objects ").trim())
                    } else {
                        None
                    };

                    let objects = unsafe {
                        let rt = &*runtime.get();
                        if let Some(filter) = type_filter {
                            rt.list_objects_by_type(filter)
                        } else {
                            rt.list_objects()
                        }
                    };

                    if objects.is_empty() {
                        if let Some(filter) = type_filter {
                            println!("No objects of type '{}' found", filter);
                        } else {
                            println!("No objects in heap");
                        }
                    } else {
                        println!(
                            "\n╔═════════════════════════════════ Live Objects ═════════════════════════════════╗"
                        );
                        println!("  Address          Type         Size    Value");
                        println!("  ───────────────────────────────────────────────────────────────────────────────");
                        for obj in &objects {
                            let value_str = obj.value_preview.as_deref().unwrap_or("-");
                            println!(
                                "  0x{:012x}  {:12} {:>5}   {}",
                                obj.address,
                                obj.type_name,
                                obj.size_bytes,
                                value_str
                            );
                        }
                        println!("  ───────────────────────────────────────────────────────────────────────────────");
                        println!("  Total: {} objects", objects.len());
                        println!(
                            "╚═════════════════════════════════════════════════════════════════════════════════╝\n"
                        );
                    }
                    continue;
                }

                if input.starts_with("/inspect-addr ") {
                    // Inspect object at address
                    let addr_str = input.trim_start_matches("/inspect-addr ").trim();
                    let addr = if addr_str.starts_with("0x") || addr_str.starts_with("0X") {
                        usize::from_str_radix(&addr_str[2..], 16)
                    } else {
                        addr_str.parse::<usize>()
                    };

                    match addr {
                        Ok(tagged_ptr) => unsafe {
                            let rt = &*runtime.get();
                            if let Some(info) = rt.inspect_object(tagged_ptr) {
                                println!(
                                    "\n╔═══════════════ Object @ 0x{:x} ═══════════════╗",
                                    info.address
                                );
                                println!("  Type:       {} (id={})", info.type_name, info.type_id);
                                println!("  Size:       {} bytes", info.size_bytes);
                                println!("  Fields:     {}", info.field_count);
                                println!("  TypeData:   {}", info.type_data);
                                println!("  Opaque:     {}", info.is_opaque);

                                let fields = rt.object_fields(tagged_ptr);
                                if !fields.is_empty() {
                                    println!();
                                    println!("  Fields:");
                                    println!("  ───────────────────────────────────────────────");
                                    for (idx, value, desc) in &fields {
                                        println!("  [{}] 0x{:x} = {}", idx, value, desc);
                                    }
                                }
                                println!("╚═══════════════════════════════════════════════════╝\n");
                            } else {
                                eprintln!("No object found at address 0x{:x}", tagged_ptr);
                            }
                        },
                        Err(_) => {
                            eprintln!("Invalid address: {}", addr_str);
                        }
                    }
                    continue;
                }

                if input.starts_with("/refs ") {
                    // Find references to object
                    let addr_str = input.trim_start_matches("/refs ").trim();
                    let addr = if addr_str.starts_with("0x") || addr_str.starts_with("0X") {
                        usize::from_str_radix(&addr_str[2..], 16)
                    } else {
                        addr_str.parse::<usize>()
                    };

                    match addr {
                        Ok(tagged_ptr) => {
                            let refs = unsafe {
                                let rt = &*runtime.get();
                                rt.find_references_to(tagged_ptr)
                            };

                            if refs.is_empty() {
                                println!("No references found to 0x{:x}", tagged_ptr);
                            } else {
                                println!(
                                    "\n╔═══════════ References to 0x{:x} ═══════════╗",
                                    tagged_ptr
                                );
                                println!("  From Address       Field    Tagged Value");
                                println!("  ─────────────────────────────────────────────────");
                                for r in &refs {
                                    println!(
                                        "  0x{:012x}   [{}]      0x{:x}",
                                        r.from_address, r.field_index, r.tagged_value
                                    );
                                }
                                println!("  ─────────────────────────────────────────────────────");
                                println!("  Total: {} references", refs.len());
                                println!(
                                    "╚═══════════════════════════════════════════════════════╝\n"
                                );
                            }
                        }
                        Err(_) => {
                            eprintln!("Invalid address: {}", addr_str);
                        }
                    }
                    continue;
                }

                if input == "/roots" {
                    // List all GC roots
                    let roots = unsafe {
                        let rt = &*runtime.get();
                        rt.list_gc_roots()
                    };

                    if roots.is_empty() {
                        println!("No GC roots");
                    } else {
                        println!("\n╔═══════════════════════ GC Roots ═══════════════════════╗");
                        println!("  Namespace         Symbol           Tagged Pointer");
                        println!("  ───────────────────────────────────────────────────────");
                        for (ns, sym, ptr) in &roots {
                            println!("  {:16} {:16} 0x{:x}", ns, sym, ptr);
                        }
                        println!("  ───────────────────────────────────────────────────────");
                        println!("  Total: {} roots", roots.len());
                        println!("╚═════════════════════════════════════════════════════════╝\n");
                    }
                    continue;
                }

                // Parse command and code
                let (command, code) = if input.starts_with('/') {
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

                // Read and analyze using tagged pointers
                let ast_result = unsafe {
                    let rt = &mut *runtime.get();
                    match read_to_tagged(code, rt) {
                        Ok(tagged) => {
                            // Register parsed form as temporary root during analysis
                            let root_id = rt.register_temporary_root(tagged);
                            let result = analyze_toplevel_tagged(rt, tagged);
                            rt.unregister_temporary_root(root_id);
                            result
                        }
                        Err(e) => Err(format!("Read error: {}", e)),
                    }
                };

                match ast_result {
                    Ok(ast) => {
                        match command {
                                    "/ast" => {
                                        println!("\nAST:");
                                        print_ast(&ast, 0);
                                        println!();
                                    }
                                    "/ir" => {
                                        // Use the REPL compiler so we can access defined vars
                                        match repl_compiler.compile(&ast) {
                                            Ok(_) => {
                                                // First show any nested function IRs
                                                let fn_irs =
                                                    repl_compiler.take_compiled_function_irs();
                                                for (fn_name, fn_instructions) in &fn_irs {
                                                    let name_str = fn_name
                                                        .as_ref()
                                                        .map(|s| s.as_str())
                                                        .unwrap_or("<anonymous>");
                                                    println!(
                                                        "\nFunction '{}' IR ({} instructions):",
                                                        name_str,
                                                        fn_instructions.len()
                                                    );
                                                    for (i, inst) in
                                                        fn_instructions.iter().enumerate()
                                                    {
                                                        println!("  {:3}: {:?}", i, inst);
                                                    }
                                                }

                                                // Then show top-level IR
                                                let instructions =
                                                    repl_compiler.take_instructions();
                                                println!(
                                                    "\nTop-level IR ({} instructions):",
                                                    instructions.len()
                                                );
                                                for (i, inst) in instructions.iter().enumerate() {
                                                    println!("  {:3}: {:?}", i, inst);
                                                }
                                                println!();
                                            }
                                            Err(e) => eprintln!("Compile error: {}", e),
                                        }
                                    }
                                    "/asm" | "/machine" => {
                                        // Use the REPL compiler so we can access defined vars
                                        match repl_compiler.compile_toplevel(&ast) {
                                            Ok(_) => {
                                                let instructions =
                                                    repl_compiler.take_instructions();
                                                let num_locals = repl_compiler.builder.num_locals;

                                                match Arm64CodeGen::compile_function(
                                                    &instructions,
                                                    num_locals,
                                                    0,
                                                ) {
                                                    Ok(compiled) => {
                                                        // Display the machine code by reading from executable memory
                                                        let code_ptr =
                                                            compiled.code_ptr as *const u32;
                                                        let code_len = compiled.code_len;
                                                        unsafe {
                                                            let code_slice =
                                                                std::slice::from_raw_parts(
                                                                    code_ptr, code_len,
                                                                );
                                                            print_machine_code(
                                                                code_slice,
                                                            );
                                                        }
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
                                        match repl_compiler.compile_toplevel(&ast) {
                                            Ok(_) => {
                                                let instructions =
                                                    repl_compiler.take_instructions();
                                                let num_locals = repl_compiler.builder.num_locals;

                                                match Arm64CodeGen::compile_function(
                                                    &instructions,
                                                    num_locals,
                                                    0,
                                                ) {
                                                    Ok(compiled) => {
                                                        // Execute as 0-argument function via trampoline
                                                        let trampoline = Trampoline::new(64 * 1024);
                                                        let result = unsafe {
                                                            trampoline.execute(
                                                                compiled.code_ptr as *const u8,
                                                            )
                                                        };

                                                        // Sync compiler's namespace registry in case GC relocated objects
                                                        // (This can happen during allocation in gc-always mode or when heap is full)
                                                        repl_compiler.sync_namespace_registry();

                                                        // SAFETY: After successful execution, not during compilation
                                                        unsafe {
                                                            let rt = &*runtime.get();
                                                            // If this was a top-level def, print the var instead of the value
                                                            if let Expr::Def { name, .. } = &ast {
                                                                // Look up the var that was just stored
                                                                let ns_name = repl_compiler
                                                                    .get_current_namespace();
                                                                let ns_ptr = rt
                                                                    .list_namespaces()
                                                                    .iter()
                                                                    .find(|(n, _, _)| n == &ns_name)
                                                                    .map(|(_, ptr, _)| *ptr)
                                                                    .unwrap();

                                                                if let Some(var_ptr) = rt
                                                                    .namespace_lookup(ns_ptr, name)
                                                                {
                                                                    let (ns_name, symbol_name) =
                                                                        rt.var_info(var_ptr);
                                                                    println!(
                                                                        "#'{}/{}",
                                                                        ns_name, symbol_name
                                                                    );
                                                                }
                                                            } else {
                                                                // For other expressions, print the result value
                                                                print_tagged_value(result, rt);
                                                            }
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
                                        eprintln!("Type /help for available commands");
                                    }
                                }
                            }
                    Err(e) => eprintln!("Error: {}", e),
                }
    }

    println!("\n👋 Goodbye!");
}
