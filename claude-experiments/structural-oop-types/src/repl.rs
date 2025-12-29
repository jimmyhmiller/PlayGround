//! REPL (Read-Eval-Print Loop) for the type checker
//!
//! Commands:
//!   :help     - Show help
//!   :quit     - Exit the REPL
//!   :type <e> - Show the type of expression e (same as just typing e)
//!   :parse <e> - Show the parsed AST
//!   <expr>    - Type check the expression and show its type

use crate::display::display_type;
use crate::infer::infer_expr;
use crate::parser::parse;
use crate::store::NodeStore;
use std::io::{self, BufRead, Write};

const HELP: &str = r#"
Structural OOP Type Checker REPL (JavaScript-like class syntax)
================================================================

Syntax:
  true, false              - Boolean literals
  42, -10                  - Integer literals
  "hello"                  - String literals
  x, foo                   - Variables
  (x) => e                 - Arrow function (single param)
  (x, y) => e              - Arrow function (multi param)
  f(x, y)                  - Function call
  obj.field                - Field access
  obj.method(x)            - Method call
  cond ? e1 : e2           - Ternary conditional
  { x: e1, y: e2 }         - Object literal
  { ...obj, x: e }         - Object with spread (copies obj's fields)
  this                     - Self-reference (in objects/classes)
  s1 ++ s2                 - String concatenation

  class Name(params) {     - Class definition (Cook-style constructor)
    ...inner,              - Spread: copy all fields from inner
    field: value,
    method: (x) => body
  }

  {                        - Block with multiple classes
    class A(x) { ... }
    class B(y) { ... }
    expr
  }

Commands:
  :help     - Show this help
  :quit     - Exit the REPL
  :q        - Exit the REPL
  :parse e  - Show parsed AST
  :examples - Show example expressions

Note: Multi-line input is supported. Keep typing until braces are balanced.
"#;

const EXAMPLES: &str = r#"
Example expressions to try:
===========================

// Basic types
true
42
"hello"

// Functions with call syntax
((x) => x)(42)
((x, y) => x + y)(1, 2)

// Objects
{ x: 42, y: true }
{ x: 42 }.x

// Self-reference (recursive types!)
{ self: this }
{ inc: (n) => this }

// Spread operator - copies all fields from another object
{ ...{ x: 42, y: true }, z: "new" }
(obj) => { ...obj, extra: 1 }

// Cook-style set object
{ isEmpty: true, contains: (i) => false, insert: (i) => this }

// Class definitions
class Box(x) { value: x, get: () => x }
class Counter(n) { value: n, inc: () => this }

// Class with spread - composable wrappers!
{
  class Inner(x) { value: x, double: () => x + x }
  class Wrapper(inner) { ...inner, extra: 99 }
  Wrapper(Inner(21)).extra
}

// Block with multiple classes
{
  class Unit(a) { bind: (k) => k(a), show: "ok" }
  class Error(s) { bind: (k) => this, show: "Error: " ++ s }
  Unit(42).show
}

// Ternary conditionals
true ? 1 : 2
"#;

pub fn run_repl() {
    println!("Structural OOP Type Checker");
    println!("Type :help for help, :quit to exit\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("λ> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        let mut brace_depth = 0i32;
        let mut first_line = true;

        // Read lines until braces are balanced
        loop {
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => {
                    // EOF
                    println!("\nGoodbye!");
                    return;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    break;
                }
            }

            // Update brace depth
            for ch in line.chars() {
                match ch {
                    '{' => brace_depth += 1,
                    '}' => brace_depth -= 1,
                    _ => {}
                }
            }

            input.push_str(&line);

            // If this is a single line without opening brace, or braces are balanced, we're done
            if first_line && brace_depth == 0 {
                break;
            }
            if !first_line && brace_depth <= 0 {
                break;
            }

            first_line = false;

            // Show continuation prompt
            print!(".. ");
            stdout.flush().unwrap();
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with(':') {
            handle_command(input);
            continue;
        }

        // Type check the expression
        typecheck_and_print(input);
    }
}

fn handle_command(line: &str) {
    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ":help" | ":h" | ":?" => {
            println!("{}", HELP);
        }
        ":quit" | ":q" | ":exit" => {
            println!("Goodbye!");
            std::process::exit(0);
        }
        ":examples" | ":ex" => {
            println!("{}", EXAMPLES);
        }
        ":parse" | ":p" => {
            if arg.is_empty() {
                println!("Usage: :parse <expression>");
            } else {
                parse_and_print(arg);
            }
        }
        ":type" | ":t" => {
            if arg.is_empty() {
                println!("Usage: :type <expression>");
            } else {
                typecheck_and_print(arg);
            }
        }
        _ => {
            println!("Unknown command: {}", cmd);
            println!("Type :help for available commands");
        }
    }
}

fn parse_and_print(input: &str) {
    match parse(input) {
        Ok(expr) => {
            println!("AST: {:?}", expr);
        }
        Err(e) => {
            println!("Parse error: {}", e);
        }
    }
}

fn typecheck_and_print(input: &str) {
    // Parse
    let expr = match parse(input) {
        Ok(expr) => expr,
        Err(e) => {
            println!("Parse error: {}", e);
            return;
        }
    };

    // Type check
    let mut store = NodeStore::new();
    match infer_expr(&expr, &mut store) {
        Ok(ty) => {
            let type_str = display_type(&store, ty);
            println!("{}", type_str);
        }
        Err(e) => {
            println!("Type error: {}", e);
        }
    }
}

/// Run the REPL or type-check a single expression
pub fn run(input: Option<&str>) {
    match input {
        Some(expr) => typecheck_and_print(expr),
        None => run_repl(),
    }
}
