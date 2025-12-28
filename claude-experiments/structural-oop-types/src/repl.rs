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
Structural OOP Type Checker REPL (JavaScript-like syntax)
==========================================================

Syntax:
  true, false           - Boolean literals
  42, -10               - Integer literals
  x, foo                - Variables
  x => e  or  (x) => e  - Arrow function
  f x                   - Application
  let x = e1 in e2      - Let binding
  cond ? e1 : e2        - Ternary conditional
  { x: e1, y: e2 }      - Object literal
  e.field               - Field access
  this                  - Self-reference (in objects)

Commands:
  :help     - Show this help
  :quit     - Exit the REPL
  :q        - Exit the REPL
  :parse e  - Show parsed AST
  :examples - Show example expressions

Examples:
  x => x                          // Identity function
  { x: 42, y: true }              // Simple object
  { isEmpty: true, get: this }    // Self-referential object
"#;

const EXAMPLES: &str = r#"
Example expressions to try:
===========================

// Basic types
true
42

// Functions
x => x
x => y => x

// Let bindings
let id = x => x in id 42

// Objects
{ x: 42 }
{ x: 42, y: true }
{ name: true, value: 42 }

// Field access
{ x: 42 }.x
let obj = { x: 42, y: true } in obj.x

// Self-reference (recursive types!)
{ self: this }
{ id: x => this }

// Cook-style set
{ isEmpty: true, contains: i => false, insert: i => this }

// Structural polymorphism
let check = s => s.isEmpty in check { isEmpty: true, extra: 42 }

// Ternary conditionals
true ? 1 : 2
let x = true in x ? { a: 1 } : { a: 2 }
"#;

pub fn run_repl() {
    println!("Structural OOP Type Checker");
    println!("Type :help for help, :quit to exit\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("λ> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF
                println!("\nGoodbye!");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                continue;
            }
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Handle commands
        if line.starts_with(':') {
            handle_command(line);
            continue;
        }

        // Type check the expression
        typecheck_and_print(line);
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
