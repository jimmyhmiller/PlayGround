//! Structural OOP Type Checker
//!
//! A type checker for a small object-oriented language with:
//! - Row polymorphism for structural typing
//! - Equi-recursive types for self-reference
//!
//! Based on:
//! - Wand's "Complete Type Inference for Simple Objects" (LICS 1987)
//! - Rémy's "Type Inference for Records in a Natural Extension of ML" (1993)
//! - Cook's "On Understanding Data Abstraction, Revisited" (2009)

mod display;
mod expr;
mod infer;
mod lexer;
mod node;
mod parser;
mod repl;
mod store;
mod types;
mod unify;

#[cfg(test)]
mod tests;

use display::display_type;
use expr::Expr;
use infer::infer_expr;
use store::NodeStore;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("--repl") | Some("-r") => {
            // Interactive REPL mode
            repl::run_repl();
        }
        Some("--examples") | Some("-e") => {
            // Run examples
            run_examples();
        }
        Some("--help") | Some("-h") => {
            print_usage();
        }
        Some(expr) if !expr.starts_with('-') => {
            // Type-check a single expression from command line
            repl::run(Some(expr));
        }
        None => {
            // Default: run REPL
            repl::run_repl();
        }
        Some(arg) => {
            eprintln!("Unknown option: {}", arg);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("Structural OOP Type Checker (JavaScript-like syntax)");
    println!();
    println!("Usage:");
    println!("  structural-oop-types              Start interactive REPL");
    println!("  structural-oop-types --repl       Start interactive REPL");
    println!("  structural-oop-types --examples   Run built-in examples");
    println!("  structural-oop-types '<expr>'     Type-check a single expression");
    println!();
    println!("Examples:");
    println!("  structural-oop-types 'x => x'");
    println!("  structural-oop-types '{{ x: 42, y: true }}'");
    println!("  structural-oop-types '{{ self: this }}'");
}

fn run_examples() {
    println!("=== Structural OOP Type Checker ===\n");

    // Example 1: Simple object with boolean field
    example("Simple object", || {
        Expr::object(vec![("x", Expr::bool(true)), ("y", Expr::int(42))])
    });

    // Example 2: Object with a method
    example("Object with method", || {
        Expr::object(vec![
            ("value", Expr::int(0)),
            ("increment", Expr::lambda("n", Expr::var("n"))),
        ])
    });

    // Example 3: Self-referential object (the key feature!)
    example("Self-referential object", || {
        // { getSelf = this }
        // Should infer: μα. { getSelf: α | ρ }
        Expr::object(vec![("getSelf", Expr::this())])
    });

    // Example 4: Object with method returning self
    example("Method returning self", || {
        // { id = λx. this }
        // Should infer a recursive type where the method returns the object
        Expr::object(vec![("id", Expr::lambda("x", Expr::this()))])
    });

    // Example 5: Cook's Empty set (simplified)
    // Empty = μ this. {
    //   isEmpty  = true,
    //   contains = λi. false,
    // }
    example("Cook's Empty set (simplified)", || {
        Expr::object(vec![
            ("isEmpty", Expr::bool(true)),
            ("contains", Expr::lambda("i", Expr::bool(false))),
        ])
    });

    // Example 6: Object with method that returns this
    // Similar to: insert = λi. this (returning the same set)
    example("Set-like object with identity insert", || {
        Expr::object(vec![
            ("isEmpty", Expr::bool(true)),
            ("contains", Expr::lambda("i", Expr::bool(false))),
            ("insert", Expr::lambda("i", Expr::this())),
        ])
    });

    // Example 7: Nested lambda
    example("Curried function", || {
        // λx. λy. x
        Expr::lambda("x", Expr::lambda("y", Expr::var("x")))
    });

    // Example 8: Identity function
    example("Identity function", || {
        // λx. x
        Expr::lambda("x", Expr::var("x"))
    });

    // Example 9: Application
    example("Function application", || {
        // (λx. x)(true)
        Expr::app(Expr::lambda("x", Expr::var("x")), Expr::bool(true))
    });

    // Example 10: Let binding
    example("Let binding", || {
        // let id = λx. x in id(42)
        Expr::let_(
            "id",
            Expr::lambda("x", Expr::var("x")),
            Expr::app(Expr::var("id"), Expr::int(42)),
        )
    });

    // Example 11: If expression
    example("Conditional", || {
        // if true then 1 else 2
        Expr::if_(Expr::bool(true), Expr::int(1), Expr::int(2))
    });

    // Example 12: Field access
    example("Field access", || {
        // { x = 42 }.x
        Expr::field(Expr::object(vec![("x", Expr::int(42))]), "x")
    });

    // Example 13: Object with method accessing field via this
    example("Method accessing this.field", || {
        // { value = 42, getValue = this.value }
        // Note: this.value returns the field type
        Expr::object(vec![
            ("value", Expr::int(42)),
            ("getValue", Expr::field(Expr::this(), "value")),
        ])
    });

    // =====================================================
    // STRUCTURAL POLYMORPHISM EXAMPLES
    // =====================================================
    println!("\n=== Structural Polymorphism ===\n");

    // Example 14: A function that works with ANY object that has isEmpty
    // checkEmpty : { isEmpty: α | ρ } → α
    example("Polymorphic isEmpty checker", || {
        // λs. s.isEmpty
        Expr::lambda("s", Expr::field(Expr::var("s"), "isEmpty"))
    });

    // Example 15: Apply checkEmpty to simple set (without insert)
    example("checkEmpty applied to simple set", || {
        // let checkEmpty = λs. s.isEmpty in
        // let simpleSet = { isEmpty = true, contains = λi. false } in
        // checkEmpty(simpleSet)
        Expr::let_(
            "checkEmpty",
            Expr::lambda("s", Expr::field(Expr::var("s"), "isEmpty")),
            Expr::let_(
                "simpleSet",
                Expr::object(vec![
                    ("isEmpty", Expr::bool(true)),
                    ("contains", Expr::lambda("i", Expr::bool(false))),
                ]),
                Expr::app(Expr::var("checkEmpty"), Expr::var("simpleSet")),
            ),
        )
    });

    // Example 16: Apply checkEmpty to extended set (with insert)
    example("checkEmpty applied to extended set", || {
        // let checkEmpty = λs. s.isEmpty in
        // let extendedSet = { isEmpty = true, contains = λi. false, insert = λi. this } in
        // checkEmpty(extendedSet)
        Expr::let_(
            "checkEmpty",
            Expr::lambda("s", Expr::field(Expr::var("s"), "isEmpty")),
            Expr::let_(
                "extendedSet",
                Expr::object(vec![
                    ("isEmpty", Expr::bool(true)),
                    ("contains", Expr::lambda("i", Expr::bool(false))),
                    ("insert", Expr::lambda("i", Expr::this())),
                ]),
                Expr::app(Expr::var("checkEmpty"), Expr::var("extendedSet")),
            ),
        )
    });

    // Example 17: SAME checkEmpty function applied to BOTH sets
    // This demonstrates that one function works with structurally compatible objects
    example("Same function, both sets", || {
        // let checkEmpty = λs. s.isEmpty in
        // let simple = { isEmpty = true, contains = λi. false } in
        // let extended = { isEmpty = false, contains = λi. true, insert = λi. this, union = λs. this } in
        // if checkEmpty(simple) then checkEmpty(extended) else false
        Expr::let_(
            "checkEmpty",
            Expr::lambda("s", Expr::field(Expr::var("s"), "isEmpty")),
            Expr::let_(
                "simple",
                Expr::object(vec![
                    ("isEmpty", Expr::bool(true)),
                    ("contains", Expr::lambda("i", Expr::bool(false))),
                ]),
                Expr::let_(
                    "extended",
                    Expr::object(vec![
                        ("isEmpty", Expr::bool(false)),
                        ("contains", Expr::lambda("i", Expr::bool(true))),
                        ("insert", Expr::lambda("i", Expr::this())),
                        ("union", Expr::lambda("s", Expr::this())),
                    ]),
                    // if checkEmpty(simple) then checkEmpty(extended) else false
                    Expr::if_(
                        Expr::app(Expr::var("checkEmpty"), Expr::var("simple")),
                        Expr::app(Expr::var("checkEmpty"), Expr::var("extended")),
                        Expr::bool(false),
                    ),
                ),
            ),
        )
    });

    // Example 18: A function that calls a method on any object with 'contains'
    example("Polymorphic contains caller", || {
        // λset. λx. set.contains(x)
        // Type: { contains: α → β | ρ } → α → β
        Expr::lambda(
            "set",
            Expr::lambda(
                "x",
                Expr::app(
                    Expr::field(Expr::var("set"), "contains"),
                    Expr::var("x"),
                ),
            ),
        )
    });

    // Example 19: Using the polymorphic contains caller with both sets
    example("contains caller with simple set", || {
        Expr::let_(
            "callContains",
            Expr::lambda(
                "set",
                Expr::lambda(
                    "x",
                    Expr::app(
                        Expr::field(Expr::var("set"), "contains"),
                        Expr::var("x"),
                    ),
                ),
            ),
            Expr::let_(
                "mySet",
                Expr::object(vec![
                    ("isEmpty", Expr::bool(true)),
                    ("contains", Expr::lambda("i", Expr::bool(false))),
                ]),
                Expr::app(
                    Expr::app(Expr::var("callContains"), Expr::var("mySet")),
                    Expr::int(42),
                ),
            ),
        )
    });

    // =====================================================
    // NEGATIVE TESTS - These should FAIL to type check
    // =====================================================
    println!("\n=== Negative Tests (should fail) ===\n");

    // Example 20: SHOULD FAIL - checkEmpty on object WITHOUT isEmpty
    example("SHOULD FAIL: checkEmpty on object without isEmpty", || {
        // let checkEmpty = λs. s.isEmpty in
        // let badObj = { value = 42 } in   <-- no isEmpty field!
        // checkEmpty(badObj)
        Expr::let_(
            "checkEmpty",
            Expr::lambda("s", Expr::field(Expr::var("s"), "isEmpty")),
            Expr::let_(
                "badObj",
                Expr::object(vec![("value", Expr::int(42))]), // Missing isEmpty!
                Expr::app(Expr::var("checkEmpty"), Expr::var("badObj")),
            ),
        )
    });

    // Example 21: SHOULD FAIL - contains caller on object without contains
    example("SHOULD FAIL: contains caller on object without contains", || {
        Expr::let_(
            "callContains",
            Expr::lambda(
                "set",
                Expr::lambda(
                    "x",
                    Expr::app(
                        Expr::field(Expr::var("set"), "contains"),
                        Expr::var("x"),
                    ),
                ),
            ),
            Expr::let_(
                "notASet",
                Expr::object(vec![("name", Expr::bool(true))]), // No contains!
                Expr::app(
                    Expr::app(Expr::var("callContains"), Expr::var("notASet")),
                    Expr::int(42),
                ),
            ),
        )
    });

    // Example 22: SHOULD FAIL - wrong type for isEmpty (int instead of using it as bool)
    example("SHOULD FAIL: isEmpty used as function but is bool", || {
        // let useIsEmpty = λs. s.isEmpty(42) in  <-- treating isEmpty as a function
        // let set = { isEmpty = true } in        <-- but isEmpty is a bool
        // useIsEmpty(set)
        Expr::let_(
            "useIsEmpty",
            Expr::lambda(
                "s",
                Expr::app(Expr::field(Expr::var("s"), "isEmpty"), Expr::int(42)),
            ),
            Expr::let_(
                "set",
                Expr::object(vec![("isEmpty", Expr::bool(true))]),
                Expr::app(Expr::var("useIsEmpty"), Expr::var("set")),
            ),
        )
    });

    println!("\n=== All examples complete ===");
}

fn example<F>(name: &str, make_expr: F)
where
    F: FnOnce() -> Expr,
{
    println!("--- {} ---", name);

    let expr = make_expr();
    let mut store = NodeStore::new();

    match infer_expr(&expr, &mut store) {
        Ok(ty) => {
            let type_str = display_type(&store, ty);
            println!("Type: {}", type_str);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    println!();
}
