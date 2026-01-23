pub mod emit;
pub mod env;
pub mod eval;
pub mod passes;
pub mod residual;

use oxc_allocator::Allocator;
use oxc_ast::ast::Program;
use oxc_codegen::Codegen;
use oxc_parser::Parser;
use oxc_span::SourceType;

pub fn parse<'a>(allocator: &'a Allocator, source: &'a str) -> Program<'a> {
    let source_type = SourceType::mjs();
    let parser = Parser::new(allocator, source, source_type);
    let result = parser.parse();

    if !result.errors.is_empty() {
        for error in &result.errors {
            eprintln!("Parse error: {:?}", error);
        }
        panic!("Parse failed");
    }

    result.program
}

pub fn emit(program: &Program<'_>) -> String {
    Codegen::new().build(program).code
}

pub fn roundtrip(source: &str) -> String {
    let allocator = Allocator::default();
    let program = parse(&allocator, source);
    emit(&program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[derive(Debug, PartialEq)]
    struct JsResult {
        stdout: String,
        stderr: String,
        success: bool,
    }

    fn run_js(code: &str) -> JsResult {
        let output = Command::new("node")
            .arg("-e")
            .arg(code)
            .output()
            .expect("Failed to execute node");

        JsResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            success: output.status.success(),
        }
    }

    fn assert_semantic_equivalence(source: &str) {
        let original_result = run_js(source);
        let transformed = roundtrip(source);
        let transformed_result = run_js(&transformed);

        assert!(
            original_result.success,
            "Original code failed to run:\nCode:\n{}\nStderr: {}",
            source,
            original_result.stderr
        );

        assert!(
            transformed_result.success,
            "Transformed code failed to run:\nOriginal:\n{}\nTransformed:\n{}\nStderr: {}",
            source,
            transformed,
            transformed_result.stderr
        );

        assert_eq!(
            original_result.stdout, transformed_result.stdout,
            "Output mismatch!\nOriginal code:\n{}\nTransformed code:\n{}\nOriginal stdout: {:?}\nTransformed stdout: {:?}",
            source,
            transformed,
            original_result.stdout,
            transformed_result.stdout
        );
    }

    #[test]
    fn test_simple_assignment() {
        assert_semantic_equivalence("let x = 5; console.log(x);");
    }

    #[test]
    fn test_arithmetic() {
        assert_semantic_equivalence("console.log(2 + 3 * 4);");
    }

    #[test]
    fn test_function_call() {
        assert_semantic_equivalence(r#"
            function add(a, b) { return a + b; }
            console.log(add(2, 3));
        "#);
    }

    #[test]
    fn test_array_operations() {
        assert_semantic_equivalence(r#"
            let arr = [1, 2, 3];
            arr.push(4);
            console.log(arr.length);
            console.log(arr[0]);
        "#);
    }

    #[test]
    fn test_while_loop() {
        assert_semantic_equivalence(r#"
            let x = 0;
            while (x < 5) {
                console.log(x);
                x = x + 1;
            }
        "#);
    }

    #[test]
    fn test_switch_statement() {
        assert_semantic_equivalence(r#"
            function test(op) {
                switch (op) {
                    case 0:
                        return "zero";
                    case 1:
                        return "one";
                    default:
                        return "other";
                }
            }
            console.log(test(0));
            console.log(test(1));
            console.log(test(99));
        "#);
    }

    #[test]
    fn test_vm_example() {
        assert_semantic_equivalence(r#"
            function run(program) {
                let pc = 0;
                let stack = [];
                while (pc < program.length) {
                    switch (program[pc]) {
                        case 0:
                            stack.push(program[pc + 1]);
                            pc += 2;
                            break;
                        case 1:
                            let a = stack.pop();
                            let b = stack.pop();
                            stack.push(a + b);
                            pc += 1;
                            break;
                    }
                }
                return stack;
            }
            console.log(run([0, 5, 0, 3, 1]));
        "#);
    }

    #[test]
    fn test_no_output_is_fine() {
        // Programs with no console.log should still work
        assert_semantic_equivalence("let x = 5;");
        assert_semantic_equivalence("function foo() { return 42; }");
    }

    #[test]
    fn test_string_operations() {
        assert_semantic_equivalence(r#"
            let s = "hello";
            console.log(s.length);
            console.log(s + " world");
        "#);
    }

    #[test]
    fn test_conditionals() {
        assert_semantic_equivalence(r#"
            function check(x) {
                if (x > 10) {
                    return "big";
                } else if (x > 5) {
                    return "medium";
                } else {
                    return "small";
                }
            }
            console.log(check(15));
            console.log(check(7));
            console.log(check(2));
        "#);
    }

    #[test]
    fn test_nested_functions() {
        assert_semantic_equivalence(r#"
            function outer(x) {
                function inner(y) {
                    return y * 2;
                }
                return inner(x) + 1;
            }
            console.log(outer(5));
        "#);
    }
}
