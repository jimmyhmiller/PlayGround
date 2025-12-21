//! Integration tests that verify all example files compile to valid MLIR IR

use lispier::{DialectRegistry, IRGenerator, Parser, Reader, Tokenizer};
use std::fs;
use std::path::Path;

fn compile_example(path: &Path) -> Result<String, String> {
    let source = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut tokenizer = Tokenizer::new(&source);
    let tokens = tokenizer
        .tokenize()
        .map_err(|e| format!("Tokenizer error: {:?}", e))?;

    let mut reader = Reader::new(&tokens);
    let values = reader
        .read()
        .map_err(|e| format!("Reader error: {:?}", e))?;

    let mut parser = Parser::new();
    let nodes = parser
        .parse(&values)
        .map_err(|e| format!("Parser error: {:?}", e))?;

    let registry = DialectRegistry::new();
    let generator = IRGenerator::new(&registry);
    let module = generator
        .generate(&nodes)
        .map_err(|e| format!("IR generation error: {:?}", e))?;

    // Verify the module is valid
    if !generator.verify(&module) {
        return Err("Module verification failed".to_string());
    }

    Ok(generator.print_module_to_string(&module))
}

macro_rules! example_test {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let path = Path::new("examples").join($file);
            match compile_example(&path) {
                Ok(ir) => {
                    assert!(
                        ir.contains("module") || ir.contains("builtin.module"),
                        "Expected module in IR output"
                    );
                }
                Err(e) => panic!("Failed to compile {}: {}", $file, e),
            }
        }
    };
}

// Basic operations
example_test!(test_example_simple, "simple.lisp");
example_test!(test_example_addi_chain, "addi_chain.lisp");

// Function definitions
example_test!(test_example_add, "add.lisp");
example_test!(test_example_multiply, "multiply.lisp");
example_test!(test_example_subtract, "subtract.lisp");
example_test!(test_example_float_add, "float_add.lisp");
example_test!(test_example_f64_precision, "f64_precision.lisp");
example_test!(test_example_i64_large, "i64_large.lisp");
example_test!(test_example_nested_ops, "nested_ops.lisp");
example_test!(test_example_type_inference, "type_inference.lisp");
example_test!(test_example_variables, "variables.lisp");

// Namespace aliases
example_test!(test_example_arithmetic, "arithmetic.lsp");

// Control flow
example_test!(test_example_control_flow, "control_flow.lsp");

// Structured control flow (SCF)
example_test!(test_example_scf_loops, "scf_loops.lsp");

// Memory operations
example_test!(test_example_memory, "memory.lsp");

// Simple entry point
example_test!(test_example_hello, "hello.lsp");

#[test]
fn test_all_examples_exist() {
    let expected_files = [
        "simple.lisp",
        "addi_chain.lisp",
        "add.lisp",
        "multiply.lisp",
        "subtract.lisp",
        "float_add.lisp",
        "f64_precision.lisp",
        "i64_large.lisp",
        "nested_ops.lisp",
        "type_inference.lisp",
        "variables.lisp",
        "arithmetic.lsp",
        "control_flow.lsp",
        "scf_loops.lsp",
        "memory.lsp",
        "hello.lsp",
    ];

    for file in &expected_files {
        let path = Path::new("examples").join(file);
        assert!(path.exists(), "Example file {} does not exist", file);
    }
}
