// Library interface for quick-clojure-poc
//
// This allows tests and external code to use our compiler as a library

pub mod arm_codegen;
pub mod arm_instructions;
pub mod builtins;
pub mod clojure_ast;
pub mod compiler;
pub mod gc;
pub mod gc_runtime;
pub mod ir;
pub mod reader;
pub mod register_allocation;
pub mod trampoline;
pub mod value;
