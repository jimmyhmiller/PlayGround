// Library interface for quick-clojure-poc
//
// This allows tests and external code to use our compiler as a library

pub mod value;
pub mod reader;
pub mod clojure_ast;
pub mod ir;
pub mod compiler;
pub mod arm_codegen;
pub mod gc;
pub mod gc_runtime;
pub mod register_allocation;
pub mod trampoline;
