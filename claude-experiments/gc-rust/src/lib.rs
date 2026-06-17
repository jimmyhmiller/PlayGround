//! gc-rust: a fast, monomorphized, GC'd systems language.
//!
//! The garbage collector ([`gc`]) is the one piece lifted wholesale from the
//! ai-lang project — a self-contained, precise, semi-space copying collector
//! with a shadow-stack root system and a safepoint protocol designed to
//! integrate with LLVM-compiled code. Everything else in this crate (the
//! language, frontend, type system, monomorphization, codegen) is new.

pub mod ast;
pub mod codegen;
pub mod compile;
pub mod core;
pub mod diag;
// `gc` and `runtime` live in the LLVM-free `gcrust-rt` crate so AOT binaries
// can statically link them without LLVM. Re-export under their original paths.
pub use gcrust_rt::gc;
pub mod layout;
pub mod lexer;
pub mod lower;
pub mod manifest;
pub mod parser;
pub mod resolve;
pub use gcrust_rt::runtime;
pub mod types;
