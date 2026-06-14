//! gc-rust: a fast, monomorphized, GC'd systems language.
//!
//! The garbage collector ([`gc`]) is the one piece lifted wholesale from the
//! ai-lang project — a self-contained, precise, semi-space copying collector
//! with a shadow-stack root system and a safepoint protocol designed to
//! integrate with LLVM-compiled code. Everything else in this crate (the
//! language, frontend, type system, monomorphization, codegen) is new.

pub mod gc;
pub mod runtime;
