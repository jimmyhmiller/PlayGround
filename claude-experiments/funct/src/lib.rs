//! **funct** — a small functional embeddable language (see `docs/funct-spec.md`).
//!
//! Highlights:
//! - stack-based bytecode VM with fully reified state: pause between any two
//!   instructions, inspect, snapshot (Clone), serialize to disk, resume
//! - `|>` pipes + UFCS, real pattern matching, `Result`/`Option` + `?`
//! - atoms as the only escaping mutable state (capture/restore them)
//! - hot reload by name-keyed function table
//! - frictionless Rust interop (`register1`, `register_type`, `call_typed`)

pub mod ast;
pub mod bytecode;
pub mod compiler;
pub mod interop;
pub mod json;
pub mod lexer;
pub mod parser;
pub mod prelude;
pub mod snapshot;
pub mod testing;
pub mod value;
pub mod vm;

pub use interop::{FromValue, ToValue};
pub use value::{Value, VariantPayload};
pub use vm::{Cause, Fault, Funct, FunctError, RunResult, Status, StepResult, StopWhen, VmState};
