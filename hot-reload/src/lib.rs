//! Live & Typed runtime prototype.
//!
//! Execution state lives in explicit, heap-resident frames. The interpreter is
//! the reference executor for an IR whose instructions can later be lowered to
//! LLVM `step(frame, runtime) -> StepResult` functions without changing pause
//! and resume semantics.

mod jit;
mod model;
mod mt;
mod runtime;
mod verify;

pub use jit::*;
pub use model::*;
pub use mt::*;
pub use runtime::*;
pub use verify::*;
