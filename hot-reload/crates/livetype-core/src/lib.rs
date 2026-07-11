//! Live & Typed runtime core — the LLVM-free half of the system: the IR, the
//! CFG verifier, the reference interpreter, and the thread-safe (`mt`) tier.
//!
//! The JIT lives in the parent `livetype` crate, which depends on this one. The
//! split keeps this crate free of the inkwell/LLVM static library so its
//! concurrency can be verified under ThreadSanitizer.

mod model;
mod mt;
mod runtime;
mod verify;

pub use model::*;
pub use mt::*;
pub use runtime::*;
pub use verify::*;
