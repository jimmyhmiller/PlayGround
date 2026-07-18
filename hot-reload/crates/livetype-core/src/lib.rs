//! Live & Typed runtime core — the LLVM-free half of the system: the IR, the
//! CFG verifier, the one shared runtime ([`Shared`]), and the one executor
//! ([`Engine`] — a tiered actor loop that runs interpreted frames itself and
//! native frames through a pluggable [`TierSource`]).
//!
//! The compiler behind [`TierSource`] lives in the parent `livetype` crate,
//! which depends on this one. The split keeps this crate free of the
//! inkwell/LLVM static library so its concurrency can be verified under Miri —
//! a [`NoJit`] engine exercises the identical loop with no native code.

mod engine;
mod exec;
mod frontend;
mod heap;
mod model;
mod mt;
mod native;
mod runtime;
mod verify;

pub use engine::*;
pub use exec::*;
pub use frontend::*;
pub use heap::*;
pub use model::*;
pub use mt::*;
pub use native::*;
pub use runtime::*;
pub use verify::*;
