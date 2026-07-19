//! Live & Typed runtime prototype.
//!
//! The LLVM-free core — IR, verifier, interpreter, and the thread-safe tier —
//! lives in the [`livetype_core`] crate and is re-exported here. This crate adds
//! the LLVM `step` backend (`jit`), so `livetype` is the full system and
//! `livetype-core` is the runtime you can build (and ThreadSanitize) without
//! LLVM.

mod jit;

pub use jit::*;
pub use livetype_core::*;
