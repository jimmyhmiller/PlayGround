//! gc-rust runtime crate.
//!
//! This is the GC collector ([`gc`]) plus the LLVM↔collector ABI boundary
//! ([`runtime`]) — and nothing else. It is split out from the main `gcrust`
//! crate (which pulls in LLVM/inkwell for codegen) so that AOT-compiled
//! standalone binaries can statically link *only* the runtime: building this
//! crate as a `staticlib` yields a `libgcrust_rt.a` that contains the
//! `ai_gc_*` / `ai_print_*` / `gcr_runtime_main` symbols without dragging in
//! the multi-hundred-megabyte LLVM toolchain.
//!
//! The main `gcrust` crate re-exports `gcrust_rt::gc` and `gcrust_rt::runtime`
//! so existing `crate::gc` / `crate::runtime` paths keep working.

pub mod gc;
pub mod runtime;
