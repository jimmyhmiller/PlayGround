//! lambda-Tally: a dependent + linear (quantitative) language for memory-safe
//! low-level code.
//!
//! Pipeline:
//! ```text
//!   rust_surface  (lex → parse → elaborate)   the surface language
//!        │  emits a checked dep::Term
//!        ▼
//!      dep         (NbE kernel: eval/quote/conv + bidirectional QTT checker)
//!        │  erase types/indices/proofs (multiplicity 0)
//!        ▼
//!   dep_codegen   (LLVM lowering, behind the `llvm` feature)
//! ```
//! Permissions, regions, indices, and proofs are ERASED, so the backend lowers an
//! already-checked program with no runtime notion of them.

pub mod dep;
pub mod mult;
pub mod rust_surface;
pub(crate) mod totality;

#[cfg(feature = "llvm")]
pub mod dep_codegen;
