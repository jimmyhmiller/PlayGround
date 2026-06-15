//! coil — a low-level Lisp with first-class MLIR.
//!
//! Design docs live in `../mlir-lisp-design/` (DESIGN, SPEC, KERNEL,
//! ELABORATION, prelude). This crate is the implementation, starting from the
//! reader and growing toward the single-pass elaborator.

pub mod backend;
pub mod printer;
pub mod reader;
pub mod value;

pub use printer::print;
pub use reader::{read_all, read_one, ReadError};
pub use value::Val;
