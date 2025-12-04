pub mod schema;
pub mod ir_impl;
pub mod convert;

pub use schema::{UniversalIR, UniversalBlock, UniversalInstruction, UNIVERSAL_VERSION};
pub use ir_impl::UniversalCompilerIR;
pub use convert::{ion_to_universal, pass_to_universal, llvm_to_universal};
