pub mod schema;
pub mod ir_impl;

pub use schema::{LLVMModule, LLVMFunction, LLVMBasicBlock, LLVMInstruction};
pub use ir_impl::LLVMIR;
