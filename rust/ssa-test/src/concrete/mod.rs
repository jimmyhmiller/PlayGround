//! Concrete implementation of the SSA library types.
//!
//! This module provides a reference implementation showing how to use the
//! generic SSA library with concrete value and instruction types.

pub mod value;
pub mod instruction;
pub mod ast;
pub mod syntax;

pub use value::Value;
pub use instruction::Instruction;
pub use ast::{Ast, BinaryOperator, UnaryOperator};
