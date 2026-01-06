//! Optimization passes.
//!
//! This module contains the implementation of various optimization passes:
//!
//! - **DCE** - Dead Code Elimination
//! - **Copy Propagation** - Replace copies with original values
//! - **Constant Folding** - Evaluate constant expressions at compile time
//! - **Constant Propagation** - Replace variables with known constant values
//! - **CSE** - Common Subexpression Elimination

pub mod dce;
pub mod copy_prop;
pub mod const_fold;
pub mod const_prop;
pub mod cse;

pub use dce::DeadCodeElimination;
pub use copy_prop::CopyPropagation;
pub use const_fold::ConstantFolding;
pub use const_prop::ConstantPropagation;
pub use cse::CommonSubexpressionElimination;
