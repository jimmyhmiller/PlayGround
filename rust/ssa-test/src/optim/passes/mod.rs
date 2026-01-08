//! Optimization passes.
//!
//! This module contains the implementation of various optimization passes:
//!
//! - **DCE** - Dead Code Elimination
//! - **Copy Propagation** - Replace copies with original values
//! - **Constant Folding** - Evaluate constant expressions at compile time
//! - **Constant Propagation** - Replace variables with known constant values
//! - **CSE** - Common Subexpression Elimination
//! - **Control Flow Simplification** - Simplify branches with constant conditions
//! - **Jump Threading** - Eliminate blocks that only contain unconditional jumps
//! - **CFG Cleanup** - Remove unreachable blocks and fix predecessor edges

pub mod dce;
pub mod copy_prop;
pub mod const_fold;
pub mod const_prop;
pub mod cse;
pub mod control_flow;
pub mod jump_threading;
pub mod cfg_cleanup;

pub use dce::DeadCodeElimination;
pub use copy_prop::CopyPropagation;
pub use const_fold::ConstantFolding;
pub use const_prop::ConstantPropagation;
pub use cse::CommonSubexpressionElimination;
pub use control_flow::ControlFlowSimplificationPass;
pub use jump_threading::JumpThreading;
pub use cfg_cleanup::CfgCleanup;
