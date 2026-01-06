//! SSA Library - Generic Static Single Assignment Form Construction
//!
//! This library provides a generic implementation of SSA construction based on
//! the Braun et al. algorithm ("Simple and Efficient Construction of Static
//! Single Assignment Form", CC 2013).
//!
//! # Overview
//!
//! To use this library with your own IR, implement these traits on your types:
//!
//! - [`SsaValue`] - For your value/operand type
//! - [`SsaInstruction`] - For your instruction type
//! - [`InstructionFactory`] - For creating SSA-specific instructions
//!
//! # Example
//!
//! ```ignore
//! use ssa_lib::prelude::*;
//!
//! // Define your value type
//! #[derive(Clone, PartialEq, Eq, Hash, Debug)]
//! enum MyValue {
//!     Int(i64),
//!     Var(SsaVariable),
//!     Phi(PhiId),
//!     Undef,
//! }
//!
//! impl SsaValue for MyValue {
//!     fn from_phi(id: PhiId) -> Self { MyValue::Phi(id) }
//!     fn from_var(v: SsaVariable) -> Self { MyValue::Var(v) }
//!     fn undefined() -> Self { MyValue::Undef }
//!     fn as_phi(&self) -> Option<PhiId> {
//!         match self { MyValue::Phi(id) => Some(*id), _ => None }
//!     }
//!     fn as_var(&self) -> Option<&SsaVariable> {
//!         match self { MyValue::Var(v) => Some(v), _ => None }
//!     }
//! }
//!
//! // Use the SSA translator
//! let mut translator = SSATranslator::<MyValue, MyInstr, MyFactory>::new();
//! ```

// Core modules
pub mod types;
pub mod traits;
pub mod translator;
pub mod validation;
pub mod visualizer;
pub mod cfg;
pub mod cfg_validation;

// Optimization framework
pub mod optim;

// Concrete implementation example
pub mod concrete;

// Linear IR example (demonstrates CFG → SSA pipeline)
pub mod linear;

// Re-export core types
pub use types::{Block, BlockId, Phi, PhiId, PhiReference, SsaVariable};

// Re-export traits
pub use traits::{InstructionFactory, SsaInstruction, SsaValue};

// Re-export translator
pub use translator::SSATranslator;

// Re-export validation
pub use validation::{assert_valid_ssa, debug_ssa_state, validate_ssa, SSAViolation};

// Re-export visualizer
pub use visualizer::{FormatInstruction, FormatValue, SSAVisualizer};

// Re-export CFG
pub use cfg::{Cfg, CfgBlock, CfgBlockId, CfgBuilder, CfgInstruction, ControlFlow};

// Re-export CFG validation
pub use cfg_validation::{CfgViolation, validate_cfg, assert_valid_cfg, compute_reachable};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::types::{Block, BlockId, Phi, PhiId, PhiReference, SsaVariable};
    pub use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
    pub use crate::translator::SSATranslator;
    pub use crate::validation::{assert_valid_ssa, debug_ssa_state, validate_ssa, SSAViolation};
    pub use crate::visualizer::{FormatInstruction, FormatValue, SSAVisualizer};
    pub use crate::cfg::{Cfg, CfgBlock, CfgBlockId, CfgBuilder, CfgInstruction, ControlFlow};
    pub use crate::cfg_validation::{CfgViolation, validate_cfg, assert_valid_cfg, compute_reachable};
}
