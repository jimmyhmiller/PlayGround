//! Simple linear IR for demonstrating CFG construction and SSA translation.
//!
//! This module provides two IR representations:
//!
//! - **Input IR** ([`input`]) - User-facing representation with string variable names
//! - **SSA IR** ([`ssa`]) - SSA form output with `SsaVariable` everywhere
//!
//! The [`translate_to_ssa`] function converts input IR to SSA form, with
//! automatic validation to ensure correctness.
//!
//! # Example
//!
//! ```ignore
//! use ssa_test::linear::input::{InputInstr, InputValue, Label, BinOp};
//! use ssa_test::linear::translate_to_ssa;
//!
//! let program = vec![
//!     InputInstr::Assign { dest: "x".into(), value: InputValue::Const(1) },
//!     InputInstr::Assign { dest: "x".into(), value: InputValue::Const(2) },
//!     InputInstr::Return(InputValue::Var("x".into())),
//! ];
//!
//! let result = translate_to_ssa(program).expect("SSA translation failed");
//! result.print();  // Shows: v0 := 1, v1 := 2, return v1
//! ```

pub mod input;
pub mod ssa;
mod translate;

// Re-export commonly used types
pub use input::{BinOp, InputInstr, InputValue, Label};
pub use ssa::{SsaInstr, SsaValue};
pub use translate::{translate_to_ssa, LinearTranslator, SsaResult, TranslationError};
