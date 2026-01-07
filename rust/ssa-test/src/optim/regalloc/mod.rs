//! Linear Scan Register Allocator
//!
//! This module implements a linear scan register allocator based on the
//! Poletto & Sarkar algorithm. It integrates with the SSA framework by
//! using liveness analysis and operating on SSA form code.
//!
//! # Architecture
//!
//! The allocator is designed with a trait-based architecture, allowing
//! users to supply their own target architecture description:
//!
//! - [`PhysicalRegister`] - Describes physical registers
//! - [`RegisterClass`] - Groups registers by capability
//! - [`TargetArchitecture`] - Describes the complete target
//!
//! # Usage
//!
//! ```ignore
//! use ssa_lib::optim::regalloc::*;
//!
//! // 1. Eliminate phi nodes (convert to copies)
//! PhiElimination::eliminate(&mut translator);
//!
//! // 2. Compute live intervals
//! let liveness = LivenessAnalysis::compute(&translator);
//! let intervals = IntervalAnalysis::compute(&translator, &liveness);
//!
//! // 3. Run register allocation
//! let mut allocator = LinearScanAllocator::new(target);
//! let result = allocator.allocate(&mut intervals);
//!
//! // 4. Insert spill code
//! insert_spill_code(&mut translator, &result);
//! ```
//!
//! # Register Constraints
//!
//! Some instructions require operands in specific registers (e.g., division
//! on x86 requires RAX/RDX). Use [`HasRegisterConstraints`] to describe these:
//!
//! ```ignore
//! impl HasRegisterConstraints for MyInstruction {
//!     type Register = X86Reg;
//!
//!     fn register_constraints(&self) -> Option<OperandConstraints<X86Reg>> {
//!         match self {
//!             MyInstruction::Div { .. } => Some(
//!                 OperandConstraints::new()
//!                     .with_dest(RegisterConstraint::Fixed(X86Reg::RAX))
//!             ),
//!             _ => None,
//!         }
//!     }
//! }
//! ```

pub mod target;
pub mod constraints;
pub mod interval;
pub mod phi_elim;
pub mod linear_scan;
pub mod spill;
pub mod lowered;

// Re-export main types
pub use target::{PhysicalRegister, RegisterClass, TargetArchitecture};
pub use constraints::{RegisterConstraint, OperandConstraints, HasRegisterConstraints};
pub use interval::{ProgramPoint, LiveRange, LiveInterval, Location, IntervalAnalysis};
pub use phi_elim::{PhiElimination, PhiEliminationViolation, assert_valid_after_phi_elimination};
pub use linear_scan::{LinearScanAllocator, LinearScanConfig, AllocationResult, AllocationStats};
pub use spill::SpillCodeFactory;
pub use lowered::{LoweredOperand, LoweredInstruction, LoweredBlock, LoweredFunction};
