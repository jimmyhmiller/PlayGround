//! The MLIR boundary.
//!
//! Per `mlir-lisp-design/KERNEL.md §4`, the language talks to MLIR through a
//! small, fixed primitive catalog. We capture that catalog as a `Backend`
//! trait so the core (reader/evaluator/elaboration) depends only on this
//! interface — never on `melior` directly. Two implementations are planned:
//!
//!   * `NullBackend` (here): no MLIR; parsing/printing only. Lets the whole
//!     core build and be tested without LLVM/MLIR installed.
//!   * `MeliorBackend` (future, behind the `mlir` feature): the real thing.
//!
//! Opaque handles (`TypeRef`, `ValueRef`, …) are indices the backend owns, so
//! `Val` can carry reflected-MLIR cases without the core knowing their layout.

#![allow(dead_code)]

/// Opaque handle to a backend-owned object (Type, Value, Op, …). The core
/// passes these around without interpreting them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Handle(pub u64);

/// Errors a backend primitive can raise (parse failures, verify failures, …).
#[derive(Debug, Clone, PartialEq)]
pub struct BackendError(pub String);

/// The full primitive surface between coil and MLIR. Kept intentionally small:
/// if a feature can be a prelude macro, it is not here. This trait mirrors
/// `KERNEL.md §4` and will grow op/region/pass/jit methods as they are wired.
pub trait Backend {
    /// Parse a `!…` type literal (without the `!`).
    fn parse_type(&mut self, text: &str) -> Result<Handle, BackendError>;
    /// Parse a `#…` attribute literal (without the `#`).
    fn parse_attr(&mut self, text: &str) -> Result<Handle, BackendError>;
    /// `iN` integer type.
    fn integer_type(&mut self, width: u32, signed: bool) -> Result<Handle, BackendError>;
    /// The type of an SSA value (pure; ELABORATION.md §3).
    fn value_type(&mut self, value: Handle) -> Result<Handle, BackendError>;

    // The remaining catalog — op construction, infer_results, regions/blocks,
    // module, verify, passes, jit, build/with-scratch — lands as elaboration
    // is implemented. See KERNEL.md §4 for the target signatures.
}

/// A backend that refuses every MLIR primitive. Used to build & test the core
/// (reader, printer, evaluator of non-MLIR forms) without MLIR present.
#[derive(Default)]
pub struct NullBackend;

impl NullBackend {
    const MSG: &'static str = "the MLIR backend is not available (build with --features mlir)";
}

impl Backend for NullBackend {
    fn parse_type(&mut self, _text: &str) -> Result<Handle, BackendError> {
        Err(BackendError(Self::MSG.into()))
    }
    fn parse_attr(&mut self, _text: &str) -> Result<Handle, BackendError> {
        Err(BackendError(Self::MSG.into()))
    }
    fn integer_type(&mut self, _width: u32, _signed: bool) -> Result<Handle, BackendError> {
        Err(BackendError(Self::MSG.into()))
    }
    fn value_type(&mut self, _value: Handle) -> Result<Handle, BackendError> {
        Err(BackendError(Self::MSG.into()))
    }
}
