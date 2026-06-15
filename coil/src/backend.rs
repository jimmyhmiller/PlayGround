//! The MLIR boundary (AOT codegen target).
//!
//! Per `mlir-lisp-design/AOT.md`, `emit` walks core forms and drives MLIR
//! through this `Backend` trait to *build* a module (ordinary AOT codegen, not
//! interpretation). The core depends only on this interface, never on `melior`,
//! so it builds and is tested without MLIR installed.
//!
//! Planned implementations:
//!   * `RecordingBackend` (`recording.rs`): logs every call, hands out fake
//!     handles — lets us test the lisp→MLIR mapping with no MLIR present.
//!   * `NullBackend` (here): refuses every primitive (used by the CLI when
//!     built without `--features mlir`).
//!   * `MeliorBackend` (future, `--features mlir`): the real thing.

#![allow(dead_code)]

/// Opaque handle to a backend-owned object (Type, Value, Op, Block, Region,
/// Module). The core passes these around without interpreting them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Handle(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How an operation's result types are determined (ELABORATION.md §3).
#[derive(Debug, Clone, PartialEq)]
pub enum ResultTypes {
    /// Let MLIR's `InferTypeOpInterface` decide.
    Infer,
    /// Exactly these result types.
    Explicit(Vec<Handle>),
}

/// An attribute on an operation. Values are kept as rendered text in this
/// increment (enough to verify the mapping); the real backend parses them.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedAttr {
    pub name: String,
    pub value: String,
}

/// The codegen surface between coil and MLIR. Small by construction: anything
/// expressible as a prelude macro is not here (KERNEL.md §4).
pub trait Backend {
    // --- types ---
    fn parse_type(&mut self, text: &str) -> Result<Handle, BackendError>;
    fn integer_type(&mut self, width: u32, signed: bool) -> Result<Handle, BackendError>;

    // --- module / regions / blocks ---
    fn create_module(&mut self) -> Result<Handle, BackendError>;
    /// The module's top-level block, where top-level ops are inserted.
    fn module_body(&mut self, module: Handle) -> Result<Handle, BackendError>;
    fn create_region(&mut self) -> Result<Handle, BackendError>;
    /// Append a block (with the given argument types) to a region.
    fn create_block(&mut self, region: Handle, arg_types: &[Handle]) -> Result<Handle, BackendError>;
    /// The i-th block argument as an SSA value.
    fn block_arg(&mut self, block: Handle, i: usize) -> Result<Handle, BackendError>;
    /// Set the insertion point to the end of `block`.
    fn set_insertion_end(&mut self, block: Handle) -> Result<(), BackendError>;

    // --- operations ---
    /// Build an operation at the current insertion point. Returns its result
    /// SSA values (possibly empty).
    fn build_op(
        &mut self,
        name: &str,
        operands: &[Handle],
        results: ResultTypes,
        attrs: &[NamedAttr],
        regions: &[Handle],
        successors: &[Handle],
    ) -> Result<Vec<Handle>, BackendError>;
}

/// A backend that refuses every MLIR primitive. Lets the CLI link and run the
/// reader without MLIR; `emit` against it fails fast with a clear message.
#[derive(Default)]
pub struct NullBackend;

impl NullBackend {
    const MSG: &'static str = "the MLIR backend is not available (build with --features mlir)";
    fn err<T>() -> Result<T, BackendError> {
        Err(BackendError(Self::MSG.into()))
    }
}

impl Backend for NullBackend {
    fn parse_type(&mut self, _t: &str) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn integer_type(&mut self, _w: u32, _s: bool) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn create_module(&mut self) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn module_body(&mut self, _m: Handle) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn create_region(&mut self) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn create_block(&mut self, _r: Handle, _a: &[Handle]) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn block_arg(&mut self, _b: Handle, _i: usize) -> Result<Handle, BackendError> {
        Self::err()
    }
    fn set_insertion_end(&mut self, _b: Handle) -> Result<(), BackendError> {
        Self::err()
    }
    fn build_op(
        &mut self,
        _n: &str,
        _o: &[Handle],
        _r: ResultTypes,
        _a: &[NamedAttr],
        _rg: &[Handle],
        _s: &[Handle],
    ) -> Result<Vec<Handle>, BackendError> {
        Self::err()
    }
}
