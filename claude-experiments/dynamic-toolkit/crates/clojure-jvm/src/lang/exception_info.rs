//! `clojure.lang.ExceptionInfo` — the exception type behind `ex-info` /
//! `ex-data`: a RuntimeException that carries a persistent map of
//! additional data (plus an optional cause).
//!
//! Original Java: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/ExceptionInfo.java`
//!
//! Unlike the Arc-backed host classes (Atom, Matcher, …), ExceptionInfo
//! instances are plain GC heap cells: the ObjType is registered in
//! `Compiler::new` (`clojure.lang.ExceptionInfo`) with four traced
//! `Value` fields, so the GC scans/forwards the held references like any
//! other heap object. The pieces live in:
//!
//!   * layout — this module's offset constants (single source of truth
//!     for the slot positions, shared by ctor and accessors)
//!   * construction — `runtime::clj_exception_info_{2,3}`, dispatched by
//!     `host_class::exception_info_ctor` for `(ExceptionInfo. msg map)` /
//!     `(ExceptionInfo. msg map cause)`
//!   * accessors — `runtime::cljvm_inst_getData` / `cljvm_inst_getMessage`
//!     / `cljvm_inst_getCause` / `cljvm_inst_getStackTrace` /
//!     `cljvm_inst_setStackTrace` instance-method externs
//!   * `instance?` — `host_class::is_exception_info`, the predicate for
//!     `clojure.lang.ExceptionInfo` and `clojure.lang.IExceptionInfo`
//!     (see `i_exception_info.rs`)
//!
//! Java's ctor contract is preserved: `data` must be non-nil
//! (`IllegalArgumentException("Additional data must be non-nil.")`).

/// Byte offset of the traced `message` slot (String or nil). The cell
/// header occupies the first 8 bytes; dynlang lays the declared `Value`
/// fields out in declaration order after it.
pub const MESSAGE_OFFSET: usize = 8;

/// Byte offset of the traced `data` slot (persistent map; non-nil per the
/// constructor contract).
pub const DATA_OFFSET: usize = 16;

/// Byte offset of the traced `cause` slot (nil, or the wrapped exception
/// for the 3-arg constructor).
pub const CAUSE_OFFSET: usize = 24;

/// Byte offset of the traced `stack_trace` slot — a Java-array-as-Vector
/// (the same analog `RT.toArray` uses). Initialized to a zero-length
/// vector at construction (this runtime records no JVM frames — the
/// `Throwable.getStackTrace` contract's zero-length-array case) and
/// replaced wholesale by `.setStackTrace`. Never nil.
pub const STACK_TRACE_OFFSET: usize = 32;
