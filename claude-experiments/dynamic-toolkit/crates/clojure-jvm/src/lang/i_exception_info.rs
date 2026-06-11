//! `clojure.lang.IExceptionInfo` — the interface exposing
//! `IPersistentMap getData()` on exceptions that carry a data map.
//!
//! Original Java: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/IExceptionInfo.java`
//!
//! We model the interface through the host-class registry rather than a
//! Rust trait:
//!
//!   * `(instance? clojure.lang.IExceptionInfo x)` — the registered
//!     `IExceptionInfo` entry in `host_class.rs` uses the
//!     `is_exception_info` predicate. `clojure.lang.ExceptionInfo` (see
//!     `exception_info.rs`) is the interface's only implementor, exactly
//!     as in upstream Clojure.
//!   * `(.getData x)` — dispatches to the `cljvm_inst_getData` extern in
//!     `runtime.rs`, which reads the ExceptionInfo cell's `data` slot and
//!     panics with a clear message on any other receiver.
//!
//! Like other interfaces, the registry entry has no constructor —
//! `(new clojure.lang.IExceptionInfo …)` panics.
