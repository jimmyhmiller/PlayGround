//! clojure-jvm — direct 1-for-1 port of Clojure JVM (`clojure/lang/*.java`) to
//! Rust on top of the dynamic-toolkit.
//!
//! The intent is to mirror Clojure's Java source as literally as possible:
//! one Java file → one Rust module under [`lang`]. Where Java relies on JVM
//! features that don't translate (bytecode emission via ASM, reflection, host
//! class file output), the corresponding code is replaced by toolkit primitives
//! (`dynir`, `dynlang`, `dynruntime`, …). Where a piece isn't ported yet, it
//! panics with `unimplemented_port!(...)` carrying the Java class+method name.
//!
//! See `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/` for the
//! original Java source.

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(clippy::module_inception)]

/// Panic with a clear "not yet ported" message naming the Java class/method.
///
/// Required by global CLAUDE.md: stubs must throw hard errors, never silently
/// return defaults.
#[macro_export]
macro_rules! unimplemented_port {
    ($where:literal) => {
        panic!(
            "clojure-jvm: not yet ported from Clojure JVM: {} (at {}:{})",
            $where,
            file!(),
            line!()
        )
    };
    ($where:literal, $($arg:tt)*) => {
        panic!(
            "clojure-jvm: not yet ported from Clojure JVM: {} — {} (at {}:{})",
            $where,
            format_args!($($arg)*),
            file!(),
            line!()
        )
    };
}

pub mod lang;
// GC lock-in: the bare `Rooted::get()/set()` accessors are the form-430
// stale-pointer footgun. Denying their (deprecated) use in the runtime
// makes "hold a rooted/arg value as bare bits across an allocation" a
// compile error here. Reads must go through `get_raw(&Heap)`, which the
// borrow checker cannot let outlive an allocation.
#[deny(deprecated)]
pub mod runtime;
