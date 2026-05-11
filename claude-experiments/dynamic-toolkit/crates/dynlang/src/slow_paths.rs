//! Default slow-path thunks for dynamic operations.
//!
//! When a `dyn_*` operation hits its slow path (operands aren't both
//! NanBox-encoded numbers), it calls into the embedder-supplied extern.
//! For frontends that don't actually exercise the slow path — benchmarks,
//! arithmetic-only subsets, early bring-up — these panic stubs are the
//! sensible default: clear failure if the slow path is hit, no
//! per-language boilerplate to write.
//!
//! Bound automatically when the embedder calls
//! [`DynModule::register_slow_paths_with_defaults`](crate::DynModule::register_slow_paths_with_defaults).
//! Override individual operations via
//! [`DynModule::override_extern`](crate::DynModule::override_extern).

macro_rules! panic_stub_2 {
    ($name:ident, $op:literal) => {
        pub extern "C" fn $name(a: u64, b: u64) -> u64 {
            panic!(
                "dynlang slow-path `{}` hit: a=0x{:x} b=0x{:x} \
                 — embedder did not override this default panic stub",
                $op, a, b,
            );
        }
    };
}

macro_rules! panic_stub_1 {
    ($name:ident, $op:literal) => {
        pub extern "C" fn $name(v: u64) -> u64 {
            panic!(
                "dynlang slow-path `{}` hit: v=0x{:x} \
                 — embedder did not override this default panic stub",
                $op, v,
            );
        }
    };
}

panic_stub_2!(panic_add, "add");
panic_stub_2!(panic_sub, "sub");
panic_stub_2!(panic_mul, "mul");
panic_stub_2!(panic_div, "div");
panic_stub_2!(panic_eq, "eq");
panic_stub_2!(panic_lt, "lt");
panic_stub_2!(panic_gt, "gt");
panic_stub_1!(panic_neg, "neg");
panic_stub_1!(panic_not, "not");

#[cfg(test)]
mod tests {
    use crate::{DynModule, GcConfig, NanBoxTags};

    #[test]
    fn register_with_defaults_populates_auto_externs() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dyn_module.register_slow_paths_with_defaults("rt");

        for op in [
            "rt_add", "rt_sub", "rt_mul", "rt_div", "rt_eq", "rt_lt", "rt_gt", "rt_neg", "rt_not",
        ] {
            assert!(
                dyn_module.auto_externs.contains_key(op),
                "auto_externs missing `{op}`"
            );
        }
        assert_eq!(dyn_module.auto_externs.len(), 9);
    }

    #[test]
    fn override_extern_replaces_entry() {
        extern "C" fn my_add(_: u64, _: u64) -> u64 {
            0xDEADBEEF
        }

        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dyn_module.register_slow_paths_with_defaults("rt");
        let default_add = dyn_module.auto_externs["rt_add"];

        dyn_module.override_extern("rt_add", my_add as *const u8);
        let new_add = dyn_module.auto_externs["rt_add"];

        assert_ne!(default_add, new_add);
        assert_eq!(new_add, my_add as *const u8);
        // Other entries unchanged.
        assert_ne!(dyn_module.auto_externs["rt_sub"], my_add as *const u8);
    }

    // Note: panic_add / panic_neg etc. are `extern "C" fn` and abort on
    // panic rather than unwinding, so they can't be tested with
    // `should_panic`. The bodies are five-line macros; visual review is
    // sufficient. Their registration in `auto_externs` is verified above.

    #[test]
    fn register_without_defaults_leaves_auto_externs_empty() {
        let mut dyn_module = DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        dyn_module.register_slow_paths("rt"); // declare-only form
        assert!(dyn_module.auto_externs.is_empty());
    }
}
