//! Runtime primitives the JIT calls back into.
//!
//! JIT-compiled Clojure code talks to the host through `extern "C"` functions
//! declared in [`lang::compiler::RuntimeExterns`] and bound into each
//! `JitModule` at compile time.
//!
//! Conventions:
//!   * All NanBox values cross the boundary as `u64`.
//!   * Var pointers cross as `u64` (the raw `*const Var`); the corresponding
//!     `Arc<Var>` lives in [`crate::lang::namespace::Namespace::mappings`]
//!     for the program's lifetime, so the pointer never dangles.

use std::sync::Arc;

use crate::lang::object::Object;
use crate::lang::var::Var;

// ── NanBox layout (matches dynlang's default NanBoxTags) ────────────────

const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const TAG_MASK: u64 = 0x0003_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

// Tags (matches `dynlang::NanBoxTags::default()` + our own fn-handle tag).
const TAG_NIL: u32 = 0;
const TAG_BOOL: u32 = 1;
const TAG_PTR: u32 = 2;
const TAG_FN: u32 = 3;

fn nanbox_encode(tag: u32, payload: u64) -> u64 {
    TAG_PATTERN | ((tag as u64) << 48) | (payload & PAYLOAD_MASK)
}

fn nanbox_tag(bits: u64) -> Option<u32> {
    if (bits & FULL_MASK) != TAG_PATTERN {
        return None;
    }
    Some(((bits & TAG_MASK) >> 48) as u32)
}

fn nanbox_payload(bits: u64) -> u64 { bits & PAYLOAD_MASK }

/// Decode a NanBox u64 into an `Object`. The inverse of `object_to_nanbox`.
pub fn nanbox_to_object(bits: u64) -> Object {
    match nanbox_tag(bits) {
        None => {
            // Untagged → IEEE 754 float (or natural NaN).
            Object::Double(f64::from_bits(bits))
        }
        Some(TAG_NIL) => Object::Nil,
        Some(TAG_BOOL) => Object::Bool(nanbox_payload(bits) != 0),
        Some(TAG_PTR) => Object::Unported {
            java_class: "pointer (NanBox tag=ptr) — not yet decoded",
        },
        Some(TAG_FN) => Object::Unported {
            java_class: "fn handle (NanBox tag=fn) — not yet decoded",
        },
        Some(_) => Object::Unported {
            java_class: "unknown NanBox tag",
        },
    }
}

/// Encode an `Object` into a NanBox u64. The inverse of `nanbox_to_object`.
pub fn object_to_nanbox(obj: &Object) -> u64 {
    match obj {
        Object::Nil => nanbox_encode(TAG_NIL, 0),
        Object::Bool(b) => nanbox_encode(TAG_BOOL, *b as u64),
        Object::Long(n) => (*n as f64).to_bits(),
        Object::Double(x) => x.to_bits(),
        _ => panic!(
            "clojure-jvm: object_to_nanbox: variant {obj:?} not yet representable as NanBox"
        ),
    }
}

// ── Var externs ────────────────────────────────────────────────────────

/// JIT extern: bind `val_bits` (NanBox) as the root of the Var at `var_ptr`.
/// Returns `val_bits` so the caller can use it as the expression value.
///
/// Safety: `var_ptr` must be a pointer obtained from `Arc::as_ptr` on a
/// `Var` that remains alive (the global namespace mapping holds it). The
/// caller (`DefExpr.emit`) bakes the pointer in as a compile-time constant
/// after looking up the Var via `Namespace::intern`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_bind_root(var_ptr: u64, val_bits: u64) -> u64 {
    let v: &Var = unsafe { &*(var_ptr as *const Var) };
    let obj = nanbox_to_object(val_bits);
    v.bind_root(obj);
    val_bits
}

/// JIT extern: return the current value of the Var at `var_ptr`, NanBox-encoded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_deref(var_ptr: u64) -> u64 {
    let v: &Var = unsafe { &*(var_ptr as *const Var) };
    let obj = v.deref();
    object_to_nanbox(&obj)
}

/// Resolve an `Arc<Var>` to a `u64` suitable for baking into IR. Holds onto
/// the Arc through the namespace mapping (the caller's responsibility), so
/// the pointer remains valid for the program's lifetime.
pub fn var_to_jit_ptr(v: &Arc<Var>) -> u64 {
    Arc::as_ptr(v) as u64
}
