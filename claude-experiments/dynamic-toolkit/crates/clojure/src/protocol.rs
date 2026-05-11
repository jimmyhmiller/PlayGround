//! Bridge from Rust into Clojure's protocol-method dispatch. Lets the
//! printer, seq cursor, apply extern, and any other Rust-side walker
//! ask `(-seq xs)` / `(-first xs)` / `(-next xs)` etc. on an arbitrary
//! value without knowing what heap shape it is. Records, built-in
//! __ReaderList cells, and any future deftype that implements ISeq
//! all dispatch the same way.
//!
//! Mirrors what the macroexpander does to invoke a macro fn: look up
//! the impl Fn in `host.method_table`, read its `func_ref`, fetch the
//! code pointer through `host.jit`, and call it with the unified
//! single-list ABI `(self_fn, args_list)`.
//!
//! Lock-free w.r.t. `host.sym`: the lookup is keyed by sym-ids the
//! caller already holds (typically pre-interned on Host) and the JIT
//! call doesn't touch the symbol table. Safe to invoke while expand
//! or compile holds `host.sym.lock()`.

use dynir::ir::FuncRef;

use crate::host::with_host;
use crate::namespace as ns;
use crate::value::{self as v, NIL};

/// `(method-sym receiver)` — invoke the protocol method named by
/// `method_sym` on `receiver`.
///
/// **Returns NIL as a sentinel** when:
///   - `receiver` is not a heap value `value_type_name_sym` can name
///     (numbers, bools, nil), OR
///   - the receiver's type has no impl of `method_sym` registered.
///
/// This NIL-as-fallback contract is intentional: the printer and
/// `SeqCursor` use it to probe whether a value satisfies `ISeq`
/// without committing to the call. Callers that *require* the method
/// to exist must check `has_method` first, or use a strict path
/// (e.g. the JIT-emitted protocol wrapper, which panics via
/// `clj_method_lookup`).
pub fn invoke_method_0(method_sym: u32, receiver: u64) -> u64 {
    let type_sym = match value_type_name_sym(receiver) {
        Some(s) => s,
        None => return NIL,
    };
    let fn_obj = with_host(|h| {
        let table = h.method_table.lock().unwrap();
        let idx = match table.get(&(method_sym, type_sym)).copied() {
            Some(i) => i,
            None => return NIL,
        };
        h.method_roots.get(idx)
    });
    if !v::is_ptr(fn_obj) {
        return NIL;
    }
    // args_list = (receiver) — a single-element __ReaderList.
    let args_list = dynobj::roots::with_scope(3, |scope| {
        v::alloc_list_cell_from_raw(scope, receiver, NIL).get()
    });
    jit_call_method(fn_obj, args_list)
}

/// `(method-sym receiver arg1)` — same as above, two user args.
/// Used by the binary protocol forms (e.g. `-nth`'s 2-arg arity).
///
/// Same NIL-as-fallback contract as `invoke_method_0` — see that
/// docstring before adding a new caller.
pub fn invoke_method_1(method_sym: u32, receiver: u64, arg1: u64) -> u64 {
    let type_sym = match value_type_name_sym(receiver) {
        Some(s) => s,
        None => return NIL,
    };
    let fn_obj = with_host(|h| {
        let table = h.method_table.lock().unwrap();
        let idx = match table.get(&(method_sym, type_sym)).copied() {
            Some(i) => i,
            None => return NIL,
        };
        h.method_roots.get(idx)
    });
    if !v::is_ptr(fn_obj) {
        return NIL;
    }
    let args_list = dynobj::roots::with_scope(6, |scope| {
        let tail =
            v::alloc_list_cell_from_raw(scope, arg1, NIL).get();
        v::alloc_list_cell_from_raw(scope, receiver, tail).get()
    });
    jit_call_method(fn_obj, args_list)
}

fn jit_call_method(fn_obj: u64, args_list: u64) -> u64 {
    let fr = ns::fn_func_ref(fn_obj);
    // Use `gc.run_jit` rather than a direct `function_ptr` call so the
    // GC's safepoint session is installed for the duration of the
    // method body. Without this, the body's safepoint poll panics
    // ("no active JIT safepoint session installed") when we're not
    // already inside a JIT extern.
    with_host(|h| {
        debug_assert!(!h.gc.is_null(), "protocol: host has no GC runtime");
        debug_assert!(!h.jit.is_null(), "protocol: host has no JitModule");
        let gc = unsafe { &*h.gc };
        let jit = unsafe { &*h.jit };
        match gc.run_jit(
            jit,
            FuncRef::from_u32(fr),
            &[fn_obj, args_list],
            dynlang::GcPolicy::NeverAuto,
        ) {
            dynlower::JitOutcome::Value(v) => v,
            dynlower::JitOutcome::Void => crate::value::NIL,
            other => panic!(
                "protocol: unexpected JIT outcome from method call: {other:?}"
            ),
        }
    })
}

/// True iff `receiver`'s type has a method named `method_sym`
/// registered. Cheap — no JIT call. Used for shape-detection in the
/// printer (`has_method(IVector's marker, x)`-style probes).
pub fn has_method(method_sym: u32, receiver: u64) -> bool {
    let type_sym = match value_type_name_sym(receiver) {
        Some(s) => s,
        None => return false,
    };
    with_host(|h| {
        let table = h.method_table.lock().unwrap();
        table.contains_key(&(method_sym, type_sym))
    })
}

/// True iff `receiver`'s type was registered (via `extend-type`) as
/// satisfying the protocol named by `proto_sym`. Mirrors the runtime
/// `clj_satisfies_p` extern but lock-free w.r.t. `host.sym`.
pub fn type_satisfies(proto_sym: u32, receiver: u64) -> bool {
    let type_sym = match value_type_name_sym(receiver) {
        Some(s) => s,
        None => return false,
    };
    with_host(|h| {
        let mem = h.protocol_membership.lock().unwrap();
        mem.contains(&(type_sym, proto_sym))
    })
}

/// Resolve a heap value's user-facing type-name symbol. Same logic
/// as `clj_method_lookup`: records carry their type name on the
/// instance; built-ins look up via `host.builtin_type_names`.
/// Returns None for non-heap values (numbers, bools, nil) and for
/// heap types that haven't been mapped to a name.
pub fn value_type_name_sym(x: u64) -> Option<u32> {
    if !v::is_ptr(x) {
        return None;
    }
    if crate::collections::is_record(x) {
        let t = crate::collections::record_type_name(x);
        if v::is_sym_id(t) {
            return Some(v::as_sym_id(t));
        }
        return None;
    }
    let type_id = unsafe { v::read_type_id(v::as_ptr(x)) } as usize;
    with_host(|h| {
        h.builtin_type_names
            .lock()
            .unwrap()
            .get(&type_id)
            .copied()
    })
}
