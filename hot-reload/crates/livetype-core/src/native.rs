//! The C-ABI value/frame representation and runtime externs for natively
//! compiled `step` functions — the LLVM-free half of the JIT contract.
//!
//! The compiler for `step` functions lives in the parent `livetype` crate (it
//! links LLVM); everything native code touches at *run time* is defined here,
//! against [`Shared`], so the engine can execute compiled code without this
//! crate ever seeing an LLVM type. Under Miri (where no compiled code can
//! exist) none of the extern entry points are ever reached.

use crate::mt::Shared;
use crate::{Condition, DefId, FieldId, ForeignFnId, ObjectId, Value, VariantId};

pub const TAG_EMPTY: i64 = 0;
pub const TAG_UNIT: i64 = 1;
pub const TAG_I64: i64 = 2;
pub const TAG_BOOL: i64 = 3;
pub const TAG_REF: i64 = 4;
pub const TAG_FOREIGN: i64 = 5;
/// An interned string: the payload is its [`crate::StrId`]. A scalar — nothing
/// to trace, and equal ids are equal strings (interning dedups).
pub const TAG_STR: i64 = 6;
/// Low-byte mask for the tag. A `Foreign` slot carries its (small `u32`) kind in
/// the tag's high bits and its native pointer in the payload, so a two-word slot
/// still represents every value — no wider frame layout needed. Only `Foreign`
/// uses the high bits; every other tag is a bare low value, so the exact-`==`
/// tag guards the codegen emits for arithmetic/branch operands are unaffected.
pub const TAG_KIND_SHIFT: i64 = 8;
pub const TAG_MASK: i64 = 0xff;

/// One typed register slot. For `TAG_REF` the payload is the [`ObjectId`] — a
/// GC root the collector reads directly out of the frame. For `TAG_FOREIGN` the
/// payload is the native pointer and the kind is in `tag >> TAG_KIND_SHIFT`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawSlot {
    pub tag: i64,
    pub payload: i64,
}

impl RawSlot {
    pub const EMPTY: RawSlot = RawSlot {
        tag: TAG_EMPTY,
        payload: 0,
    };

    pub fn from_value(value: &Value) -> RawSlot {
        match value {
            Value::Unit => RawSlot {
                tag: TAG_UNIT,
                payload: 0,
            },
            Value::I64(n) => RawSlot {
                tag: TAG_I64,
                payload: *n,
            },
            Value::Bool(b) => RawSlot {
                tag: TAG_BOOL,
                payload: *b as i64,
            },
            Value::Str(id) => RawSlot {
                tag: TAG_STR,
                payload: *id as i64,
            },
            Value::Ref(id) => RawSlot {
                tag: TAG_REF,
                payload: *id as i64,
            },
            Value::Foreign { kind, ptr } => RawSlot {
                tag: TAG_FOREIGN | ((*kind as i64) << TAG_KIND_SHIFT),
                payload: *ptr as i64,
            },
        }
    }

    pub fn to_value(self) -> Value {
        match self.tag & TAG_MASK {
            TAG_UNIT => Value::Unit,
            TAG_I64 => Value::I64(self.payload),
            TAG_BOOL => Value::Bool(self.payload != 0),
            TAG_STR => Value::Str(self.payload as crate::StrId),
            TAG_REF => Value::Ref(self.payload as ObjectId),
            TAG_FOREIGN => Value::Foreign {
                kind: (self.tag >> TAG_KIND_SHIFT) as u32,
                ptr: self.payload as u64,
            },
            other => panic!("empty or unknown slot tag {other} escaped a step boundary"),
        }
    }
}

/// The heap-resident frame native code operates on. Its LLVM struct layout
/// (built by the compiler in the `livetype` crate) matches this `#[repr(C)]`
/// exactly.
#[repr(C)]
pub struct RawFrame {
    pub func_id: i64,
    pub version: i64,
    pub pc: i64,
    pub n_regs: i64,
    pub regs: *mut RawSlot,
    pub scratch: RawSlot,
    pub return_reg: i64,
}

/// A constructor field passed to `lt_new` — laid out as three consecutive
/// `i64`s (`field_id`, then the slot's `tag`/`payload`), which is what the
/// codegen writes into its stack array.
#[repr(C)]
pub struct SuppliedField {
    pub field_id: i64,
    pub value: RawSlot,
}

// Native `step` outcomes (the function's `i64` return).
pub const OUT_RETURN: i64 = 0;
pub const OUT_CALL: i64 = 1;
pub const OUT_CONDITION: i64 = 2;
pub const OUT_YIELD: i64 = 3;
/// An operand-tag check failed (`SubI64`/`LtI64`/`Branch` saw a value of the
/// wrong representation — the con-freeness trap). The driver reconstructs the
/// exact condition from the instruction at `frame->pc` so it matches the
/// interpreter's.
pub const OUT_TYPE_ERROR: i64 = 4;

/// A compiled `step` function's signature.
pub type StepFn = unsafe extern "C" fn(*mut RawFrame, *mut NativeHost) -> i64;

/// Transmute a compiled step address to a callable. The address must come from
/// a live execution engine; the compiler side guarantees engines outlive every
/// caller (it leaks one per world epoch).
pub fn step_at(addr: usize) -> StepFn {
    unsafe { std::mem::transmute::<usize, StepFn>(addr) }
}

/// The runtime side of a native step — allocation, migration, and effects —
/// behind one type so there is a single set of externs. It is a thin bridge to
/// the *same* [`Shared`] operations the interpreter uses, so compiled code
/// cannot diverge on them. A trapped condition is stashed per call (per thread)
/// for the driver to pick up on an `OUT_CONDITION`.
pub struct NativeHost<'a> {
    pub shared: &'a Shared,
    pub pending: Option<Condition>,
}

impl<'a> NativeHost<'a> {
    pub fn new(shared: &'a Shared) -> NativeHost<'a> {
        NativeHost {
            shared,
            pending: None,
        }
    }
    pub fn take_pending(&mut self) -> Option<Condition> {
        self.pending.take()
    }
}

/// Returns 0 on success (writes `*out_objid`), 1 when construction trips the
/// soundness check (the condition is stashed in the host).
///
/// # Safety
/// `host` is a live `*mut NativeHost`, `fields` points to `n` `SuppliedField`s,
/// `out_objid` is a writable `*mut i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_new(
    host: *mut NativeHost,
    type_id: i64,
    fields: *const SuppliedField,
    n: i64,
    out_objid: *mut i64,
) -> i64 {
    let host = unsafe { &mut *host };
    let mut supplied = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        let f = unsafe { &*fields.offset(i) };
        supplied.push((f.field_id as FieldId, f.value.to_value()));
    }
    match host.shared.jit_new(type_id as DefId, &supplied) {
        Ok(id) => {
            unsafe { *out_objid = id as i64 };
            0
        }
        Err(condition) => {
            host.pending = Some(condition);
            1
        }
    }
}

/// Returns 0 on success (writes `*out`), 1 when a migration barrier trips (the
/// condition is stashed in the host for the driver).
///
/// # Safety
/// `host` is a live `*mut NativeHost`, `out` a writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_get_field(
    host: *mut NativeHost,
    objid: i64,
    field: i64,
    out: *mut RawSlot,
) -> i64 {
    let host = unsafe { &mut *host };
    match host.shared.jit_get_field(objid as ObjectId, field as FieldId) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            host.pending = Some(condition);
            1
        }
    }
}

/// # Safety
/// `host` is a live `*mut NativeHost`, `value` a readable `*const RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_emit(host: *mut NativeHost, value: *const RawSlot) {
    let host = unsafe { &mut *host };
    host.shared.jit_emit(unsafe { *value }.to_value());
}

/// Returns 0 on success (writes the result to `*out`), 1 when the call traps
/// (unregistered fn, or a native return that fails the type check) — the
/// condition is stashed in the host.
///
/// # Safety
/// `host` is a live `*mut NativeHost`, `args` points to `n` `RawSlot`s, `out`
/// is a writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_call_foreign(
    host: *mut NativeHost,
    foreign: i64,
    args: *const RawSlot,
    n: i64,
    out: *mut RawSlot,
) -> i64 {
    let host = unsafe { &mut *host };
    let mut values = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        values.push(unsafe { *args.offset(i) }.to_value());
    }
    match host.shared.jit_call_foreign(foreign as ForeignFnId, &values) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            host.pending = Some(condition);
            1
        }
    }
}

/// Returns 0 on success (writes `*out`), 1 when the global is unset (condition
/// stashed in the host).
///
/// # Safety
/// `host` is a live `*mut NativeHost`, `out` a writable `*mut RawSlot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_load_global(
    host: *mut NativeHost,
    global: i64,
    out: *mut RawSlot,
) -> i64 {
    let host = unsafe { &mut *host };
    match host.shared.jit_load_global(global as DefId) {
        Ok(value) => {
            unsafe { *out = RawSlot::from_value(&value) };
            0
        }
        Err(condition) => {
            host.pending = Some(condition);
            1
        }
    }
}

/// Concatenate two interned strings (the `ConcatStr` fast op). Interning
/// cannot fail, so this returns the result id directly — no host, no status.
///
/// # Safety
/// Trivially safe (integer in, integer out); `extern "C"` for the JIT.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_concat_str(left: i64, right: i64) -> i64 {
    crate::strings::concat(left as crate::StrId, right as crate::StrId) as i64
}

/// Construct an enum variant — the enum counterpart of [`lt_new`]. Returns 0
/// on success (writes `*out_objid`), 1 when construction trips the soundness
/// check (the condition is stashed in the host).
///
/// # Safety
/// `host` is a live `*mut NativeHost`, `fields` points to `n` `SuppliedField`s,
/// `out_objid` is a writable `*mut i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_new_variant(
    host: *mut NativeHost,
    type_id: i64,
    variant: i64,
    fields: *const SuppliedField,
    n: i64,
    out_objid: *mut i64,
) -> i64 {
    let host = unsafe { &mut *host };
    let mut supplied = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        let f = unsafe { &*fields.offset(i) };
        supplied.push((f.field_id as FieldId, f.value.to_value()));
    }
    match host
        .shared
        .jit_new_variant(type_id as DefId, variant as VariantId, &supplied)
    {
        Ok(id) => {
            unsafe { *out_objid = id as i64 };
            0
        }
        Err(condition) => {
            host.pending = Some(condition);
            1
        }
    }
}

/// The `match` barrier (migrate + variant lookup + unhandled-variant trap):
/// returns 0 on success and writes the matching ARM INDEX to `*out_index`; 1
/// when the barrier traps (missing migration, unhandled variant — condition
/// stashed in the host). `arms` points to `n` variant ids in arm order.
///
/// # Safety
/// `host` is a live `*mut NativeHost`, `arms` points to `n` `i64`s,
/// `out_index` is a writable `*mut i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lt_case_variant(
    host: *mut NativeHost,
    objid: i64,
    arms: *const i64,
    n: i64,
    out_index: *mut i64,
) -> i64 {
    let host = unsafe { &mut *host };
    let mut arm_ids = Vec::with_capacity(n as usize);
    for i in 0..n as isize {
        arm_ids.push((unsafe { *arms.offset(i) } as VariantId, 0usize));
    }
    match host.shared.jit_case_variant(objid as ObjectId, &arm_ids) {
        Ok(index) => {
            unsafe { *out_index = index as i64 };
            0
        }
        Err(condition) => {
            host.pending = Some(condition);
            1
        }
    }
}
