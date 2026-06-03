//! Primitive externs callable from microlisp code.
//!
//! Each primitive takes and returns NanBox-encoded `u64` values. They run
//! against the host context installed by `Engine::run` (via
//! `dynlang::host`-style mechanism, but kept simple — we use a thread-local
//! pointer set in `host.rs`).

use crate::host::with_host;
use crate::value::*;

// ── Numeric ───────────────────────────────────────────────────────

pub extern "C" fn ml_add(a: u64, b: u64) -> u64 {
    encode_num(num(a) + num(b))
}
pub extern "C" fn ml_sub(a: u64, b: u64) -> u64 {
    encode_num(num(a) - num(b))
}
pub extern "C" fn ml_mul(a: u64, b: u64) -> u64 {
    encode_num(num(a) * num(b))
}
pub extern "C" fn ml_div(a: u64, b: u64) -> u64 {
    encode_num(num(a) / num(b))
}
pub extern "C" fn ml_neg(a: u64) -> u64 {
    encode_num(-num(a))
}
pub extern "C" fn ml_num_eq(a: u64, b: u64) -> u64 {
    bool_(num(a) == num(b))
}
pub extern "C" fn ml_lt(a: u64, b: u64) -> u64 {
    bool_(num(a) < num(b))
}
pub extern "C" fn ml_gt(a: u64, b: u64) -> u64 {
    bool_(num(a) > num(b))
}
pub extern "C" fn ml_le(a: u64, b: u64) -> u64 {
    bool_(num(a) <= num(b))
}
pub extern "C" fn ml_ge(a: u64, b: u64) -> u64 {
    bool_(num(a) >= num(b))
}

// ── Cons / list ───────────────────────────────────────────────────

pub extern "C" fn ml_cons(a: u64, b: u64) -> u64 {
    // FFI boundary: JIT hands us raw bits via the C ABI. Root them in a
    // fresh scope so any `gc.alloc`-fired collection (none today, but the
    // contract permits it) sees them as live; the result is freshly
    // returned to JIT where the next safepoint's stack map roots it.
    // Three slots: car + cdr + result.
    dynobj::roots::with_scope(3, |scope| alloc_cons_from_raw(scope, a, b).get())
}
pub extern "C" fn ml_car(v: u64) -> u64 {
    car(v)
}
pub extern "C" fn ml_cdr(v: u64) -> u64 {
    cdr(v)
}
pub extern "C" fn ml_set_car(v: u64, x: u64) -> u64 {
    set_car(v, x);
    NIL
}
pub extern "C" fn ml_set_cdr(v: u64, x: u64) -> u64 {
    set_cdr(v, x);
    NIL
}

pub extern "C" fn ml_null_p(v: u64) -> u64 {
    bool_(is_nil(v))
}
pub extern "C" fn ml_pair_p(v: u64) -> u64 {
    bool_(is_cons(v))
}
pub extern "C" fn ml_symbol_p(v: u64) -> u64 {
    bool_(is_symbol(v))
}
pub extern "C" fn ml_number_p(v: u64) -> u64 {
    bool_(is_number(v))
}

pub extern "C" fn ml_eq_p(a: u64, b: u64) -> u64 {
    bool_(a == b)
}
pub extern "C" fn ml_equal_p(a: u64, b: u64) -> u64 {
    bool_(equal(a, b))
}
pub extern "C" fn ml_not(v: u64) -> u64 {
    bool_(!is_true_value(v))
}

/// `append` — copy the spine of `a`, then connect to `b`. Allocates one
/// cons per element of `a`; each allocation gets its own fresh scope so
/// we don't grow the outer scope unbounded for long lists.
pub extern "C" fn ml_append(a: u64, b: u64) -> u64 {
    if is_nil(a) {
        return b;
    }
    dynobj::roots::with_scope(2, |scope| {
        let a_root = scope.root::<NanBoxTag>(a);
        let elems: Vec<u64> = list_iter(a_root.get()).collect();
        let tail = scope.root::<NanBoxTag>(b);
        for x in elems.into_iter().rev() {
            let new_bits = dynobj::roots::with_scope(3, |inner| {
                alloc_cons_from_raw(inner, x, tail.get()).get()
            });
            tail.set(new_bits);
        }
        tail.get()
    })
}

pub extern "C" fn ml_length(v: u64) -> u64 {
    encode_int(list_len(v) as i64)
}

// ── Symbol helpers ────────────────────────────────────────────────

/// `gensym` with a given symbol/string tag. We accept a symbol for the tag
/// (because we don't have first-class strings in v0); if anything else is
/// passed we just fall back to "g".
pub extern "C" fn ml_gensym(tag: u64) -> u64 {
    with_host(|h| {
        let mut sym = h.sym.borrow_mut();
        let tag_str: String = if is_symbol(tag) {
            sym.name(as_symbol_id(tag)).to_string()
        } else {
            "g".to_string()
        };
        let id = sym.gensym(&tag_str);
        encode_sym(id)
    })
}

/// Concatenate the printed names of two symbols and intern as a new symbol.
pub extern "C" fn ml_symbol_append(a: u64, b: u64) -> u64 {
    with_host(|h| {
        let mut sym = h.sym.borrow_mut();
        let an = if is_symbol(a) {
            sym.name(as_symbol_id(a)).to_string()
        } else {
            String::new()
        };
        let bn = if is_symbol(b) {
            sym.name(as_symbol_id(b)).to_string()
        } else {
            String::new()
        };
        let id = sym.intern(&format!("{an}{bn}"));
        encode_sym(id)
    })
}

// ── I/O ───────────────────────────────────────────────────────────

pub extern "C" fn ml_print(v: u64) -> u64 {
    with_host(|h| {
        let sym = h.sym.borrow();
        let out = crate::printer::print(v, &sym);
        println!("{out}");
    });
    NIL
}

pub extern "C" fn ml_error(v: u64) -> u64 {
    with_host(|h| {
        let sym = h.sym.borrow();
        let out = crate::printer::print(v, &sym);
        panic!("microlisp error: {out}");
    });
    NIL
}

// ── helpers ───────────────────────────────────────────────────────

fn num(v: u64) -> f64 {
    if !is_number(v) {
        panic!("expected number, got 0x{:016x}", v);
    }
    as_number(v)
}

fn bool_(b: bool) -> u64 {
    if b { TRUE } else { FALSE }
}

// ── Registry: name → (fn pointer, signature shape) ────────────────

#[derive(Debug, Clone, Copy)]
pub enum PrimSig {
    Unary,
    Binary,
}

pub struct Prim {
    pub name: &'static str,
    pub ptr: *const u8,
    pub sig: PrimSig,
}

unsafe impl Sync for Prim {}

pub fn all_prims() -> &'static [Prim] {
    static PRIMS: &[Prim] = &[
        Prim {
            name: "+",
            ptr: ml_add as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "-",
            ptr: ml_sub as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "*",
            ptr: ml_mul as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "/",
            ptr: ml_div as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "neg",
            ptr: ml_neg as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "=",
            ptr: ml_num_eq as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "<",
            ptr: ml_lt as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: ">",
            ptr: ml_gt as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "<=",
            ptr: ml_le as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: ">=",
            ptr: ml_ge as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "cons",
            ptr: ml_cons as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "car",
            ptr: ml_car as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "cdr",
            ptr: ml_cdr as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "set-car!",
            ptr: ml_set_car as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "set-cdr!",
            ptr: ml_set_cdr as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "null?",
            ptr: ml_null_p as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "pair?",
            ptr: ml_pair_p as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "symbol?",
            ptr: ml_symbol_p as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "number?",
            ptr: ml_number_p as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "eq?",
            ptr: ml_eq_p as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "equal?",
            ptr: ml_equal_p as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "not",
            ptr: ml_not as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "append",
            ptr: ml_append as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "length",
            ptr: ml_length as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "gensym",
            ptr: ml_gensym as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "symbol-append",
            ptr: ml_symbol_append as *const u8,
            sig: PrimSig::Binary,
        },
        Prim {
            name: "print",
            ptr: ml_print as *const u8,
            sig: PrimSig::Unary,
        },
        Prim {
            name: "error",
            ptr: ml_error as *const u8,
            sig: PrimSig::Unary,
        },
    ];
    PRIMS
}
