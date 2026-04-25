use beagle::lower::{IcContext, lower_program, StringPool, STRING_TAG};
use beagle::parser::Parser;

use dynexec::NanBoxConfig;
use dynlower::JitOutcome;
use dynsym::Symbol;
use dynvalue::{Decoded, NanBox, TagScheme};

/// NaN-box tag pattern — needed locally only for `nanbox_nil`. The GC's
/// ptr tag, PtrPolicy, and safepoint wiring all live inside `DynGcRuntime`.
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;

/// Layout info for the synthetic `__Array__` GC type, captured at
/// lowering time. `ext_length` and `ext_get` consult this to recognise
/// array NanBoxes (vs. nil / scalar) and read their `len` field.
#[derive(Clone, Copy)]
struct ArrayInfo {
    type_id_u16: u16,
    len_offset: i32,
}

thread_local! {
    static STRINGS: std::cell::RefCell<Option<StringPool>> =
        const { std::cell::RefCell::new(None) };

    /// Inline-cache state for dynamic property access. The JIT embeds a raw
    /// pointer into `IcContext::array` at compile time, so this must stay
    /// alive (and at a stable address) for the lifetime of the JIT module.
    static IC: std::cell::RefCell<Option<IcContext>> =
        const { std::cell::RefCell::new(None) };

    static ARRAY_INFO: std::cell::Cell<Option<ArrayInfo>> =
        const { std::cell::Cell::new(None) };

    /// Monotonic origin for `core/time-now()`. Set the first time the
    /// extern is called; subsequent calls report nanoseconds since.
    static TIME_ORIGIN: std::cell::Cell<Option<std::time::Instant>> =
        const { std::cell::Cell::new(None) };
}

// ── JIT extern thunks ─────────────────────────────────────────────
//
// `__gc_alloc__` is NOT listed here — `DynGcRuntime::compile_jit` binds
// it automatically. Language authors who try to bind it manually are
// overridden.

extern "C" fn ext_print(val: u64) {
    print_value(val, false);
}

extern "C" fn ext_println(val: u64) {
    print_value(val, true);
}

/// `length(v)`: returns the number of elements when `v` is an `__Array__`
/// pointer, else falls back to the binary_trees stub (nil → 0, scalar → 1).
extern "C" fn ext_length(val: u64) -> u64 {
    if let Some(len) = array_len_of(val) {
        return encode_f64_int(len as i64);
    }
    if val == nanbox_nil() {
        encode_f64_int(0)
    } else {
        encode_f64_int(1)
    }
}

/// `get(v, i)`: array index for `__Array__`, scalar passthrough otherwise
/// (matches the binary_trees stub where `args` is treated as a one-element
/// vector).
extern "C" fn ext_get(val: u64, idx: u64) -> u64 {
    if let Some(elem) = array_elem_at(val, idx) {
        return elem;
    }
    val
}

extern "C" fn ext_to_number(val: u64) -> u64 {
    val
}

extern "C" fn ext_cos(val: u64) -> u64 {
    let f = f64::from_bits(val);
    f.cos().to_bits()
}

extern "C" fn ext_sin(val: u64) -> u64 {
    let f = f64::from_bits(val);
    f.sin().to_bits()
}

/// `core/time-now()`: nanoseconds since the first invocation, encoded as
/// a NanBox float. The bench only uses this for elapsed-time subtraction,
/// so an arbitrary monotonic origin is fine.
extern "C" fn ext_time_now() -> u64 {
    let now = std::time::Instant::now();
    let origin = TIME_ORIGIN.with(|cell| {
        let cur = cell.get();
        match cur {
            Some(o) => o,
            None => {
                cell.set(Some(now));
                now
            }
        }
    });
    let ns = now.duration_since(origin).as_nanos() as u64;
    encode_f64_int(ns as i64)
}

macro_rules! slow_stub {
    ($name:ident) => {
        extern "C" fn $name(a: u64, b: u64) -> u64 {
            panic!(
                "beagle slow-path `{}` hit: a=0x{:x} b=0x{:x} \
                 — binary_trees subset shouldn't need it",
                stringify!($name),
                a,
                b,
            );
        }
    };
}

slow_stub!(ext_add);
slow_stub!(ext_sub);
slow_stub!(ext_mul);
slow_stub!(ext_div);
slow_stub!(ext_eq);
slow_stub!(ext_lt);
slow_stub!(ext_gt);

extern "C" fn ext_neg(_v: u64) -> u64 {
    panic!("beagle slow-path `ext_neg` hit");
}
extern "C" fn ext_not(_v: u64) -> u64 {
    panic!("beagle slow-path `ext_not` hit");
}

/// IC miss: the JIT calls this when `cached_class_id` didn't match the
/// object's `TypeInfo*`. We read the TypeInfo pointer from the header,
/// look up the field's byte offset in the per-type dispatch table, write
/// that back into the IC entry so the next call takes the fast path, and
/// return the loaded field value.
///
/// Safety: this must not allocate. The object pointer is a live GC root on
/// the caller's frame; there are no safepoints in here.
/// The dynalloc semispace collector sets `1 << 63` in the header word
/// when it moves an object, using the low 63 bits as the to-space
/// address (see `dynalloc::semi_space::FORWARDING_BIT`).
const FORWARDING_BIT: u64 = 1 << 63;

extern "C" fn ext_prop_slow(obj_bits: u64, sym_id: u64, cache_id: u64) -> u64 {
    // Decode the NanBox to a raw object pointer. Tag 2 is the default
    // pointer tag (see `NanBoxTags::default`).
    let mut raw_ptr = match NanBox::decode(obj_bits) {
        Decoded::Tagged { tag: 2, payload } => payload as usize as *const u8,
        other => panic!("beagle: property access on non-object NanBox: {:?}", other),
    };

    // Read the header word. For `dynobj::Compact`, that's u16 type_id at
    // offset 0 + 6 bytes of zeroed padding. But this extern runs at the
    // call-site safepoint: the stack slot holding our NanBox is updated
    // by the GC, the arg register is not — so `raw_ptr` may already be a
    // stale from-space address whose header now carries a forwarding
    // pointer. Follow it before keying the cache.
    let mut header = unsafe { *(raw_ptr as *const u64) };
    if header & FORWARDING_BIT != 0 {
        raw_ptr = (header & !FORWARDING_BIT) as usize as *const u8;
        header = unsafe { *(raw_ptr as *const u64) };
        debug_assert_eq!(
            header & FORWARDING_BIT,
            0,
            "beagle: forwarding pointer chain (to-space object {:p} also forwarded)",
            raw_ptr,
        );
    }

    let class_key = header + 1;
    let sym = Symbol::from_raw(sym_id as u32);

    IC.with(|cell| {
        let mut guard = cell.borrow_mut();
        let ic = guard.as_mut().expect("IC not installed");
        let table = ic.per_type.get(&class_key).unwrap_or_else(|| {
            panic!(
                "beagle: no dispatch table for class_key {} (sym `{}`)",
                class_key,
                ic.symbols.try_name(sym).unwrap_or("<unknown>"),
            )
        });
        let offset = table.get(sym).unwrap_or_else(|| {
            panic!(
                "beagle: class_key {} has no field `{}`",
                class_key,
                ic.symbols.try_name(sym).unwrap_or("<unknown>"),
            )
        });

        let entry = ic.array.get_mut(cache_id as u32);
        entry.cached_class_id = class_key;
        entry.cached_value = offset;

        unsafe { *(raw_ptr.add(offset as usize) as *const u64) }
    })
}

/// User extern map. `__gc_alloc__` is intentionally absent — the runtime
/// owns it. Returning `None` tells dynlang the extern is unresolved.
fn jit_extern_for(name: &str) -> Option<*const u8> {
    Some(match name {
        "beagle_print" => ext_print as *const u8,
        "beagle_println" => ext_println as *const u8,
        "beagle_length" => ext_length as *const u8,
        "beagle_get" => ext_get as *const u8,
        "beagle_to_number" => ext_to_number as *const u8,
        "beagle_cos" => ext_cos as *const u8,
        "beagle_sin" => ext_sin as *const u8,
        "beagle_time_now" => ext_time_now as *const u8,
        "beagle_add" => ext_add as *const u8,
        "beagle_sub" => ext_sub as *const u8,
        "beagle_mul" => ext_mul as *const u8,
        "beagle_div" => ext_div as *const u8,
        "beagle_neg" => ext_neg as *const u8,
        "beagle_eq" => ext_eq as *const u8,
        "beagle_lt" => ext_lt as *const u8,
        "beagle_gt" => ext_gt as *const u8,
        "beagle_not" => ext_not as *const u8,
        "beagle_prop_slow" => ext_prop_slow as *const u8,
        _ => return None,
    })
}

fn main() {
    // Beagle's binary_trees recursion (doWorkHelper with iterations up to
    // 2M at n=21) blows the default 8 MiB thread stack. Spawn a worker
    // thread with room for it.
    let handle = std::thread::Builder::new()
        .name("beagle-main".into())
        .stack_size(2 * 1024 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn beagle-main");
    handle.join().expect("beagle-main panicked");
}

fn real_main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: beagle <file.bg> [N]");
        std::process::exit(1);
    }
    let path = &args[1];
    let cli_n: Option<i64> = args.get(2).map(|s| {
        s.parse::<i64>()
            .unwrap_or_else(|_| panic!("expected integer for N, got {s:?}"))
    });

    let src = std::fs::read_to_string(path).expect("read source file");
    let mut parser = Parser::new(path.clone(), src).expect("create parser");
    let ast = parser.parse().expect("parse");
    let lowered = lower_program(&ast);

    STRINGS.with(|cell| *cell.borrow_mut() = Some(lowered.strings));
    // Install the IC *before* JIT compile: the compiled code embeds a raw
    // pointer into `ic.array`. Moving the IcContext into the thread-local
    // keeps that pointer valid (Vec's heap storage doesn't move on move).
    IC.with(|cell| *cell.borrow_mut() = Some(lowered.ic));
    ARRAY_INFO.with(|cell| {
        cell.set(Some(ArrayInfo {
            type_id_u16: lowered.array_type_id_u16,
            len_offset: lowered.array_len_offset,
        }))
    });

    // Compile with `NanBoxConfig` (precise stack maps) + LinearScan regalloc
    // for tighter frames (binary_trees recurses millions deep).
    //
    // `gc.compile_jit` auto-binds `__gc_alloc__` and wires the safepoint
    // handler — language code supplies only its own externs.
    let jit = lowered.gc.compile_jit::<
        NanBoxConfig,
        dynlower::Arm64Backend,
        dynlower::regalloc::LinearScanAllocator,
    >(&lowered.module, jit_extern_for);

    if std::env::var("BEAGLE_DUMP_IR").is_ok() {
        for func in &lowered.module.functions {
            eprintln!("{}", func);
            eprintln!("---");
        }
    }

    if std::env::var("BEAGLE_DUMP_JIT").is_ok() {
        jit.dump_code();
    }

    let gc_threshold: f64 = std::env::var("BEAGLE_GC_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.75);

    // Build the `args` nanbox. If an N was passed on the CLI, encode
    // it as a plain float NanBox and rely on beagle_length/get/to_number
    // stubs above to treat it as a single-element arg vector.
    let args_val = match cli_n {
        Some(n) => encode_f64_int(n),
        None => nanbox_nil(),
    };

    // Match the call to `main`'s declared arity. binary_trees uses
    // `fn main(args)`; ray_cast_bench uses `fn main()` — passing one
    // arg to the latter would mismatch the function's signature.
    let main_args: Vec<u64> = match lowered.main_arity {
        0 => vec![],
        1 => vec![args_val],
        n => panic!("beagle: fn main must take 0 or 1 args, got {n}"),
    };

    // `run_jit` installs the stack-map safepoint session, the PtrPolicy,
    // and this thread as the active runtime for `__gc_alloc__` callbacks.
    let result =
        lowered
            .gc
            .run_jit_with_threshold(&jit, lowered.main, &main_args, gc_threshold);

    match result {
        JitOutcome::Value(_) | JitOutcome::Void => {}
        other => {
            eprintln!("beagle: JIT error: {:?}", other);
            std::process::exit(1);
        }
    }
}

fn nanbox_nil() -> u64 {
    TAG_PATTERN // tag 0, payload 0
}

fn encode_f64_int(n: i64) -> u64 {
    (n as f64).to_bits()
}

/// If `bits` is a pointer-tagged NanBox referring to an `__Array__`
/// object, return the array's logical length. Returns `None` otherwise.
fn array_len_of(bits: u64) -> Option<usize> {
    let info = ARRAY_INFO.with(|c| c.get())?;
    let raw = decode_array_ptr(bits, info)?;
    let len = unsafe { *(raw.add(info.len_offset as usize) as *const u64) };
    Some(len as usize)
}

/// If `bits` is a pointer-tagged NanBox referring to an `__Array__`
/// object, return its `idx`-th element. `idx` must be a NanBox-encoded
/// float; we truncate to i64. Returns `None` if the value isn't an array.
fn array_elem_at(bits: u64, idx_bits: u64) -> Option<u64> {
    let info = ARRAY_INFO.with(|c| c.get())?;
    let raw = decode_array_ptr(bits, info)?;
    // Element base = varlen_element_offset(0). We don't have that here —
    // but the `len` field is a Raw64 placed right at the end of the fixed
    // section, with the varlen elements immediately after, so element 0
    // sits at `len_offset + 8`.
    let elem_base = info.len_offset as usize + 8;
    let idx_f = f64::from_bits(idx_bits);
    let idx = idx_f as i64 as usize;
    let addr = unsafe { raw.add(elem_base + idx * 8) } as *const u64;
    Some(unsafe { *addr })
}

/// Common prefix: confirm `bits` is a ptr-tagged NanBox whose object
/// header carries the array's u16 type_id, walk any forwarding pointer,
/// return the live raw pointer.
fn decode_array_ptr(bits: u64, info: ArrayInfo) -> Option<*const u8> {
    let mut raw = match NanBox::decode(bits) {
        Decoded::Tagged { tag: 2, payload } => payload as usize as *const u8,
        _ => return None,
    };
    let mut header = unsafe { *(raw as *const u64) };
    if header & FORWARDING_BIT != 0 {
        raw = (header & !FORWARDING_BIT) as usize as *const u8;
        header = unsafe { *(raw as *const u64) };
    }
    // Compact header: u16 type_id at offset 0, padded with zeroes to a
    // full word, so `header as u16` is the type_id.
    if (header as u16) != info.type_id_u16 {
        return None;
    }
    Some(raw)
}

fn print_value(bits: u64, newline: bool) {
    match NanBox::decode(bits) {
        Decoded::Float(f) => {
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
                print!("{}", f as i64);
            } else {
                print!("{}", f);
            }
        }
        Decoded::Tagged { tag, payload } => match tag {
            0 => print!("null"),
            1 => print!("{}", if payload == 0 { "false" } else { "true" }),
            2 => print!("<ptr {:#x}>", payload),
            t if t == STRING_TAG => STRINGS.with(|cell| {
                let guard = cell.borrow();
                let pool = guard.as_ref().expect("string pool not installed");
                print!("{}", pool.get(payload as u32).unwrap_or("<bad str>"));
            }),
            _ => print!("<tag{} {:#x}>", tag, payload),
        },
    }
    if newline {
        println!();
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
}
