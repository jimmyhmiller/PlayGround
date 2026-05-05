use beagle::lower::{lower_program, StringPool, STRING_TAG};
use beagle::parser::Parser;

use dynexec::NanBoxConfig;
use dynlang::host::host;
use dynlang::stdlib::IndexedSeq;
use dynlower::JitOutcome;
use dynvalue::{Decoded, NanBox, TagScheme};

/// Per-program host context for JIT extern thunks. Owns the string
/// pool, the array obj-type handle, and the lazily-set monotonic time
/// origin. Installed once via `dynlang::host::install_thread` for the
/// duration of `run_jit`.
struct BeagleHost {
    strings: StringPool,
    array_seq: IndexedSeq,
    time_origin: std::cell::Cell<Option<std::time::Instant>>,
}

// ── JIT extern thunks ─────────────────────────────────────────────
//
// `__gc_alloc__` and `__dynlang_prop_slow__` aren't listed —
// `DynGcRuntime::compile_jit` binds those automatically. The arithmetic
// slow paths default to panic stubs from `dynlang::slow_paths` (also
// auto-bound). Embedder externs go through this map.

extern "C" fn ext_print(val: u64) {
    print_value(val, false);
}

extern "C" fn ext_println(val: u64) {
    print_value(val, true);
}

/// `length(v)`: returns the number of elements when `v` is an `__Array__`
/// pointer, else falls back to the binary_trees stub (nil → 0, scalar → 1).
extern "C" fn ext_length(val: u64) -> u64 {
    let h = host::<BeagleHost>();
    if let Some(view) = h.array_seq.view(val) {
        return NanBox::from_int(view.len() as i64);
    }
    if val == NanBox::NIL {
        NanBox::from_int(0)
    } else {
        NanBox::from_int(1)
    }
}

/// `get(v, i)`: array index for `__Array__`, scalar passthrough otherwise
/// (matches the binary_trees stub where `args` is treated as a one-element
/// vector).
extern "C" fn ext_get(val: u64, idx: u64) -> u64 {
    let h = host::<BeagleHost>();
    if let Some(view) = h.array_seq.view(val) {
        let idx_f = f64::from_bits(idx);
        let idx_i = idx_f as i64 as usize;
        return view.get(idx_i);
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
    let h = host::<BeagleHost>();
    let origin = match h.time_origin.get() {
        Some(o) => o,
        None => {
            h.time_origin.set(Some(now));
            now
        }
    };
    let ns = now.duration_since(origin).as_nanos() as u64;
    NanBox::from_int(ns as i64)
}

/// User extern map. `__gc_alloc__`, `__dynlang_prop_slow__`, and the
/// arithmetic / comparison slow paths are intentionally absent — the
/// runtime auto-binds them (the slow paths default to panic stubs from
/// `dynlang::slow_paths`). Returning `None` tells dynlang the extern is
/// unresolved.
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

    let host = BeagleHost {
        strings: lowered.strings,
        array_seq: lowered.array_seq,
        time_origin: std::cell::Cell::new(None),
    };
    let _host_guard = dynlang::host::install_thread(&host);

    if std::env::var("BEAGLE_DUMP_IR").is_ok() {
        for func in &lowered.module.functions {
            eprintln!("{}", func);
            eprintln!("---");
        }
    }
    // IR-level soundness check: every direct `Call` to a GC allocator
    // (currently just `__gc_alloc__`) must be preceded by `Inst::Safepoint`.
    // Catches frontend bugs at compile time rather than corrupting at
    // the next collection.
    lowered.module.validate_safepoints(&lowered.allocator_frefs);

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

    // BEAGLE_GC_THRESHOLD overrides the default OnPressure threshold.
    // Special string "stress" maps to GcPolicy::EveryPoint (collect at
    // every safepoint — slow, useful for finding root-coverage bugs).
    // Special string "never" disables auto-collect entirely.
    let gc_policy: dynlang::GcPolicy = match std::env::var("BEAGLE_GC_THRESHOLD").as_deref() {
        Ok("stress") => dynlang::GcPolicy::EveryPoint,
        Ok("never") => dynlang::GcPolicy::NeverAuto,
        Ok(s) => match s.parse::<f64>() {
            Ok(threshold) => dynlang::GcPolicy::OnPressure { threshold },
            Err(_) => dynlang::GcPolicy::OnPressure { threshold: 0.75 },
        },
        Err(_) => dynlang::GcPolicy::OnPressure { threshold: 0.75 },
    };

    // Build the `args` nanbox. If an N was passed on the CLI, encode
    // it as a plain float NanBox and rely on beagle_length/get/to_number
    // stubs above to treat it as a single-element arg vector.
    let args_val = match cli_n {
        Some(n) => NanBox::from_int(n),
        None => NanBox::NIL,
    };

    // Match the call to `main`'s declared arity. binary_trees uses
    // `fn main(args)`; ray_cast_bench uses `fn main()` — passing one
    // arg to the latter would mismatch the function's signature.
    let main_args: Vec<u64> = match lowered.main_arity {
        0 => vec![],
        1 => vec![args_val],
        n => panic!("beagle: fn main must take 0 or 1 args, got {n}"),
    };

    // Install the property IC for the duration of the run. The toolkit
    // thunk (`__dynlang_prop_slow__`) reads it from TLS on cache miss.
    // `run_jit` independently installs the stack-map safepoint session,
    // the PtrPolicy, and this thread as the active runtime for
    // `__gc_alloc__` callbacks.
    let _ic_guard = lowered.ic.install_thread();
    let result = lowered
        .gc
        .run_jit(&jit, lowered.main, &main_args, gc_policy);

    match result {
        JitOutcome::Value(_) | JitOutcome::Void => {}
        other => {
            eprintln!("beagle: JIT error: {:?}", other);
            std::process::exit(1);
        }
    }
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
            t if t == STRING_TAG => {
                let h = host::<BeagleHost>();
                print!("{}", h.strings.get(payload as u32).unwrap_or("<bad str>"));
            }
            _ => print!("<tag{} {:#x}>", tag, payload),
        },
    }
    if newline {
        println!();
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
}
