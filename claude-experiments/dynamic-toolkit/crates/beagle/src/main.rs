use beagle::lower::{lower_program, StringPool, STRING_TAG};
use beagle::parser::Parser;

use dynalloc::{PtrPolicy, SemiSpace};
use dynir::dynexec::ContinuationTypes;
use dynir::gc_runtime::GcInterpCtx;
use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
use dynobj::{Compact, TypeInfo};
use dynvalue::{Decoded, NanBox, TagScheme};

/// PtrPolicy hard-coded to the default beagle tag scheme: ptr_tag = 2.
/// dynlang's `NanBoxPolicy` uses a thread-local that the private
/// `DynGcRuntime` sets; since we drive the GC directly, use this.
struct BeaglePtrPolicy;

const BEAGLE_PTR_TAG: u32 = 2;
const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const TAG_FIELD_MASK: u64 = 0x0003_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

impl PtrPolicy for BeaglePtrPolicy {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        let expected = TAG_PATTERN | ((BEAGLE_PTR_TAG as u64) << 48);
        if (bits & (FULL_MASK | TAG_FIELD_MASK)) != expected {
            return None;
        }
        let payload = bits & PAYLOAD_MASK;
        if payload == 0 {
            None
        } else {
            Some(payload as *mut u8)
        }
    }

    fn encode_ptr(ptr: *mut u8) -> u64 {
        TAG_PATTERN | ((BEAGLE_PTR_TAG as u64) << 48) | ((ptr as u64) & PAYLOAD_MASK)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: beagle <file.bg>");
        std::process::exit(1);
    }
    let path = &args[1];
    let src = std::fs::read_to_string(path).expect("read source file");
    let mut parser = Parser::new(path.clone(), src).expect("create parser");
    let ast = parser.parse().expect("parse");

    let lowered = lower_program(&ast);

    // Install the string pool for the print extern.
    STRINGS.with(|cell| *cell.borrow_mut() = Some(lowered.strings));

    // Build the GC context: beagle object types + continuation types.
    let mut type_table: Vec<TypeInfo> = lowered.type_infos.clone();
    let cont_types = ContinuationTypes::register_into::<Compact>(&mut type_table);

    let heap = SemiSpace::new::<Compact>(2 * 1024 * 1024 * 1024);
    let ctx: GcInterpCtx<Compact, BeaglePtrPolicy> =
        GcInterpCtx::new(heap, type_table, cont_types);
    ctx.set_gc_threshold(1_000_000);

    // ── Build interpreter. ────────────────────────────────────────
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&lowered.module, &ctx);

    if module_has_extern(&lowered.module, "__gc_alloc__") {
        interp.bind_by_name("__gc_alloc__", |args| {
            let type_id = args[0] as usize;
            let varlen_len = args[1] as usize;
            let info = &ctx.type_table()[type_id];
            let ptr = unsafe { dynalloc::alloc_obj::<Compact>(&ctx, info, varlen_len) };
            assert!(!ptr.is_null(), "gc alloc failed (OOM even after collection)");
            ExternCallResult::Value(Some(ptr as u64))
        });
    }

    interp.bind_by_name("beagle_print", |args| {
        print_value(args[0], false);
        ExternCallResult::Value(None)
    });
    interp.bind_by_name("beagle_println", |args| {
        print_value(args[0], true);
        ExternCallResult::Value(None)
    });

    // Minimal stdlib stubs — just enough for `main(args)` to accept
    // that args is nil and fall through to its default branch.
    if module_has_extern(&lowered.module, "beagle_length") {
        interp.bind_by_name("beagle_length", |_args| {
            // nil has length 0 in our stub
            ExternCallResult::Value(Some(encode_f64_int(0)))
        });
    }
    if module_has_extern(&lowered.module, "beagle_get") {
        interp.bind_by_name("beagle_get", |_args| {
            ExternCallResult::Value(Some(nanbox_nil()))
        });
    }
    if module_has_extern(&lowered.module, "beagle_to_number") {
        interp.bind_by_name("beagle_to_number", |args| {
            // Pass through; in a real stdlib we'd parse strings.
            ExternCallResult::Value(Some(args[0]))
        });
    }

    for name in [
        "beagle_add", "beagle_sub", "beagle_mul", "beagle_div",
        "beagle_neg", "beagle_eq", "beagle_lt", "beagle_gt", "beagle_not",
    ] {
        interp.bind_by_name(name, move |_args| {
            panic!(
                "beagle slow-path `{name}` hit — binary_trees subset shouldn't need it",
            );
        });
    }

    // ── Run main with a single nil arg. ──────────────────────────
    // binary_trees declares `fn main(args)`; we pass nil and rely on
    // `length(args) > 0` being false → falls through to `n = 10`.
    // Our lowering doesn't implement `length` yet though, so if main
    // tries to call it we'll panic. For now the benchmark file uses
    // `if length(args) > 0 { to-number(get(args, 0)) } else { 10 }` —
    // we need those externs, or a workaround.
    //
    // Simplest workaround for MVP: bind `length` as a fake extern that
    // always returns 0 (so `length(args) > 0` is false → n = 10).
    // This only works if lowering routes `length(args)` to an extern,
    // which it does NOT — `Call { name: "length", ... }` will panic
    // at lowering as "unknown function". We need to handle that.
    //
    // For now just run and see what happens.
    let nil = nanbox_nil();
    match interp.run(lowered.main, &[nil]) {
        Ok(InterpResult::Value(_)) | Ok(InterpResult::Void) => {}
        Ok(other) => {
            eprintln!("beagle: unexpected result: {:?}", other);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("beagle: runtime error: {:?}", e);
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

fn module_has_extern(module: &dynir::Module, name: &str) -> bool {
    module.func_table.iter().any(|def| match def {
        dynir::FuncDef::Extern(ef) => ef.name == name,
        _ => false,
    })
}

thread_local! {
    static STRINGS: std::cell::RefCell<Option<StringPool>> =
        std::cell::RefCell::new(None);
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
