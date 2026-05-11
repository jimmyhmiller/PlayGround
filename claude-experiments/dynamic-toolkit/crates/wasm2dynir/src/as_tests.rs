use crate::translate_wasm;
use dynalloc::NanBoxPtrPolicy;
use dynir::gc_runtime::GcInterpCtx;
use dynir::interp::*;
use dynir::ir::Module;
use dynir::verify::verify;
use dynobj::Compact;
use dynvalue::NanBox;

fn run_wasm_file_i32(wasm_bytes: &[u8], args: &[u64]) -> i32 {
    let (func, _imports) = translate_wasm(wasm_bytes).expect("failed to translate");
    verify(&func).unwrap_or_else(|errors| {
        eprintln!("IR:\n{}", func);
        panic!("IR verification failed: {:?}", errors);
    });
    let (module, entry) = Module::from_function(func.clone());
    let roots: GcInterpCtx<Compact, NanBoxPtrPolicy> = GcInterpCtx::new_unallocating();
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    match interp.run(entry, args).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("expected Value, got {:?}", other),
    }
}

fn run_wasm_file_i64(wasm_bytes: &[u8], args: &[u64]) -> i64 {
    let (func, _imports) = translate_wasm(wasm_bytes).expect("failed to translate");
    verify(&func).unwrap_or_else(|errors| {
        eprintln!("IR:\n{}", func);
        panic!("IR verification failed: {:?}", errors);
    });
    let (module, entry) = Module::from_function(func.clone());
    let roots: GcInterpCtx<Compact, NanBoxPtrPolicy> = GcInterpCtx::new_unallocating();
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    match interp.run(entry, args).unwrap() {
        InterpResult::Value(v) => v as i64,
        other => panic!("expected Value, got {:?}", other),
    }
}

#[test]
fn as_fib() {
    let wasm = include_bytes!("../as-programs/fib.wasm");
    assert_eq!(run_wasm_file_i32(wasm, &[0]), 0);
    assert_eq!(run_wasm_file_i32(wasm, &[1]), 1);
    assert_eq!(run_wasm_file_i32(wasm, &[10]), 55);
    assert_eq!(run_wasm_file_i32(wasm, &[20]), 6765);
    assert_eq!(run_wasm_file_i32(wasm, &[30]), 832040);
}

#[test]
fn as_collatz() {
    let wasm = include_bytes!("../as-programs/collatz.wasm");
    // Known Collatz step counts
    assert_eq!(run_wasm_file_i32(wasm, &[1]), 0);
    assert_eq!(run_wasm_file_i32(wasm, &[2]), 1);
    assert_eq!(run_wasm_file_i32(wasm, &[3]), 7);
    assert_eq!(run_wasm_file_i32(wasm, &[6]), 8);
    assert_eq!(run_wasm_file_i32(wasm, &[27]), 111);
}

#[test]
fn as_gcd() {
    let wasm = include_bytes!("../as-programs/gcd.wasm");
    assert_eq!(run_wasm_file_i32(wasm, &[12, 8]), 4);
    assert_eq!(run_wasm_file_i32(wasm, &[100, 75]), 25);
    assert_eq!(run_wasm_file_i32(wasm, &[17, 13]), 1);
    assert_eq!(run_wasm_file_i32(wasm, &[0, 5]), 5);
    assert_eq!(run_wasm_file_i32(wasm, &[1000000, 999999]), 1);
}

#[test]
fn as_power() {
    let wasm = include_bytes!("../as-programs/power.wasm");
    assert_eq!(run_wasm_file_i64(wasm, &[2, 0]), 1);
    assert_eq!(run_wasm_file_i64(wasm, &[2, 10]), 1024);
    assert_eq!(run_wasm_file_i64(wasm, &[3, 5]), 243);
    assert_eq!(run_wasm_file_i64(wasm, &[10, 9]), 1_000_000_000);
}

#[test]
fn as_primes() {
    let wasm = include_bytes!("../as-programs/primes.wasm");
    assert_eq!(run_wasm_file_i32(wasm, &[1]), 0);
    assert_eq!(run_wasm_file_i32(wasm, &[10]), 4); // 2, 3, 5, 7
    assert_eq!(run_wasm_file_i32(wasm, &[100]), 25);
    assert_eq!(run_wasm_file_i32(wasm, &[1000]), 168);
}

#[test]
fn print_as_fib_ir() {
    let wasm = include_bytes!("../as-programs/fib.wasm");
    let (func, _) = translate_wasm(wasm).unwrap();
    verify(&func).unwrap();
    println!("=== AssemblyScript fib IR ===\n{}", func);
}
