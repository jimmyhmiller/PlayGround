use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dynir::ir::Module;
use dynir::opt::{self, OptConfig};
use dynvalue::NanBox;
use wasm2dynir::{translate_wasm, translate_wasm_module};

// ─── WAT programs ──────────────────────────────────────────────────

const FIBONACCI_WAT: &str = r#"(module
    (func (export "fib") (param $n i32) (result i32)
        (local $a i32)
        (local $b i32)
        (local $i i32)
        (local $tmp i32)
        i32.const 0
        local.set $a
        i32.const 1
        local.set $b
        i32.const 0
        local.set $i
        block $exit
            loop $loop
                local.get $i
                local.get $n
                i32.ge_s
                br_if $exit
                local.get $a
                local.get $b
                i32.add
                local.set $tmp
                local.get $b
                local.set $a
                local.get $tmp
                local.set $b
                local.get $i
                i32.const 1
                i32.add
                local.set $i
                br $loop
            end
        end
        local.get $a))"#;

const FACTORIAL_WAT: &str = r#"(module
    (func (export "fact") (param $n i32) (result i32)
        (local $result i32)
        i32.const 1
        local.set $result
        block $exit
            loop $loop
                local.get $n
                i32.const 1
                i32.le_s
                br_if $exit
                local.get $result
                local.get $n
                i32.mul
                local.set $result
                local.get $n
                i32.const 1
                i32.sub
                local.set $n
                br $loop
            end
        end
        local.get $result))"#;

const SUM_WAT: &str = r#"(module
    (func (export "sum") (param $n i32) (result i32)
        (local $i i32)
        (local $acc i32)
        i32.const 0
        local.set $i
        i32.const 0
        local.set $acc
        block $exit
            loop $loop
                local.get $i
                local.get $n
                i32.ge_s
                br_if $exit
                local.get $acc
                local.get $i
                i32.add
                local.set $acc
                local.get $i
                i32.const 1
                i32.add
                local.set $i
                br $loop
            end
        end
        local.get $acc))"#;

const RECURSIVE_FACTORIAL_WAT: &str = r#"(module
    (func $fact (param $n i32) (result i32)
        local.get $n
        i32.const 1
        i32.le_s
        if (result i32)
            i32.const 1
        else
            local.get $n
            local.get $n
            i32.const 1
            i32.sub
            call $fact
            i32.mul
        end)
    (func (export "main") (param i32) (result i32)
        local.get 0
        call $fact))"#;

/// Nested loop: matrix-style double loop summing i*j
const NESTED_LOOP_WAT: &str = r#"(module
    (func (export "nested") (param $n i32) (result i32)
        (local $i i32)
        (local $j i32)
        (local $acc i32)
        i32.const 0
        local.set $acc
        i32.const 0
        local.set $i
        block $exit_i
            loop $loop_i
                local.get $i
                local.get $n
                i32.ge_s
                br_if $exit_i
                i32.const 0
                local.set $j
                block $exit_j
                    loop $loop_j
                        local.get $j
                        local.get $n
                        i32.ge_s
                        br_if $exit_j
                        local.get $acc
                        local.get $i
                        local.get $j
                        i32.mul
                        i32.add
                        local.set $acc
                        local.get $j
                        i32.const 1
                        i32.add
                        local.set $j
                        br $loop_j
                    end
                end
                local.get $i
                i32.const 1
                i32.add
                local.set $i
                br $loop_i
            end
        end
        local.get $acc))"#;

// ─── Helpers ───────────────────────────────────────────────────────

fn compile_single(wat: &str, config: Option<&OptConfig>) -> (dynlower::JitModule, dynir::ir::FuncRef) {
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (mut func, _) = translate_wasm(&wasm).expect("translate");
    if let Some(cfg) = config {
        opt::optimize_with(&mut func, cfg);
    }
    let (module, entry) = Module::from_function(func);
    let jit = dynlower::JitModule::compile::<NanBox>(&module, &[]);
    (jit, entry)
}

fn compile_module(wat: &str, config: Option<&OptConfig>) -> (dynlower::JitModule, dynir::ir::FuncRef) {
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (mut module, entry) = translate_wasm_module(&wasm).expect("translate");
    if let Some(cfg) = config {
        for func in &mut module.functions {
            opt::optimize_with(func, cfg);
        }
    }
    let jit = dynlower::JitModule::compile::<NanBox>(&module, &[]);
    (jit, entry)
}

// ─── Benchmarks ────────────────────────────────────────────────────

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib");
    let opt_all = OptConfig::all();

    let (jit_no_opt, entry_no) = compile_single(FIBONACCI_WAT, None);
    let (jit_opt, entry_opt) = compile_single(FIBONACCI_WAT, Some(&opt_all));

    for n in [10u64, 20, 30] {
        group.bench_with_input(BenchmarkId::new("no_opt", n), &n, |b, &n| {
            b.iter(|| black_box(jit_no_opt.call(entry_no, &[n])))
        });
        group.bench_with_input(BenchmarkId::new("opt_all", n), &n, |b, &n| {
            b.iter(|| black_box(jit_opt.call(entry_opt, &[n])))
        });
    }
    group.finish();
}

fn bench_factorial(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorial");
    let opt_all = OptConfig::all();

    let (jit_no_opt, entry_no) = compile_single(FACTORIAL_WAT, None);
    let (jit_opt, entry_opt) = compile_single(FACTORIAL_WAT, Some(&opt_all));

    for n in [10u64, 15, 20] {
        group.bench_with_input(BenchmarkId::new("no_opt", n), &n, |b, &n| {
            b.iter(|| black_box(jit_no_opt.call(entry_no, &[n])))
        });
        group.bench_with_input(BenchmarkId::new("opt_all", n), &n, |b, &n| {
            b.iter(|| black_box(jit_opt.call(entry_opt, &[n])))
        });
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
    let opt_all = OptConfig::all();

    let (jit_no_opt, entry_no) = compile_single(SUM_WAT, None);
    let (jit_opt, entry_opt) = compile_single(SUM_WAT, Some(&opt_all));

    for n in [100u64, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("no_opt", n), &n, |b, &n| {
            b.iter(|| black_box(jit_no_opt.call(entry_no, &[n])))
        });
        group.bench_with_input(BenchmarkId::new("opt_all", n), &n, |b, &n| {
            b.iter(|| black_box(jit_opt.call(entry_opt, &[n])))
        });
    }
    group.finish();
}

fn bench_recursive_factorial(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_fact");
    let opt_all = OptConfig::all();

    let (jit_no_opt, entry_no) = compile_module(RECURSIVE_FACTORIAL_WAT, None);
    let (jit_opt, entry_opt) = compile_module(RECURSIVE_FACTORIAL_WAT, Some(&opt_all));

    for n in [10u64, 15, 20] {
        group.bench_with_input(BenchmarkId::new("no_opt", n), &n, |b, &n| {
            b.iter(|| black_box(jit_no_opt.call(entry_no, &[n])))
        });
        group.bench_with_input(BenchmarkId::new("opt_all", n), &n, |b, &n| {
            b.iter(|| black_box(jit_opt.call(entry_opt, &[n])))
        });
    }
    group.finish();
}

fn bench_nested_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_loop");
    let opt_all = OptConfig::all();

    let (jit_no_opt, entry_no) = compile_single(NESTED_LOOP_WAT, None);
    let (jit_opt, entry_opt) = compile_single(NESTED_LOOP_WAT, Some(&opt_all));

    for n in [10u64, 50, 100] {
        group.bench_with_input(BenchmarkId::new("no_opt", n), &n, |b, &n| {
            b.iter(|| black_box(jit_no_opt.call(entry_no, &[n])))
        });
        group.bench_with_input(BenchmarkId::new("opt_all", n), &n, |b, &n| {
            b.iter(|| black_box(jit_opt.call(entry_opt, &[n])))
        });
    }
    group.finish();
}

fn bench_compile_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_time");
    let opt_all = OptConfig::all();

    let programs: &[(&str, &str)] = &[
        ("fib", FIBONACCI_WAT),
        ("factorial", FACTORIAL_WAT),
        ("sum", SUM_WAT),
        ("nested_loop", NESTED_LOOP_WAT),
    ];

    for (name, wat) in programs {
        group.bench_function(BenchmarkId::new("no_opt", name), |b| {
            b.iter(|| compile_single(black_box(wat), None))
        });
        group.bench_function(BenchmarkId::new("opt_all", name), |b| {
            b.iter(|| compile_single(black_box(wat), Some(&opt_all)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fibonacci,
    bench_factorial,
    bench_sum,
    bench_recursive_factorial,
    bench_nested_loop,
    bench_compile_time,
);
criterion_main!(benches);
