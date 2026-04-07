#![cfg(target_arch = "aarch64")]
use std::time::Instant;
use tensor_lang_backend::arm::ArmBackend;
use tensor_lang_backend::arm_runtime::ArmRuntime;
use tensor_lang_graph::compile;

fn bench_matmul(m: usize, k: usize, n: usize) {
    let prog = format!("let a = load([{m}, {k}]) let b = load([{k}, {n}]) let c = matmul(a, b)");
    let graph = compile(&prog);
    let arm_code = ArmBackend.emit_fused(&graph);
    let mut rt = ArmRuntime::new(&arm_code);
    let a: Vec<f32> = (0..m*k).map(|i| (i % 100) as f32 / 100.0).collect();
    let b: Vec<f32> = (0..k*n).map(|i| (i % 100) as f32 / 100.0).collect();
    let inputs: Vec<&[f32]> = vec![&a, &b];
    for _ in 0..3 { rt.run(&inputs, m*n); }
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters { rt.run(&inputs, m*n); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gflops = 2.0 * m as f64 * k as f64 * n as f64 / ms / 1e6;
    eprintln!("[{m}x{k}]x[{k}x{n}]: {ms:.2} ms ({gflops:.1} GFLOP/s)");
}

#[test]
fn bench_matmul_gflops() {
    bench_matmul(16, 768, 2304);
    bench_matmul(16, 768, 768);
    bench_matmul(16, 768, 3072);
    bench_matmul(64, 768, 768);
}
