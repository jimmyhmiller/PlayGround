//! Benchmark: measures compile time (including register allocation)
//! and execution time for ARM64 backend test cases.
//!
//! Run with: cargo test -p tensor-lang-backend --test bench_regalloc --release -- --nocapture

#![cfg(target_arch = "aarch64")]

use std::collections::HashMap;
use std::time::Instant;
use ndarray::ArrayD;
use tensor_lang_backend::arm::ArmBackend;
use tensor_lang_backend::arm_runtime::ArmRuntime;
use tensor_lang_graph::compile;

struct BenchResult {
    name: &'static str,
    compile_us: f64,
    exec_us: f64,
}

fn bench_case(
    name: &'static str,
    program: &str,
    input_data: &HashMap<String, ArrayD<f32>>,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let graph = compile(program);
    let backend = ArmBackend;

    // Measure compile time (includes register allocation).
    let mut compile_times = Vec::with_capacity(iters);
    for _ in 0..warmup {
        let _ = backend.emit_fused(&graph);
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = backend.emit_fused(&graph);
        compile_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    // Emit once for execution benchmark.
    let arm_code = backend.emit_fused(&graph);
    let last_node = &graph.nodes[graph.nodes.len() - 1];
    let output_size: usize = last_node
        .shape
        .iter()
        .map(|d| d.as_usize().expect("non-concrete dim"))
        .product();

    let mut ordered_inputs: Vec<(&str, &ArrayD<f32>)> = Vec::new();
    for node in &graph.nodes {
        if let tensor_lang_graph::Op::Input { name } = &node.op {
            ordered_inputs.push((name.as_str(), input_data.get(name.as_str()).unwrap()));
        }
    }
    let flat_inputs: Vec<Vec<f32>> = ordered_inputs
        .iter()
        .map(|(_, arr)| arr.iter().copied().collect())
        .collect();
    let input_slices: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

    let mut rt = ArmRuntime::new(&arm_code);

    // Warmup execution.
    for _ in 0..warmup {
        let _ = rt.run(&input_slices, output_size);
    }

    // Measure execution time.
    let mut exec_times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = rt.run(&input_slices, output_size);
        exec_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }

    compile_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    exec_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use median.
    let compile_median = compile_times[compile_times.len() / 2];
    let exec_median = exec_times[exec_times.len() / 2];

    BenchResult {
        name,
        compile_us: compile_median,
        exec_us: exec_median,
    }
}

#[test]
fn bench_arm_regalloc() {
    let warmup = 5;
    let iters = 50;

    let mut results = Vec::new();

    // --- add ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(vec![2, 3], vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap());
        results.push(bench_case("add [2,3]", "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)", &inputs, warmup, iters));
    }

    // --- chain ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![1, 3], vec![1.0f32, 2.0, 3.0]).unwrap());
        inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(vec![1, 3], vec![10.0f32, 20.0, 30.0]).unwrap());
        results.push(bench_case("neg(add) [1,3]", "let x = load([1, 3]) let y = load([1, 3]) let z = neg(add(x, y))", &inputs, warmup, iters));
    }

    // --- reduce_sum ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        results.push(bench_case("sum [2,3]", "let x = load([2, 3]) let s = sum(x, axis: 1)", &inputs, warmup, iters));
    }

    // --- exp ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![4], vec![0.0f32, 1.0, -1.0, 2.0]).unwrap());
        results.push(bench_case("exp [4]", "let x = load([4]) let y = exp(x)", &inputs, warmup, iters));
    }

    // --- log ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![4], vec![1.0f32, 2.0, 0.5, 10.0]).unwrap());
        results.push(bench_case("log [4]", "let x = load([4]) let y = log(x)", &inputs, warmup, iters));
    }

    // --- softmax ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap());
        results.push(bench_case("softmax [2,3]",
            "let x = load([2, 3]) let m = max(x, axis: 1) let shifted = sub(x, m) let e = exp(shifted) let s = sum(e, axis: 1) let y = div(e, s)",
            &inputs, warmup, iters));
    }

    // --- matmul small ---
    {
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(vec![3, 2], vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap());
        results.push(bench_case("matmul [2,3]x[3,2]", "let a = load([2, 3]) let b = load([3, 2]) let c = matmul(a, b)", &inputs, warmup, iters));
    }

    // --- matmul medium ---
    {
        let m = 8; let k = 8; let n = 8;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.01).collect();
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![m, k], a_data).unwrap());
        inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(vec![k, n], b_data).unwrap());
        results.push(bench_case("matmul [8,8]x[8,8]", "let a = load([8, 8]) let b = load([8, 8]) let c = matmul(a, b)", &inputs, warmup, iters));
    }

    // --- matmul tiled ---
    {
        let m = 16; let k = 16; let n = 16;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.01).collect();
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![m, k], a_data).unwrap());
        inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(vec![k, n], b_data).unwrap());
        results.push(bench_case("matmul [16,16]x[16,16]", "let a = load([16, 16]) let b = load([16, 16]) let c = matmul(a, b)", &inputs, warmup, iters));
    }

    // --- matmul larger ---
    {
        let m = 64; let k = 64; let n = 64;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.001).collect();
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(vec![m, k], a_data).unwrap());
        inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(vec![k, n], b_data).unwrap());
        results.push(bench_case("matmul [64,64]x[64,64]", "let a = load([64, 64]) let b = load([64, 64]) let c = matmul(a, b)", &inputs, warmup, iters));
    }

    // Print results.
    eprintln!();
    eprintln!("{:<30} {:>12} {:>12}", "Test", "Compile(μs)", "Exec(μs)");
    eprintln!("{}", "-".repeat(56));
    for r in &results {
        eprintln!("{:<30} {:>12.1} {:>12.1}", r.name, r.compile_us, r.exec_us);
    }
    eprintln!();
}
