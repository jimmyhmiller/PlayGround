//! Test GPU reductions at sizes that match GPT-2 internals.
//! Reproduces precision divergence found in full GPT-2 benchmark.

use tensor_lang_backend::runtime::WasmRuntime;
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_gpu::plan;
use tensor_lang_gpu::runtime::GpuRuntime;
use tensor_lang_graph::compile;

fn compare_gpu_wasm(program: &str, inputs: &[&[f32]], atol: f32) {
    let graph = compile(program);
    let last_node = &graph.nodes[graph.nodes.len() - 1];
    let output_size: usize = last_node
        .shape
        .iter()
        .map(|d| d.as_usize().unwrap())
        .product();

    let wasm_bytes = WasmBackend::default().emit_fused(&graph);
    let mut wasm_rt = WasmRuntime::new(&wasm_bytes).unwrap();
    let wasm_out = wasm_rt.run(inputs, output_size);

    let gpu_plan = plan::build_plan(&graph);
    let gpu_rt = GpuRuntime::new();
    let gpu_out = gpu_rt.run(&gpu_plan, inputs, output_size);

    assert_eq!(wasm_out.len(), gpu_out.len(), "length mismatch");

    let max_diff: f32 = wasm_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mean_diff: f32 = wasm_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / output_size as f32;

    eprintln!(
        "output_size={output_size}, max_diff={max_diff:.8}, mean_diff={mean_diff:.8}, wasm[0]={:.6}, gpu[0]={:.6}",
        wasm_out[0], gpu_out[0]
    );

    assert!(
        max_diff <= atol,
        "max_diff {max_diff} exceeds tolerance {atol}"
    );
}

// --- Reduction size tests ---

#[test]
fn test_reduce_sum_768() {
    let data: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
    compare_gpu_wasm(
        "let x = load([1, 768]) let z = sum(x, axis: 1)",
        &[&data],
        1e-4,
    );
}

#[test]
fn test_reduce_sum_3072() {
    let data: Vec<f32> = (0..3072).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
    compare_gpu_wasm(
        "let x = load([1, 3072]) let z = sum(x, axis: 1)",
        &[&data],
        1e-3,
    );
}

#[test]
fn test_reduce_sum_50257() {
    let data: Vec<f32> = (0..50257)
        .map(|i| (i as f32 * 0.001).sin() * 0.01)
        .collect();
    compare_gpu_wasm(
        "let x = load([1, 50257]) let z = sum(x, axis: 1)",
        &[&data],
        1e-2,
    );
}

// --- Matmul size tests ---

#[test]
fn test_matmul_768x768() {
    let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
    let b: Vec<f32> = (0..768 * 768)
        .map(|i| (i as f32 * 0.007).cos() * 0.02)
        .collect();
    compare_gpu_wasm(
        "let a = load([1, 768]) let b = load([768, 768]) let c = matmul(a, b)",
        &[&a, &b],
        1e-3,
    );
}

#[test]
fn test_matmul_8x768_by_768x50257() {
    // Final lm_head matmul shape from GPT-2 with T=8
    let a: Vec<f32> = (0..8 * 768)
        .map(|i| (i as f32 * 0.003).sin() * 0.1)
        .collect();
    let b: Vec<f32> = (0..768 * 50257)
        .map(|i| (i as f32 * 0.0001).cos() * 0.02)
        .collect();
    compare_gpu_wasm(
        "let a = load([8, 768]) let b = load([768, 50257]) let c = matmul(a, b)",
        &[&a, &b],
        1e-2,
    );
}

// --- Layernorm-like pattern ---

#[test]
fn test_layernorm_768() {
    let x: Vec<f32> = (0..8 * 768)
        .map(|i| (i as f32 * 0.005).sin() * 0.5)
        .collect();
    let gamma: Vec<f32> = (0..768).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let beta: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin() * 0.01).collect();
    let inv_d = 1.0 / 768.0;
    compare_gpu_wasm(
        &format!(
            r#"
            let x = load([1, 8, 768])
            let gamma = load([768])
            let beta = load([768])
            let mean = mul(sum(x, axis: 2), {inv_d})
            let xc = sub(x, mean)
            let var = mul(sum(mul(xc, xc), axis: 2), {inv_d})
            let std = sqrt(add(var, 0.00001))
            let normed = mul(xc, recip(std))
            let z = add(mul(normed, gamma), beta)
            "#
        ),
        &[&x, &gamma, &beta],
        1e-3,
    );
}

// --- Softmax over large dim ---

#[test]
fn test_softmax_768() {
    let x: Vec<f32> = (0..8 * 768)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    compare_gpu_wasm(
        r#"
        let x = load([8, 768])
        let m = max(x, axis: 1)
        let e = exp(sub(x, m))
        let s = sum(e, axis: 1)
        let z = div(e, s)
        "#,
        &[&x],
        1e-4,
    );
}
