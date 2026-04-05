//! Integration tests: compile graphs, run on GPU, compare against WASM runtime.

use tensor_lang_backend::runtime::WasmRuntime;
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_gpu::plan;
use tensor_lang_gpu::runtime::GpuRuntime;
use tensor_lang_graph::compile;

/// Compare two f32 slices with absolute tolerance.
fn compare(expected: &[f32], actual: &[f32], atol: f32) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "length mismatch: expected {}, got {}",
        expected.len(),
        actual.len()
    );
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        assert!(
            diff <= atol || (e.is_nan() && a.is_nan()),
            "mismatch at index {i}: expected {e}, got {a} (diff {diff})"
        );
    }
}

/// Compile a program, run it on both WASM and GPU, compare results.
/// `inputs` are in graph order (same order as `load()` calls appear).
fn check_gpu(program: &str, inputs: &[&[f32]], atol: f32) {
    let graph = compile(program);

    // Compute output size
    let last_node = &graph.nodes[graph.nodes.len() - 1];
    let output_size: usize = last_node
        .shape
        .iter()
        .map(|d| d.as_usize().expect("non-concrete output dim"))
        .product();

    // WASM oracle
    let backend = WasmBackend::default();
    let wasm_bytes = backend.emit_fused(&graph);
    let mut wasm_rt = WasmRuntime::new(&wasm_bytes).unwrap();
    let expected = wasm_rt.run(inputs, output_size);

    // GPU
    let gpu_plan = plan::build_plan(&graph);

    // Debug: print shader sources
    for (i, shader) in gpu_plan.shaders.iter().enumerate() {
        eprintln!("=== Shader {i} ===\n{}", shader.source);
    }

    let gpu_rt = GpuRuntime::new();
    let actual = gpu_rt.run(&gpu_plan, inputs, output_size);

    compare(&expected, &actual, atol);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_add() {
    check_gpu(
        "let x = load([4]) let y = load([4]) let z = add(x, y)",
        &[&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_mul() {
    check_gpu(
        "let x = load([3]) let y = load([3]) let z = mul(x, y)",
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_neg() {
    check_gpu(
        "let x = load([3]) let z = neg(x)",
        &[&[1.0, -2.0, 3.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_fused_chain() {
    check_gpu(
        "let x = load([4]) let y = load([4]) let z = neg(add(x, y))",
        &[&[1.0, 2.0, 3.0, 4.0], &[0.5, 0.5, 0.5, 0.5]],
        1e-6,
    );
}

#[test]
fn test_gpu_exp2() {
    check_gpu(
        "let x = load([4]) let z = exp2(x)",
        &[&[0.0, 1.0, 2.0, -1.0]],
        1e-5,
    );
}

#[test]
fn test_gpu_log2() {
    check_gpu(
        "let x = load([4]) let z = log2(x)",
        &[&[1.0, 2.0, 4.0, 8.0]],
        1e-5,
    );
}

#[test]
fn test_gpu_sqrt() {
    check_gpu(
        "let x = load([4]) let z = sqrt(x)",
        &[&[1.0, 4.0, 9.0, 16.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_recip() {
    check_gpu(
        "let x = load([4]) let z = recip(x)",
        &[&[1.0, 2.0, 4.0, 5.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_reduce_sum() {
    check_gpu(
        "let x = load([2, 3]) let z = sum(x, axis: 1)",
        &[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        1e-5,
    );
}

#[test]
fn test_gpu_reduce_max() {
    check_gpu(
        "let x = load([2, 3]) let z = max(x, axis: 1)",
        &[&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_broadcast_add() {
    check_gpu(
        "let x = load([2, 3]) let y = load([3]) let z = add(x, y)",
        &[
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[10.0, 20.0, 30.0],
        ],
        1e-6,
    );
}

#[test]
fn test_gpu_matmul() {
    // [2,3] @ [3,4] -> [2,4]
    check_gpu(
        "let a = load([2, 3]) let b = load([3, 4]) let c = matmul(a, b)",
        &[
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ],
        1e-4,
    );
}

#[test]
fn test_gpu_constant() {
    check_gpu(
        "let x = load([3]) let z = add(x, 10.0)",
        &[&[1.0, 2.0, 3.0]],
        1e-6,
    );
}

#[test]
fn test_gpu_2d_elementwise() {
    check_gpu(
        "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)",
        &[
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ],
        1e-6,
    );
}

#[test]
fn test_gpu_softmax_like() {
    check_gpu(
        r#"
        let x = load([1, 4])
        let m = max(x, axis: 1)
        let e = exp(sub(x, m))
        let s = sum(e, axis: 1)
        let z = div(e, s)
        "#,
        &[&[1.0, 2.0, 3.0, 4.0]],
        1e-4,
    );
}
