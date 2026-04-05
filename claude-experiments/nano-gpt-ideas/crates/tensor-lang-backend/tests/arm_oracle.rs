//! Integration tests for the ARM64 native backend.
//! Compares ARM output against the ndarray oracle.

#![cfg(target_arch = "aarch64")]

use std::collections::HashMap;
use ndarray::{array, ArrayD};
use tensor_lang_backend::arm::ArmBackend;
use tensor_lang_backend::arm_runtime::ArmRuntime;
use tensor_lang_graph::compile;
use tensor_lang_test_oracle::{compare, eval_with_inputs};

fn check_arm(program: &str, input_data: HashMap<String, ArrayD<f32>>, atol: f32) {
    let graph = compile(program);

    // Run oracle
    let oracle_results = eval_with_inputs(&graph, &input_data);
    let oracle_output = oracle_results.last().unwrap();

    // Emit ARM code
    let backend = ArmBackend;
    let arm_code = backend.emit_fused(&graph);

    // Compute output size
    let last_node = &graph.nodes[graph.nodes.len() - 1];
    let output_size: usize = last_node
        .shape
        .iter()
        .map(|d| d.as_usize().expect("non-concrete output dim"))
        .product();

    // Collect inputs in graph order
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
    let backend_output = rt.run(&input_slices, output_size);

    compare(oracle_output, &backend_output, atol).unwrap_or_else(|e| {
        panic!("ARM backend mismatch: {e}")
    });
}

// ---------------------------------------------------------------------------
// Tests (mirroring wasm_oracle.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_arm_add() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn(),
    );
    check_arm(
        "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_chain() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0]].into_dyn(),
    );
    check_arm(
        "let x = load([1, 3]) let y = load([1, 3]) let z = neg(add(x, y))",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_reduce_sum() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_arm(
        "let x = load([2, 3]) let s = sum(x, axis: 1)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_reduce_max() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 5.0, 3.0], [6.0, 2.0, 4.0]].into_dyn(),
    );
    check_arm(
        "let x = load([2, 3]) let m = max(x, axis: 1)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_mul() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[2.0f32, 3.0, 4.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0]].into_dyn(),
    );
    check_arm(
        "let x = load([1, 3]) let y = load([1, 3]) let z = mul(x, y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_matmul_small() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn(),
    );
    check_arm(
        "let a = load([2, 3]) let b = load([3, 2]) let c = matmul(a, b)",
        inputs,
        1e-4,
    );
}

#[test]
fn test_arm_exp() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![0.0f32, 1.0, -1.0, 2.0].into_dyn(),
    );
    check_arm(
        "let x = load([4]) let y = exp(x)",
        inputs,
        1e-3,
    );
}

#[test]
fn test_arm_log() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![1.0f32, 2.0, 0.5, 10.0].into_dyn(),
    );
    check_arm(
        "let x = load([4]) let y = log(x)",
        inputs,
        1e-3,
    );
}

#[test]
fn test_arm_sqrt() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![1.0f32, 4.0, 9.0, 16.0].into_dyn(),
    );
    check_arm(
        "let x = load([4]) let y = sqrt(x)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_broadcast_add() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![10.0f32, 20.0, 30.0].into_dyn(),
    );
    check_arm(
        "let x = load([2, 3]) let y = load([3]) let z = add(x, y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_arm_softmax() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn(),
    );
    check_arm(
        "let x = load([2, 3]) \
         let m = max(x, axis: 1) \
         let shifted = sub(x, m) \
         let e = exp(shifted) \
         let s = sum(e, axis: 1) \
         let y = div(e, s)",
        inputs,
        1e-3,
    );
}

#[test]
fn test_arm_matmul_medium() {
    // Just below tiling threshold — uses non-tiled reduce
    let m = 8;
    let k = 8;
    let n = 8;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        ndarray::ArrayD::from_shape_vec(vec![m, k], a_data).unwrap(),
    );
    inputs.insert(
        "input_1".into(),
        ndarray::ArrayD::from_shape_vec(vec![k, n], b_data).unwrap(),
    );
    check_arm(
        &format!("let a = load([{m}, {k}]) let b = load([{k}, {n}]) let c = matmul(a, b)"),
        inputs,
        1e-4,
    );
}

#[test]
fn test_arm_matmul_tiled() {
    // Just above tiling threshold — minimal tiled matmul
    let m = 16;
    let k = 16;
    let n = 16;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        ndarray::ArrayD::from_shape_vec(vec![m, k], a_data).unwrap(),
    );
    inputs.insert(
        "input_1".into(),
        ndarray::ArrayD::from_shape_vec(vec![k, n], b_data).unwrap(),
    );
    check_arm(
        &format!("let a = load([{m}, {k}]) let b = load([{k}, {n}]) let c = matmul(a, b)"),
        inputs,
        0.1, // relaxed tolerance for tiled accumulation order
    );
}
