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
    check_arm_inner(program, input_data, atol, false);
}

fn check_arm_verbose(program: &str, input_data: HashMap<String, ArrayD<f32>>, atol: f32) {
    check_arm_inner(program, input_data, atol, true);
}

fn check_arm_inner(program: &str, input_data: HashMap<String, ArrayD<f32>>, atol: f32, verbose: bool) {
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

    if verbose {
        eprintln!("oracle: {:?}", &oracle_output.iter().take(32).collect::<Vec<_>>());
        eprintln!("arm:    {:?}", &backend_output[..32.min(output_size)]);
        for i in 0..output_size {
            let o = oracle_output.as_slice().unwrap()[i];
            let a = backend_output[i];
            if (o - a).abs() > atol {
                eprintln!("MISMATCH at [{}]: oracle={}, arm={}", i, o, a);
                break;
            }
        }
    }

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

#[test]
fn test_arm_vs_wasm_tiny_model() {
    // Run in a thread with a large stack to accommodate the JIT frame
    let result = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64MB stack
        .spawn(test_arm_vs_wasm_inner)
        .unwrap()
        .join();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

fn test_arm_vs_wasm_inner() {
    eprintln!("Starting ARM vs WASM full model test (multi-output)...");
    use tensor_lang_backend::wasm::WasmBackend;
    use tensor_lang_backend::runtime::WasmRuntime;

    // Load the tiny model source
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("tensor-lang-train")
        .join("model_tiny.tensor");
    let source = std::fs::read_to_string(&model_path).unwrap();

    // Same config as viz default
    let mut dims = HashMap::new();
    dims.insert("B".into(), 1usize);
    dims.insert("T".into(), 16);
    dims.insert("V".into(), 16);
    dims.insert("D".into(), 32);
    dims.insert("H".into(), 2);
    dims.insert("S".into(), 16);
    dims.insert("M".into(), 128);

    let mut constants = HashMap::new();
    constants.insert("INV_D".into(), 1.0 / 32.0);
    constants.insert("INV_SQRT_S".into(), 1.0 / (16.0f64).sqrt());
    constants.insert("NEG_EIGHT_OVER_H".into(), -8.0 / 2.0);

    let graph = tensor_lang_graph::compile_with_env(&source, &dims, &constants);

    // Multi-output: all non-input nodes (same as viz)
    let output_ids: Vec<tensor_lang_graph::NodeId> = (0..graph.nodes.len())
        .filter(|&i| !matches!(&graph.nodes[i].op, tensor_lang_graph::Op::Input { .. }))
        .map(tensor_lang_graph::NodeId)
        .collect();
    let output_sizes: Vec<usize> = output_ids.iter()
        .map(|id| {
            graph.nodes[id.0].shape.iter()
                .map(|d| d.as_usize().unwrap())
                .product::<usize>()
                .max(1)
        })
        .collect();
    let total_output: usize = output_sizes.iter().sum();

    // Collect inputs with deterministic data
    let mut flat_inputs: Vec<Vec<f32>> = Vec::new();
    for node in &graph.nodes {
        if let tensor_lang_graph::Op::Input { name } = &node.op {
            let size: usize = node.shape.iter()
                .map(|d| d.as_usize().unwrap())
                .product();
            if name == "input_0" {
                flat_inputs.push((0..size).map(|i| (i % 16) as f32).collect());
            } else {
                flat_inputs.push((0..size).map(|i| ((i as f32) * 0.001 - 0.05).sin() * 0.02).collect());
            }
        }
    }
    let input_slices: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

    // Run WASM multi-output
    let wasm_backend = WasmBackend::default();
    let wasm_bytes = wasm_backend.emit_fused_multi_output(&graph, &output_ids);
    let mut wasm_rt = WasmRuntime::new(&wasm_bytes).unwrap();
    let wasm_output = wasm_rt.run(&input_slices, total_output);

    // Run ARM multi-output
    let arm_backend = ArmBackend;
    let arm_code = arm_backend.emit_fused_multi_output(&graph, &output_ids);
    let mut arm_rt = ArmRuntime::new(&arm_code);
    let arm_output = arm_rt.run(&input_slices, total_output);

    // Compare per-output-node
    let mut offset = 0;
    let mut bad_nodes = 0;
    for (oi, (id, &sz)) in output_ids.iter().zip(output_sizes.iter()).enumerate() {
        let w = &wasm_output[offset..offset + sz];
        let a = &arm_output[offset..offset + sz];
        let max_diff: f32 = w.iter().zip(a.iter())
            .map(|(wv, av)| (wv - av).abs())
            .fold(0.0f32, f32::max);
        let any_nan = a.iter().any(|v| v.is_nan());
        let all_zero = a.iter().all(|v| *v == 0.0);
        if max_diff > 1.0 || any_nan {
            if bad_nodes < 10 {
                eprintln!("BAD node {} (output #{}, size={}): max_diff={} nan={} zero={}",
                    id.0, oi, sz, max_diff, any_nan, all_zero);
                eprintln!("  WASM[0..4]: {:?}", &w[..4.min(sz)]);
                eprintln!("  ARM [0..4]: {:?}", &a[..4.min(sz)]);
            }
            bad_nodes += 1;
        }
        offset += sz;
    }
    eprintln!("{} / {} output nodes are bad", bad_nodes, output_ids.len());
    assert!(bad_nodes == 0, "{bad_nodes} output nodes have large diffs or NaN");
}

/// Test layernorm only
#[test]
fn test_arm_layernorm_only() {
    let program = r#"
        let x = load([1, 4, 8])
        let g = load([8])
        let b = load([8])
        let mean = mul(sum(x, axis: 2), 0.125)
        let xc = sub(x, mean)
        let var = mul(sum(mul(xc, xc), axis: 2), 0.125)
        let std = sqrt(add(var, 0.00001))
        let normed = mul(xc, recip(std))
        let ln = add(mul(normed, g), b)
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 4, 8], (0..32).map(|i| i as f32 * 0.1).collect()).unwrap());
    inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(
        vec![8], vec![1.0; 8]).unwrap());
    inputs.insert("input_2".into(), ndarray::ArrayD::from_shape_vec(
        vec![8], vec![0.0; 8]).unwrap());
    check_arm(program, inputs, 0.01);
}

/// Test 3D × 2D matmul  [1,4,8] × [8,16] → [1,4,16]
#[test]
fn test_arm_matmul_3d_2d() {
    let program = r#"
        let a = load([1, 4, 8])
        let b = load([8, 16])
        let c = matmul(a, b)
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 4, 8], (0..32).map(|i| i as f32 * 0.1).collect()).unwrap());
    inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(
        vec![8, 16], (0..128).map(|i| i as f32 * 0.01).collect()).unwrap());
    check_arm(program, inputs, 0.1);
}

/// Test 2D matmul with N=16 (triggers tiling)
#[test]
fn test_arm_matmul_2d_n16() {
    let program = r#"
        let a = load([4, 8])
        let b = load([8, 16])
        let c = matmul(a, b)
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![4, 8], (0..32).map(|i| i as f32 * 0.1).collect()).unwrap());
    inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(
        vec![8, 16], (0..128).map(|i| i as f32 * 0.01).collect()).unwrap());
    check_arm(program, inputs, 0.1);
}

/// Test 3D × 2D matmul small [1,2,3] × [3,4] → [1,2,4]
#[test]
fn test_arm_matmul_3d_2d_small() {
    let program = r#"
        let a = load([1, 2, 3])
        let b = load([3, 4])
        let c = matmul(a, b)
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(
        vec![3, 4], (0..12).map(|i| i as f32 * 0.1).collect()).unwrap());
    check_arm(program, inputs, 0.01);
}

/// Test layernorm + matmul pattern (simpler than full model)
#[test]
fn test_arm_layernorm_matmul() {
    let program = r#"
        let x = load([1, 4, 8])
        let g = load([8])
        let b = load([8])
        let w = load([8, 16])
        let mean = mul(sum(x, axis: 2), 0.125)
        let xc = sub(x, mean)
        let var = mul(sum(mul(xc, xc), axis: 2), 0.125)
        let std = sqrt(add(var, 0.00001))
        let normed = mul(xc, recip(std))
        let ln = add(mul(normed, g), b)
        let out = matmul(ln, w)
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 4, 8], (0..32).map(|i| i as f32 * 0.1).collect()).unwrap());
    inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(
        vec![8], vec![1.0; 8]).unwrap());
    inputs.insert("input_2".into(), ndarray::ArrayD::from_shape_vec(
        vec![8], vec![0.0; 8]).unwrap());
    inputs.insert("input_3".into(), ndarray::ArrayD::from_shape_vec(
        vec![8, 16], (0..128).map(|i| i as f32 * 0.01).collect()).unwrap());
    check_arm(program, inputs, 0.1);
}

/// Test shrink op (used for Q/K/V splitting)
#[test]
fn test_arm_shrink() {
    let program = r#"
        let x = load([2, 3, 4])
        let y = shrink(x, [[0, 2], [1, 2], [0, 4]])
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![2, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap());
    check_arm(program, inputs, 1e-5);
}

/// Test permute (transpose) op
#[test]
fn test_arm_permute() {
    let program = r#"
        let x = load([2, 3, 4])
        let y = permute(x, [0, 2, 1])
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![2, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap());
    check_arm(program, inputs, 1e-5);
}

/// Test expand + cmplt (one-hot encoding pattern)
#[test]
fn test_arm_one_hot() {
    let program = r#"
        let tokens = load([1, 4])
        let classes = arange(8)
        let cls = reshape(classes, [1, 1, 8])
        let cls_exp = expand(cls, [1, 4, 8])
        let tok_r = reshape(tokens, [1, 4, 1])
        let tok_exp = expand(tok_r, [1, 4, 8])
        let lo = sub(cls_exp, 0.5)
        let hi = add(cls_exp, 0.5)
        let one_hot = mul(cmplt(lo, tok_exp), cmplt(tok_exp, hi))
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 4], vec![0.0, 3.0, 4.0, 1.0]).unwrap());
    check_arm(program, inputs, 1e-5);
}

/// Minimal test for expand broadcast bug
#[test]
fn test_arm_expand_broadcast_read() {
    // This isolates the expand: [1,4,1] -> [1,4,8] read pattern
    let program = r#"
        let x = load([1, 4, 1])
        let y = expand(x, [1, 4, 8])
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 4, 1], vec![10.0, 20.0, 30.0, 40.0]).unwrap());
    check_arm_verbose(program, inputs, 1e-5);
}

/// Test causal mask pattern: sub on expanded arange (no cmplt)
#[test]
fn test_arm_causal_mask() {
    // 2D version first
    let program = r#"
        let row = expand(reshape(arange(4), [4, 1]), [4, 4])
        let col = expand(reshape(arange(4), [1, 4]), [4, 4])
        let mask = cmplt(row, col)
    "#;
    let inputs = HashMap::new();
    check_arm_verbose(program, inputs, 1e-5);
}

/// Test 4D expand: [1,1,4,1] -> [1,1,4,4] (broadcast last dim)
#[test]
fn test_arm_expand_4d_broadcast() {
    let program = r#"
        let x = load([1, 1, 4, 1])
        let y = expand(x, [1, 1, 4, 4])
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 1, 4, 1], vec![10.0, 20.0, 30.0, 40.0]).unwrap());
    check_arm_verbose(program, inputs, 1e-5);
}

/// Test batched matmul with 4D tensors (attention pattern)
#[test]
fn test_arm_batched_matmul_4d() {
    let program = r#"
        let q = load([1, 2, 4, 3])
        let k = load([1, 2, 3, 4])
        let scores = matmul(q, k)
    "#;
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 2, 4, 3], (0..24).map(|i| i as f32 * 0.1).collect()).unwrap());
    inputs.insert("input_1".into(), ndarray::ArrayD::from_shape_vec(
        vec![1, 2, 3, 4], (0..24).map(|i| i as f32 * 0.1).collect()).unwrap());
    check_arm(program, inputs, 0.1);
}
