//! Integration tests for the fused loop-IR backend.
//! Compares fused AssemblyScript output against the ndarray oracle.

use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{array, ArrayD};
use tensor_lang_backend::assemblyscript::AssemblyScriptBackend;
use tensor_lang_graph::compile;
use tensor_lang_test_oracle::{compare, eval_with_inputs};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(5000);

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_assemblyscript(as_code: &str, inputs: &[&[f32]]) -> Vec<f32> {
    let root = project_root();
    let tmp_dir = std::env::temp_dir();
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tid = std::thread::current().id();

    let src_path = tmp_dir.join(format!("tensor_fused_{id}_{tid:?}.ts"));
    let wasm_path = tmp_dir.join(format!("tensor_fused_{id}_{tid:?}.wasm"));
    std::fs::write(&src_path, as_code).unwrap();

    let asc_output = Command::new("npx")
        .args([
            "asc",
            src_path.to_str().unwrap(),
            "--outFile",
            wasm_path.to_str().unwrap(),
            "--exportRuntime",
            "--optimize",
        ])
        .current_dir(&root)
        .output()
        .expect("failed to run asc");

    if !asc_output.status.success() {
        let stderr = String::from_utf8_lossy(&asc_output.stderr);
        let stdout = String::from_utf8_lossy(&asc_output.stdout);
        panic!("asc compilation failed:\nstderr: {stderr}\nstdout: {stdout}\nsource:\n{as_code}");
    }

    let inputs_json: Vec<Vec<f32>> = inputs.iter().map(|s| s.to_vec()).collect();
    let inner: Vec<String> = inputs_json.iter().map(|arr| {
        let elems: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
        format!("[{}]", elems.join(","))
    }).collect();
    let inputs_str = format!("[{}]", inner.join(","));

    let runner = root.join("test_runner.mjs");
    let node_output = Command::new("node")
        .args([
            runner.to_str().unwrap(),
            wasm_path.to_str().unwrap(),
            &inputs_str,
        ])
        .current_dir(&root)
        .output()
        .expect("failed to run node");

    if !node_output.status.success() {
        let stderr = String::from_utf8_lossy(&node_output.stderr);
        panic!("node execution failed:\n{stderr}\nsource:\n{as_code}");
    }

    let stdout = String::from_utf8_lossy(&node_output.stdout);
    let s = stdout.trim().trim_start_matches('[').trim_end_matches(']');
    if s.is_empty() {
        vec![]
    } else {
        s.split(',')
            .map(|v| v.trim().parse::<f32>().unwrap())
            .collect()
    }
}

fn check_fused(program: &str, input_data: HashMap<String, ArrayD<f32>>, atol: f32) {
    let graph = compile(program);

    // Run oracle
    let oracle_results = eval_with_inputs(&graph, &input_data);
    let oracle_output = oracle_results.last().unwrap();

    // Emit fused AssemblyScript
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit_fused(&graph);

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

    let backend_output = run_assemblyscript(&as_code, &input_slices);

    compare(oracle_output, &backend_output, atol).unwrap_or_else(|e| {
        panic!("Fused backend mismatch: {e}\ngenerated code:\n{as_code}")
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_fused_add() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_chain() {
    // neg(add(x, y)) — tests elementwise fusion
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0]].into_dyn(),
    );
    check_fused(
        "let x = load([1, 3]) let y = load([1, 3]) let z = neg(add(x, y))",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_reduce_sum() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 3]) let s = sum(x, axis: 1)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_reduce_max() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 5.0, 3.0], [6.0, 2.0, 4.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 3]) let m = max(x, axis: 1)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_broadcast_reduce() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 3]) let s = sum(x, axis: 1) let y = mul(x, s)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_matmul_2d() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn(),
    );
    check_fused(
        "let a = load([2, 3]) let b = load([3, 2]) let c = matmul(a, b)",
        inputs,
        1e-4,
    );
}

#[test]
fn test_fused_softmax() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn(),
    );
    check_fused(
        r#"
        fn softmax(x) {
            let m = max(x, axis: 1)
            let e = exp(sub(x, m))
            let s = sum(e, axis: 1)
            mul(recip(s), e)
        }
        let x = load([2, 3])
        let y = softmax(x)
        "#,
        inputs,
        1e-4,
    );
}

#[test]
fn test_fused_sub_div() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[10.0f32, 20.0, 30.0]].into_dyn());
    inputs.insert("input_1".into(), array![[2.0f32, 4.0, 5.0]].into_dyn());
    check_fused(
        "let x = load([1, 3]) let y = load([1, 3]) let z = div(sub(x, y), y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_exp_log() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 4.0]].into_dyn());
    check_fused(
        "let x = load([1, 3]) let y = exp(log(x))",
        inputs,
        1e-3,
    );
}

#[test]
fn test_fused_broadcast_scalar() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 3]) let y = add(x, 10.0)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_pad() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 2]) let y = pad(x, [[1, 1], [1, 1]])",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_permute() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_fused(
        "let x = load([2, 3]) let y = permute(x, [1, 0])",
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_reshape_permute_matmul() {
    // Simulates the multi-head attention pattern:
    // reshape to add head dim, permute, then matmul
    let mut inputs = HashMap::new();
    let x_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    inputs.insert(
        "input_0".into(),
        ArrayD::from_shape_vec(vec![1, 3, 8], x_data).unwrap(),
    );
    let w_data: Vec<f32> = (0..64).map(|i| if i / 8 == i % 8 { 1.0 } else { 0.0 }).collect();
    inputs.insert(
        "input_1".into(),
        ArrayD::from_shape_vec(vec![8, 8], w_data).unwrap(),
    );
    check_fused(
        r#"
        let x = load([1, 3, 8])
        let w = load([8, 8])
        let proj = matmul(x, w)
        let reshaped = reshape(proj, [1, 3, 2, 4])
        let permuted = permute(reshaped, [0, 2, 1, 3])
        "#,
        inputs,
        1e-4,
    );
}

#[test]
fn test_fused_attention_scores() {
    // Test Q @ K^T pattern from attention: involves permute + matmul
    // q: [1, 2, 3, 4] (B, heads, T, d_head)
    // k: [1, 2, 3, 4] -> permute to [1, 2, 4, 3] then matmul
    let mut inputs = HashMap::new();
    let q_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();
    inputs.insert(
        "input_0".into(),
        ArrayD::from_shape_vec(vec![1, 2, 3, 4], q_data).unwrap(),
    );
    inputs.insert(
        "input_1".into(),
        ArrayD::from_shape_vec(vec![1, 2, 3, 4], k_data).unwrap(),
    );
    check_fused(
        r#"
        let q = load([1, 2, 3, 4])
        let k = load([1, 2, 3, 4])
        let kt = permute(k, [0, 1, 3, 2])
        let scores = matmul(q, kt)
        "#,
        inputs,
        1e-4,
    );
}

#[test]
fn test_fused_causal_mask() {
    // Causal mask: expand + cmplt with different ndims
    let inputs = HashMap::new();
    check_fused(
        r#"
        let rows = reshape(arange(3), [1, 1, 3, 1])
        let cols = reshape(arange(3), [1, 1, 1, 3])
        let r2 = expand(rows, [1, 2, 3, 3])
        let c2 = expand(cols, [1, 2, 3, 3])
        let mask = cmplt(r2, c2)
        "#,
        inputs,
        1e-5,
    );
}

#[test]
fn test_fused_layernorm() {
    let mut inputs = HashMap::new();
    let x_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let gamma: Vec<f32> = vec![1.0; 8];
    let beta: Vec<f32> = vec![0.0; 8];
    inputs.insert(
        "input_0".into(),
        ArrayD::from_shape_vec(vec![1, 3, 8], x_data).unwrap(),
    );
    inputs.insert(
        "input_1".into(),
        ArrayD::from_shape_vec(vec![8], gamma).unwrap(),
    );
    inputs.insert(
        "input_2".into(),
        ArrayD::from_shape_vec(vec![8], beta).unwrap(),
    );
    check_fused(
        r#"
        fn layernorm(x, gamma, beta) {
            let mean = mul(sum(x, axis: 2), 0.125)
            let xc = sub(x, mean)
            let var = mul(sum(mul(xc, xc), axis: 2), 0.125)
            let std = sqrt(add(var, 0.00001))
            let normed = mul(xc, recip(std))
            add(mul(normed, gamma), beta)
        }
        let x = load([1, 3, 8])
        let gamma = load([8])
        let beta = load([8])
        let out = layernorm(x, gamma, beta)
        "#,
        inputs,
        1e-3,
    );
}

#[test]
fn test_fused_nanogpt_tiny() {
    let batch = 1;
    let seq_len = 3;
    let vocab_size = 4;
    let n_embd = 8;
    let n_head = 2;
    let n_layer = 1;

    let program = tensor_lang_graph::nanogpt::generate_nanogpt_program(
        batch, seq_len, vocab_size, n_embd, n_head, n_layer,
    );
    let graph = compile(&program);

    let mut inputs = HashMap::new();
    for node in &graph.nodes {
        if let tensor_lang_graph::Op::Input { name } = &node.op {
            let shape: Vec<usize> = node.shape.iter().map(|d| d.as_usize().unwrap()).collect();
            let size: usize = shape.iter().product();
            let data: Vec<f32> = if name == "input_0" {
                (0..size).map(|i| (i % vocab_size) as f32).collect()
            } else {
                (0..size).map(|i| (i as f32 * 0.1).sin() * 0.1).collect()
            };
            inputs.insert(name.clone(), ArrayD::from_shape_vec(shape, data).unwrap());
        }
    }

    check_fused(&program, inputs, 1e-2);
}

#[test]
fn test_fused_matmul_non_divisible_tile() {
    // Test matmul with dimensions that don't divide evenly by tile sizes
    // (tm=8, tn=32, tk=32) → M=7, N=37, K=41 are all non-divisible
    let program = r#"
        let a = load([7, 41])
        let b = load([41, 37])
        let c = matmul(a, b)
    "#;

    let graph = compile(program);
    let mut inputs = HashMap::new();
    for node in &graph.nodes {
        if let tensor_lang_graph::Op::Input { name } = &node.op {
            let shape: Vec<usize> = node.shape.iter().map(|d| d.as_usize().unwrap()).collect();
            let size: usize = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
            inputs.insert(name.clone(), ArrayD::from_shape_vec(shape, data).unwrap());
        }
    }
    check_fused(program, inputs, 1e-4);
}
