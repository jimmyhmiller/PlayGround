//! Integration tests for the direct WASM backend.
//! Compares WASM output against the ndarray oracle.

use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{array, ArrayD};
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_graph::compile;
use tensor_lang_test_oracle::{compare, eval_with_inputs};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(8000);

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_wasm(wasm_bytes: &[u8], inputs: &[&[f32]], output_size: usize) -> Vec<f32> {
    let root = project_root();
    let tmp_dir = std::env::temp_dir();
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tid = std::thread::current().id();

    let wasm_path = tmp_dir.join(format!("tensor_wasm_{id}_{tid:?}.wasm"));
    std::fs::write(&wasm_path, wasm_bytes).unwrap();

    // Build inputs JSON: [[...], [...], ..., output_size]
    let inputs_json: Vec<Vec<f32>> = inputs.iter().map(|s| s.to_vec()).collect();
    let inner: Vec<String> = inputs_json.iter().map(|arr| {
        let elems: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
        format!("[{}]", elems.join(","))
    }).collect();
    let inputs_str = if inner.is_empty() {
        format!("[{}]", output_size)
    } else {
        format!("[{},{}]", inner.join(","), output_size)
    };

    let runner = root.join("test_runner_wasm.mjs");
    let node_output = Command::new("node")
        .args([
            runner.to_str().unwrap(),
            wasm_path.to_str().unwrap(),
            &inputs_str,
        ])
        .current_dir(&root)
        .output()
        .expect("failed to run node");

    let _ = std::fs::remove_file(&wasm_path);

    if !node_output.status.success() {
        let stderr = String::from_utf8_lossy(&node_output.stderr);
        panic!("WASM execution failed:\n{stderr}");
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

fn check_wasm(program: &str, input_data: HashMap<String, ArrayD<f32>>, atol: f32) {
    let graph = compile(program);
    check_wasm_graph(&graph, input_data, atol);
}

fn check_wasm_graph(graph: &tensor_lang_graph::Graph, input_data: HashMap<String, ArrayD<f32>>, atol: f32) {
    // Run oracle
    let oracle_results = eval_with_inputs(graph, &input_data);
    let oracle_output = oracle_results.last().unwrap();

    // Emit WASM
    let backend = WasmBackend::default();
    let wasm_bytes = backend.emit_fused(graph);

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

    let backend_output = run_wasm(&wasm_bytes, &input_slices, output_size);

    compare(oracle_output, &backend_output, atol).unwrap_or_else(|e| {
        panic!("WASM backend mismatch: {e}")
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_wasm_add() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_chain() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[10.0f32, 20.0, 30.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([1, 3]) let y = load([1, 3]) let z = neg(add(x, y))",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_reduce_sum() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 3]) let s = sum(x, axis: 1)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_reduce_max() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 5.0, 3.0], [6.0, 2.0, 4.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 3]) let m = max(x, axis: 1)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_broadcast_reduce() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 3]) let s = sum(x, axis: 1) let y = mul(x, s)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_matmul_2d() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    inputs.insert(
        "input_1".into(),
        array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn(),
    );
    check_wasm(
        "let a = load([2, 3]) let b = load([3, 2]) let c = matmul(a, b)",
        inputs,
        1e-4,
    );
}

#[test]
fn test_wasm_softmax() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn(),
    );
    check_wasm(
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
        1e-3,
    );
}

#[test]
fn test_wasm_sub_div() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[10.0f32, 20.0, 30.0]].into_dyn());
    inputs.insert("input_1".into(), array![[2.0f32, 4.0, 5.0]].into_dyn());
    check_wasm(
        "let x = load([1, 3]) let y = load([1, 3]) let z = div(sub(x, y), y)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_exp_log() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 4.0]].into_dyn());
    check_wasm(
        "let x = load([1, 3]) let y = exp(log(x))",
        inputs,
        1e-2, // polynomial approx for exp2/log2 has more error
    );
}

#[test]
fn test_wasm_log2_range() {
    // log2 across a range of values — tests the polynomial approximation
    let data: Vec<f32> = (1..=20).map(|i| i as f32 * 0.2).collect();
    let arr = ArrayD::from_shape_vec(vec![1, 20], data).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), arr);
    check_wasm(
        "let x = load([1, 20]) let y = log2(x)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_exp2_range() {
    // exp2 across a range including negative and positive values
    let data: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
    let arr = ArrayD::from_shape_vec(vec![1, 21], data).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), arr);
    check_wasm(
        "let x = load([1, 21]) let y = exp2(x)",
        inputs,
        1e-2,
    );
}

#[test]
fn test_wasm_exp2_fractional() {
    // exp2 with fractional inputs — the polynomial approximation is used here
    let data: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.3).collect();
    let arr = ArrayD::from_shape_vec(vec![1, 41], data).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), arr);
    check_wasm(
        "let x = load([1, 41]) let y = exp2(x)",
        inputs,
        1e-2,
    );
}

#[test]
fn test_wasm_broadcast_scalar() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 3]) let y = add(x, 10.0)",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_pad() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 2]) let y = pad(x, [[1, 1], [1, 1]])",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_permute() {
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    );
    check_wasm(
        "let x = load([2, 3]) let y = permute(x, [1, 0])",
        inputs,
        1e-5,
    );
}

#[test]
fn test_wasm_reshape_permute_matmul() {
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
    check_wasm(
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
fn test_wasm_attention_scores() {
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
    check_wasm(
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
fn test_wasm_causal_mask() {
    let inputs = HashMap::new();
    check_wasm(
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
fn test_wasm_layernorm() {
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
    check_wasm(
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
        1e-2,
    );
}

#[test]
fn test_wasm_nanogpt_tiny() {
    let batch = 1;
    let seq_len = 3;
    let vocab_size = 4;
    let n_embd = 8;
    let n_head = 2;
    let n_layer = 1;

    let graph = tensor_lang_graph::nanogpt::compile_gpt2(
        batch, Some(seq_len), vocab_size, n_embd, n_head, n_layer,
    );

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

    check_wasm_graph(&graph, inputs, 1e-2);
}

#[test]
fn test_wasm_matmul_non_divisible_tile() {
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
    check_wasm(program, inputs, 1e-4);
}
