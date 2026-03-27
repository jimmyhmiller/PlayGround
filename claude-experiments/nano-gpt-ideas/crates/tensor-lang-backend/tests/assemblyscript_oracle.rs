use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{array, ArrayD};
use tensor_lang_backend::{assemblyscript::AssemblyScriptBackend, Backend};
use tensor_lang_graph::compile;
use tensor_lang_test_oracle::{compare, eval_with_inputs};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The project root where node_modules and test_runner.mjs live.
fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()  // crates/
        .parent().unwrap()  // project root
        .to_path_buf()
}

/// Compile AssemblyScript source to WASM, execute with given inputs,
/// and return the output as a flat Vec<f32>.
fn run_assemblyscript(as_code: &str, inputs: &[&[f32]]) -> Vec<f32> {
    let root = project_root();
    let tmp_dir = std::env::temp_dir();
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tid = std::thread::current().id();

    // Write AS source with unique name per test
    let src_path = tmp_dir.join(format!("tensor_test_{id}_{tid:?}.ts"));
    let wasm_path = tmp_dir.join(format!("tensor_test_{id}_{tid:?}.wasm"));
    std::fs::write(&src_path, as_code).unwrap();

    // Compile with asc
    let asc_output = Command::new("npx")
        .args(["asc", src_path.to_str().unwrap(),
               "--outFile", wasm_path.to_str().unwrap(),
               "--exportRuntime", "--optimize"])
        .current_dir(&root)
        .output()
        .expect("failed to run asc");

    if !asc_output.status.success() {
        let stderr = String::from_utf8_lossy(&asc_output.stderr);
        let stdout = String::from_utf8_lossy(&asc_output.stdout);
        panic!("asc compilation failed:\nstderr: {stderr}\nstdout: {stdout}\nsource:\n{as_code}");
    }

    // Build inputs JSON
    let inputs_json: Vec<Vec<f32>> = inputs.iter().map(|s| s.to_vec()).collect();
    let inputs_str = serde_json_mini(&inputs_json);

    // Run with node
    let runner = root.join("test_runner.mjs");
    let node_output = Command::new("node")
        .args([runner.to_str().unwrap(),
               wasm_path.to_str().unwrap(),
               &inputs_str])
        .current_dir(&root)
        .output()
        .expect("failed to run node");

    if !node_output.status.success() {
        let stderr = String::from_utf8_lossy(&node_output.stderr);
        panic!("node execution failed:\n{stderr}\nsource:\n{as_code}");
    }

    let stdout = String::from_utf8_lossy(&node_output.stdout);
    parse_json_f32_array(stdout.trim())
}

/// Minimal JSON serialization for Vec<Vec<f32>> (no serde dependency needed)
fn serde_json_mini(arrays: &[Vec<f32>]) -> String {
    let inner: Vec<String> = arrays.iter().map(|arr| {
        let elems: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
        format!("[{}]", elems.join(","))
    }).collect();
    format!("[{}]", inner.join(","))
}

/// Parse a JSON array of numbers
fn parse_json_f32_array(s: &str) -> Vec<f32> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    if s.is_empty() { return vec![]; }
    s.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect()
}

/// Helper: compile a program, run oracle and AS backend, compare results.
fn check_against_oracle(program: &str, input_data: HashMap<String, ArrayD<f32>>, atol: f32) {
    let graph = compile(program);

    // Run oracle
    let oracle_results = eval_with_inputs(&graph, &input_data);
    let oracle_output = oracle_results.last().unwrap();

    // Emit AssemblyScript
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit(&graph);

    // Collect inputs in order (matching the function signature)
    let mut ordered_inputs: Vec<(&str, &ArrayD<f32>)> = Vec::new();
    for node in &graph.nodes {
        if let tensor_lang_graph::Op::Input { name } = &node.op {
            ordered_inputs.push((name.as_str(), input_data.get(name.as_str()).unwrap()));
        }
    }
    let flat_inputs: Vec<Vec<f32>> = ordered_inputs.iter()
        .map(|(_, arr)| arr.iter().copied().collect())
        .collect();
    let input_slices: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

    // Run AS backend
    let backend_output = run_assemblyscript(&as_code, &input_slices);

    // Compare
    compare(oracle_output, &backend_output, atol)
        .unwrap_or_else(|e| panic!("AS backend mismatch: {e}\ngenerated code:\n{as_code}"));
}

#[test]
fn test_as_add() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    inputs.insert("input_1".into(), array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_mul() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[2.0f32, 3.0], [4.0, 5.0]].into_dyn());
    inputs.insert("input_1".into(), array![[10.0f32, 10.0], [10.0, 10.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 2]) let y = load([2, 2]) let z = mul(x, y)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_neg() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, -2.0, 3.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = neg(x)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_broadcast_scalar() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let y = add(x, 10.0)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_reduce_sum() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let s = sum(x, axis: 1)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_reduce_max() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 5.0, 3.0], [6.0, 2.0, 4.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let m = max(x, axis: 1)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_broadcast_reduce() {
    // Reduce then broadcast back via mul: mul(x, sum(x, axis:1))
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let s = sum(x, axis: 1) let y = mul(x, s)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_exp() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[0.0f32, 1.0, 2.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = exp(x)",
        inputs, 1e-4, // slightly looser for exp chain
    );
}

#[test]
fn test_as_log() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 4.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = log(x)",
        inputs, 1e-4,
    );
}

#[test]
fn test_as_sqrt() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 4.0, 9.0, 16.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 4]) let y = sqrt(x)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_recip() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[2.0f32, 4.0, 5.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = recip(x)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_binary_max() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 5.0, 3.0], [6.0, 2.0, 4.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let y = max(x, 3.0)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_cmplt() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 5.0, 3.0], [6.0, 2.0, 4.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let y = cmplt(x, 3.5)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_sub() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[10.0f32, 20.0, 30.0]].into_dyn());
    inputs.insert("input_1".into(), array![[1.0f32, 2.0, 3.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = load([1, 3]) let z = sub(x, y)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_div() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[10.0f32, 20.0, 30.0]].into_dyn());
    inputs.insert("input_1".into(), array![[2.0f32, 4.0, 5.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = load([1, 3]) let z = div(x, y)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_binop_sub() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[10.0f32, 20.0, 30.0]].into_dyn());
    inputs.insert("input_1".into(), array![[1.0f32, 2.0, 3.0]].into_dyn());
    check_against_oracle(
        "let x = load([1, 3]) let y = load([1, 3]) let z = x - y",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_reshape() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let y = reshape(x, [3, 2])",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_permute() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let y = permute(x, [1, 0])",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_expand() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32], [2.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 1]) let y = expand(x, [2, 3])",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_reduce_axis0() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 3]) let s = sum(x, axis: 0)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_pad() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn());
    check_against_oracle(
        "let x = load([2, 2]) let y = pad(x, [[1, 1], [1, 1]])",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_shrink() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]].into_dyn());
    check_against_oracle(
        "let x = load([3, 4]) let y = shrink(x, [[0, 2], [1, 3]])",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_arange() {
    let inputs = HashMap::new();
    check_against_oracle(
        "let x = arange(5)",
        inputs, 1e-5,
    );
}

#[test]
fn test_as_arange_causal_mask() {
    // Build a causal mask: cmplt(row_indices, col_indices) where rows expand down, cols expand right
    let inputs = HashMap::new();
    check_against_oracle(
        r#"
        let rows = arange(4)
        let cols = arange(4)
        let r = reshape(rows, [4, 1])
        let c = reshape(cols, [1, 4])
        let r2 = expand(r, [4, 4])
        let c2 = expand(c, [4, 4])
        let mask = cmplt(c2, r2)
        "#,
        inputs, 1e-5,
    );
}

#[test]
fn test_as_matmul_2d() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    inputs.insert("input_1".into(), array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn());
    check_against_oracle(
        "let a = load([2, 3]) let b = load([3, 2]) let c = matmul(a, b)",
        inputs, 1e-4,
    );
}

#[test]
fn test_as_one_hot_embedding() {
    // One-hot embedding: for each token id, create one-hot vector, matmul with embedding table
    // tokens are embedded as float indices in a [B, T] input
    // embedding table is [vocab, d_model]
    // one_hot: cmplt between token ids and arange(vocab) => [B, T, vocab]
    // then matmul one_hot with embedding table
    let mut inputs = HashMap::new();
    // 2 sequences of length 3, vocab size 4, embedding dim 2
    // token ids as floats
    inputs.insert("input_0".into(), array![[0.0f32, 2.0, 1.0], [3.0, 0.0, 2.0]].into_dyn());
    // embedding table [4, 2]
    inputs.insert("input_1".into(), array![
        [1.0f32, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ].into_dyn());
    check_against_oracle(
        r#"
        fn one_hot(indices, n_classes) {
            // indices: [B, T], we want [B, T, n_classes]
            // reshape indices to [B, T, 1], expand to [B, T, n_classes]
            // reshape arange to [1, 1, n_classes], expand to [B, T, n_classes]
            // compare: where indices == class_id, 1.0, else 0.0
            // Use: equal = 1 - abs(cmplt(a,b) + cmplt(b,a))
            // Actually simpler: cmplt(class - 0.5, idx) * cmplt(idx, class + 0.5)
            let classes = arange(4)
            let c = reshape(classes, [1, 1, 4])
            let c2 = expand(c, [2, 3, 4])
            let idx = reshape(indices, [2, 3, 1])
            let idx2 = expand(idx, [2, 3, 4])
            // idx == class  <=>  class - 0.5 < idx AND idx < class + 0.5
            let lo = sub(c2, 0.5)
            let hi = add(c2, 0.5)
            let mask = mul(cmplt(lo, idx2), cmplt(idx2, hi))
            mask
        }
        let tokens = load([2, 3])
        let embed_table = load([4, 2])
        let oh = one_hot(tokens, 4)
        let embedded = matmul(oh, embed_table)
        "#,
        inputs, 1e-4,
    );
}

#[test]
fn test_as_transformer_block() {
    // Minimal transformer: B=1, T=4, d_model=8, n_heads=2, d_head=4
    // Single self-attention block (no MLP for now)
    let mut inputs = HashMap::new();
    // x: [1, 4, 8] - one sequence, 4 tokens, 8-dim embeddings
    let x_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    inputs.insert("input_0".into(),
        ndarray::ArrayD::from_shape_vec(vec![1, 4, 8], x_data).unwrap());
    // Wq, Wk, Wv: [8, 8], Wo: [8, 8]
    // Use simple identity-ish weights for testability
    let eye8: Vec<f32> = (0..64).map(|i| if i / 8 == i % 8 { 1.0 } else { 0.0 }).collect();
    inputs.insert("input_1".into(),
        ndarray::ArrayD::from_shape_vec(vec![8, 8], eye8.clone()).unwrap());
    inputs.insert("input_2".into(),
        ndarray::ArrayD::from_shape_vec(vec![8, 8], eye8.clone()).unwrap());
    inputs.insert("input_3".into(),
        ndarray::ArrayD::from_shape_vec(vec![8, 8], eye8.clone()).unwrap());
    inputs.insert("input_4".into(),
        ndarray::ArrayD::from_shape_vec(vec![8, 8], eye8).unwrap());

    check_against_oracle(
        r#"
        fn softmax(x) {
            let m = max(x, axis: 3)
            let e = exp(sub(x, m))
            let s = sum(e, axis: 3)
            mul(recip(s), e)
        }

        fn attention(x, wq, wk, wv, wo) {
            // x: [B, T, D], wq/wk/wv/wo: [D, D]
            // Project Q, K, V
            let q = matmul(x, wq)
            let k = matmul(x, wk)
            let v = matmul(x, wv)

            // Reshape to multi-head: [B, T, D] -> [B, T, n_heads, d_head] -> [B, n_heads, T, d_head]
            let q2 = permute(reshape(q, [1, 4, 2, 4]), [0, 2, 1, 3])
            let k2 = permute(reshape(k, [1, 4, 2, 4]), [0, 2, 1, 3])
            let v2 = permute(reshape(v, [1, 4, 2, 4]), [0, 2, 1, 3])

            // Attention scores: Q @ K^T / sqrt(d_head)
            let kt = permute(k2, [0, 1, 3, 2])
            let scores = mul(matmul(q2, kt), 0.5)

            // Causal mask: prevent attending to future tokens
            let rows = reshape(arange(4), [1, 1, 4, 1])
            let cols = reshape(arange(4), [1, 1, 1, 4])
            let rows2 = expand(rows, [1, 2, 4, 4])
            let cols2 = expand(cols, [1, 2, 4, 4])
            let causal = cmplt(rows2, cols2)
            let masked = add(scores, mul(causal, neg(expand(reshape(1000000.0, [1, 1, 1, 1]), [1, 2, 4, 4]))))

            // Softmax over last axis
            let attn = softmax(masked)

            // Weighted sum: attn @ V
            let out = matmul(attn, v2)

            // Merge heads: [B, n_heads, T, d_head] -> [B, T, n_heads, d_head] -> [B, T, D]
            let merged = reshape(permute(out, [0, 2, 1, 3]), [1, 4, 8])

            // Output projection
            let projected = matmul(merged, wo)

            // Residual connection
            add(x, projected)
        }

        let x = load([1, 4, 8])
        let wq = load([8, 8])
        let wk = load([8, 8])
        let wv = load([8, 8])
        let wo = load([8, 8])
        let out = attention(x, wq, wk, wv, wo)
        "#,
        inputs, 1e-3,
    );
}

#[test]
fn test_nanogpt_oracle() {
    // Tiny nanoGPT: B=1, T=4, vocab=8, d=16, heads=2, layers=1
    let batch = 1;
    let seq_len = 4;
    let vocab_size = 8;
    let n_embd = 16;
    let n_head = 2;
    let n_layer = 1;

    let program = tensor_lang_graph::nanogpt::generate_nanogpt_program(
        batch, seq_len, vocab_size, n_embd, n_head, n_layer,
    );

    let graph = compile(&program);

    // Build inputs: iterate graph to find Input nodes, generate data
    let mut inputs = HashMap::new();
    for node in &graph.nodes {
        if let tensor_lang_graph::Op::Input { name } = &node.op {
            let shape: Vec<usize> = node.shape.iter().map(|d| d.as_usize().unwrap()).collect();
            let size: usize = shape.iter().product();
            let data: Vec<f32> = if name == "input_0" {
                // Token indices: random ints in [0, vocab_size)
                (0..size).map(|i| (i % vocab_size) as f32).collect()
            } else {
                // Weights: small values to keep things numerically stable
                (0..size).map(|i| (i as f32 * 0.1).sin() * 0.1).collect()
            };
            let arr = ArrayD::from_shape_vec(shape, data).unwrap();
            inputs.insert(name.clone(), arr);
        }
    }

    // Run oracle
    let oracle_results = eval_with_inputs(&graph, &inputs);
    let oracle_output = oracle_results.last().unwrap();

    // Verify output shape
    assert_eq!(oracle_output.shape(), &[batch, seq_len, vocab_size]);

    // Verify it produces finite values (not NaN/Inf)
    assert!(oracle_output.iter().all(|v| v.is_finite()),
        "oracle produced non-finite values");
}

#[test]
fn test_nanogpt_oracle_as() {
    // Even tinier config for AS backend (WASM compilation is slow)
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

    // Build inputs
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
            let arr = ArrayD::from_shape_vec(shape, data).unwrap();
            inputs.insert(name.clone(), arr);
        }
    }

    check_against_oracle(&program, inputs, 1e-2);
}

#[test]
fn test_as_softmax() {
    let mut inputs = HashMap::new();
    inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn());
    check_against_oracle(
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
        inputs, 1e-4,
    );
}
