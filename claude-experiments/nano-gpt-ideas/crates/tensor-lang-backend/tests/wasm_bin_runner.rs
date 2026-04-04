//! Test the binary runner (test_runner_wasm_bin.mjs) with dim_params.
//! This tests the same code path used by the CLI.

use std::collections::HashMap;
use std::io::Write;
use std::process::Command;

use ndarray::ArrayD;
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_graph::compile;
use tensor_lang_test_oracle::{compare, eval_with_inputs};

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

#[test]
fn test_bin_runner_symbolic_neg() {
    // Simple symbolic program: neg of a [1, T, 4] tensor
    let program = "dim T\nlet x = load([1, T, 4])\nlet y = neg(x)";
    let graph = compile(program);

    let backend = WasmBackend;
    let wasm_bytes = backend.emit_fused(&graph);

    let tmp = std::env::temp_dir();
    let wasm_path = tmp.join("test_bin_runner.wasm");
    let bin_path = tmp.join("test_bin_runner_inputs.bin");
    let manifest_path = tmp.join("test_bin_runner_manifest.json");
    std::fs::write(&wasm_path, &wasm_bytes).unwrap();

    // Input: T=3, so shape is [1, 3, 4] = 12 elements
    let actual_t = 3usize;
    let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let expected: Vec<f32> = input.iter().map(|v| -v).collect();

    // Write binary inputs
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&bin_path).unwrap());
        for &v in &input {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    // Write manifest with dim_params and output_size
    let manifest = format!(
        r#"{{"dim_params":[{}],"inputs":[{{"n_elements":{}}}],"output_size":{}}}"#,
        actual_t,
        input.len(),
        12 // output is [1, T, 4] = 12
    );
    std::fs::write(&manifest_path, &manifest).unwrap();

    let root = project_root();
    let runner = root.join("test_runner_wasm_bin.mjs");
    let output = Command::new("node")
        .args([
            runner.to_str().unwrap(),
            wasm_path.to_str().unwrap(),
            bin_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
        ])
        .current_dir(&root)
        .output()
        .expect("failed to run node");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("test_runner_wasm_bin.mjs failed:\n{stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let s = stdout.trim().trim_start_matches('[').trim_end_matches(']');
    let result: Vec<f32> = s.split(',')
        .map(|v| v.trim().parse::<f32>().unwrap())
        .collect();

    assert_eq!(result, expected);

    let _ = std::fs::remove_file(&wasm_path);
    let _ = std::fs::remove_file(&bin_path);
    let _ = std::fs::remove_file(&manifest_path);
}
