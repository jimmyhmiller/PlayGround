//! Test the persistent WASM runner protocol end-to-end.
//! This tests the exact same code path that tensor-lang-train uses.

use std::io::{Write, Read};
use std::process::Command;
use std::collections::HashMap;

use ndarray::array;
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_graph::compile;
use tensor_lang_test_oracle::{compare, eval_with_inputs};

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

struct TestRunner {
    child: std::process::Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: std::process::ChildStdout,
    output_size: usize,
}

impl TestRunner {
    fn new(wasm_path: &std::path::Path, output_size: usize) -> Self {
        let root = project_root();
        let runner = root.join("persistent_runner_wasm.mjs");
        let mut child = Command::new("node")
            .args([runner.to_str().unwrap(), wasm_path.to_str().unwrap()])
            .current_dir(&root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to spawn node");

        let stdin = std::io::BufWriter::new(child.stdin.take().unwrap());
        let stdout = child.stdout.take().unwrap();
        TestRunner { child, stdin, stdout, output_size }
    }

    fn run(&mut self, inputs: &[&[f32]]) -> Vec<f32> {
        let n = inputs.len() as u32;
        self.stdin.write_all(&n.to_le_bytes()).unwrap();
        self.stdin.write_all(&(self.output_size as u32).to_le_bytes()).unwrap();
        for arr in inputs {
            let size = arr.len() as u32;
            self.stdin.write_all(&size.to_le_bytes()).unwrap();
            for &v in *arr {
                self.stdin.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        self.stdin.flush().unwrap();

        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header).expect("failed to read output header");
        let n_outputs = u32::from_le_bytes(header) as usize;

        let mut data = vec![0u8; n_outputs * 4];
        self.stdout.read_exact(&mut data).expect("failed to read output data");

        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl Drop for TestRunner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn test_persistent_runner_add() {
    let program = "let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)";
    let graph = compile(program);

    let backend = WasmBackend::default();
    let wasm_bytes = backend.emit_fused(&graph);

    let tmp = std::env::temp_dir();
    let wasm_path = tmp.join("test_persistent_add.wasm");
    std::fs::write(&wasm_path, &wasm_bytes).unwrap();

    let mut runner = TestRunner::new(&wasm_path, 6);

    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    let result = runner.run(&[&x, &y]);
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

    // Run a second time to verify the persistent process handles multiple calls
    let x2 = [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0];
    let y2 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result2 = runner.run(&[&x2, &y2]);
    assert_eq!(result2, vec![101.0, 202.0, 303.0, 404.0, 505.0, 606.0]);

    let _ = std::fs::remove_file(&wasm_path);
}

#[test]
fn test_persistent_runner_softmax() {
    let program = r#"
        fn softmax(x) {
            let m = max(x, axis: 1)
            let e = exp(sub(x, m))
            let s = sum(e, axis: 1)
            mul(recip(s), e)
        }
        let x = load([2, 3])
        let y = softmax(x)
    "#;
    let graph = compile(program);

    // Oracle
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_0".into(),
        array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn(),
    );
    let oracle_results = eval_with_inputs(&graph, &inputs);
    let oracle_output = oracle_results.last().unwrap();

    let backend = WasmBackend::default();
    let wasm_bytes = backend.emit_fused(&graph);

    let tmp = std::env::temp_dir();
    let wasm_path = tmp.join("test_persistent_softmax.wasm");
    std::fs::write(&wasm_path, &wasm_bytes).unwrap();

    let mut runner = TestRunner::new(&wasm_path, 6);

    let x = [1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
    let result = runner.run(&[&x]);

    compare(oracle_output, &result, 1e-3).unwrap_or_else(|e| {
        panic!("Persistent runner softmax mismatch: {e}");
    });

    let _ = std::fs::remove_file(&wasm_path);
}

#[test]
fn test_persistent_runner_multi_output() {
    // Test multi-output (the pattern used by training: loss + gradients)
    let program = r#"
        let x = load([4])
        let y = load([4])
        let z = add(x, y)
        let s = sum(z, axis: 0)
    "#;
    let graph = compile(program);

    let backend = WasmBackend::default();
    // Multi-output: both z and s
    let z_id = tensor_lang_graph::NodeId(graph.nodes.len() - 2); // z = add
    let s_id = tensor_lang_graph::NodeId(graph.nodes.len() - 1); // s = sum
    let wasm_bytes = backend.emit_fused_multi_output(&graph, &[s_id, z_id]);

    let tmp = std::env::temp_dir();
    let wasm_path = tmp.join("test_persistent_multi.wasm");
    std::fs::write(&wasm_path, &wasm_bytes).unwrap();

    // Output = s (1 element) + z (4 elements) = 5
    let mut runner = TestRunner::new(&wasm_path, 5);

    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [10.0f32, 20.0, 30.0, 40.0];
    let result = runner.run(&[&x, &y]);

    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 110.0); // sum of z = 11+22+33+44 = 110
    assert_eq!(&result[1..], &[11.0, 22.0, 33.0, 44.0]); // z elements

    let _ = std::fs::remove_file(&wasm_path);
}
