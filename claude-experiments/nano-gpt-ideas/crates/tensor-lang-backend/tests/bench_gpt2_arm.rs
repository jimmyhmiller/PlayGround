//! Time the full GPT-2 forward pass on the ARM backend at concrete T=16,
//! matching the setup the optimization writeup describes.
//!
//! Reads weights from gpt2_weights/ (run `python3 export_gpt2.py` first).
#![cfg(target_arch = "aarch64")]

use std::time::Instant;

use tensor_lang_backend::arm::ArmBackend;
use tensor_lang_backend::arm_runtime::ArmRuntime;
use tensor_lang_graph::{nanogpt, Op};

#[test]
fn bench_gpt2_arm_concrete_t16() {
    // Use stock GPT-2 124M dims at concrete T=16 (matching the writeup's setup).
    let vocab_size: usize = 50257;
    let n_embd: usize = 768;
    let n_head: usize = 12;
    let n_layer: usize = 12;
    let seq_len: usize = 16;
    eprintln!("Config: vocab={vocab_size} d={n_embd} heads={n_head} layers={n_layer} T={seq_len}");

    let graph = nanogpt::compile_gpt2(1, Some(seq_len), vocab_size, n_embd, n_head, n_layer);
    eprintln!("Graph: {} nodes", graph.nodes.len());

    let input_nodes: Vec<(String, Vec<usize>)> = graph.nodes.iter()
        .filter_map(|n| if let Op::Input { name } = &n.op {
            Some((name.clone(), n.shape.iter().map(|d| d.as_usize().unwrap()).collect()))
        } else { None })
        .collect();
    eprintln!("Inputs: {}", input_nodes.len());

    // Random data of the right shape for each input. Timing doesn't depend on
    // correctness — just need shape-conforming buffers.
    let flat_inputs: Vec<Vec<f32>> = input_nodes.iter().enumerate().map(|(i, (_n, shape))| {
        let total: usize = shape.iter().product();
        // Tokens (i==0) need to be valid indices into wte. Use small ints.
        if i == 0 {
            (0..total).map(|j| (j % 100) as f32).collect()
        } else {
            (0..total).map(|j| ((j as u32).wrapping_mul(2654435761) as f32 / u32::MAX as f32) - 0.5).collect()
        }
    }).collect();

    // Compile to ARM.
    eprintln!("Emitting ARM...");
    let t0 = Instant::now();
    let arm_code = ArmBackend.emit_fused(&graph);
    eprintln!("  emit: {:.2}s, code = {} bytes", t0.elapsed().as_secs_f64(), arm_code.code.len());

    let mut rt = ArmRuntime::new(&arm_code);

    // Determine output size from final node (logits).
    let last = &graph.nodes.last().unwrap().shape;
    let output_size: usize = last.iter().map(|d| d.as_usize().unwrap()).product();

    // Warmup
    let inputs: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();
    for _ in 0..2 { rt.run(&inputs, output_size); }

    // Bench
    let iters = 5;
    let t = Instant::now();
    for _ in 0..iters { let _ = rt.run(&inputs, output_size); }
    let avg_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    eprintln!("ARM forward (T={seq_len}, {n_layer} layers): avg {avg_ms:.1} ms");
}
