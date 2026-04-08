//! GPU execution plan: a sequence of buffer allocations and shader dispatches
//! built from the fused loop IR.

use tensor_lang_backend::loop_ir::{self, Stmt};
use tensor_lang_graph::{Dim, Graph, NodeId, Op};

use crate::wgsl::{self, Shader};

pub type BufId = usize;

/// A complete GPU execution plan for a graph.
#[derive(Debug)]
pub struct GpuPlan {
    /// Ordered GPU steps.
    pub steps: Vec<GpuStep>,
    /// Generated WGSL shaders (indexed by `Dispatch::shader_idx`).
    pub shaders: Vec<Shader>,
    /// Symbolic dimension parameter names, sorted.
    pub dim_params: Vec<String>,
    /// Input buffers: (buf_id, input_name).
    pub inputs: Vec<(BufId, String)>,
    /// Output buffers with their sizes. For single-output, this is just the last node.
    /// For multi-output, one entry per requested output.
    pub output_bufs: Vec<(BufId, Dim)>,
}

/// A single step in the GPU execution plan.
#[derive(Debug, Clone)]
pub enum GpuStep {
    /// Allocate a GPU buffer of `size` f32 elements.
    AllocBuffer { buf: BufId, size: Dim },
    /// Fill buffer with a constant value.
    FillConstant {
        buf: BufId,
        value: f32,
        size: Dim,
        shader_idx: usize,
    },
    /// Fill buffer with 0, 1, 2, ..., size-1.
    FillArange {
        buf: BufId,
        size: Dim,
        shader_idx: usize,
    },
    /// Pad: zero-fill + copy with offsets.
    Pad {
        buf: BufId,
        input_buf: BufId,
        output_size: Dim,
        shader_idx: usize,
    },
    /// Dispatch a compute shader.
    Dispatch {
        shader_idx: usize,
        output_buf: BufId,
        output_size: Dim,
    },
    /// Dispatch a tiled matmul shader with 3D workgroup grid.
    DispatchMatmul {
        shader_idx: usize,
        output_buf: BufId,
        output_size: Dim,
        /// Workgroup counts: (ceil(N/TILE_N), ceil(M/TILE_M), batch_size)
        m_size: Dim,
        n_size: Dim,
        batch_size: Dim,
    },
}

/// Build a GPU plan from a graph.
pub fn build_plan(graph: &Graph) -> GpuPlan {
    build_plan_inner(graph, None)
}

/// Build a GPU plan with multiple outputs.
pub fn build_plan_multi_output(graph: &Graph, outputs: &[NodeId]) -> GpuPlan {
    build_plan_inner(graph, Some(outputs))
}

fn build_plan_inner(graph: &Graph, multi_outputs: Option<&[NodeId]>) -> GpuPlan {
    let mut stmts = if let Some(outputs) = multi_outputs {
        loop_ir::lower_with_outputs(graph, outputs)
    } else {
        loop_ir::lower(graph)
    };
    // Note: we skip tile_reduce_loops — that's CPU-specific tiling.
    // Apply unfuse_matmul_bodies to enable tiled matmul for linear projections.
    loop_ir::unfuse_matmul_bodies(&mut stmts);

    // Collect symbolic dim params
    let mut dim_params: Vec<String> = Vec::new();
    for node in &graph.nodes {
        for d in &node.shape {
            collect_params(d, &mut dim_params);
        }
    }
    dim_params.sort();
    dim_params.dedup();

    // Collect inputs
    let inputs: Vec<(BufId, String)> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            if let Op::Input { name } = &n.op {
                Some((i, name.clone()))
            } else {
                None
            }
        })
        .collect();

    let mut steps = Vec::new();
    let mut shaders = Vec::new();

    for stmt in &stmts {
        match stmt {
            Stmt::Alloc { buf, size } => {
                steps.push(GpuStep::AllocBuffer {
                    buf: *buf,
                    size: size.clone(),
                });
            }
            Stmt::Fill { buf, value } => {
                let shader = wgsl::emit_fill_constant_shader(*buf, *value as f32);
                let shader_idx = shaders.len();
                shaders.push(shader);
                steps.push(GpuStep::FillConstant {
                    buf: *buf,
                    value: *value as f32,
                    size: Dim::Lit(1),
                    shader_idx,
                });
            }
            Stmt::FillArange { buf, size } => {
                let shader = wgsl::emit_fill_arange_shader(*buf);
                let shader_idx = shaders.len();
                shaders.push(shader);
                steps.push(GpuStep::FillArange {
                    buf: *buf,
                    size: size.clone(),
                    shader_idx,
                });
            }
            Stmt::Loop { buf, shape, reduce, body, result, .. } => {
                // Try tiled matmul path for reduce loops with matmul pattern.
                // Try tiled matmul for 3-instruction reduce bodies (Load, Load, Mul)
                if let Some(rd) = reduce.as_ref() {
                    if let Some(info) = wgsl::detect_matmul(body, *result, rd, shape) {
                        // Only use tiled matmul for linear projections (B has no batch dims).
                        // Attention matmuls (B has batch dims) need different handling
                        // for correct batch decomposition.
                        let n_ok = info.n_size.as_usize().map_or(true, |v| v >= 32);
                        let k_ok = info.k_size.as_usize().map_or(true, |v| v >= 32);
                        let m_ok = info.effective_m.as_usize().map_or(true, |v| v >= 16);
                        if !info.b_has_batch && n_ok && k_ok && m_ok {
                            let shader = wgsl::emit_tiled_matmul_shader(
                                *buf,
                                shape,
                                rd,
                                &info,
                                &dim_params,
                            );
                            let shader_idx = shaders.len();
                            shaders.push(shader);
                            let output_size = Dim::product(shape);
                            steps.push(GpuStep::DispatchMatmul {
                                shader_idx,
                                output_buf: *buf,
                                output_size,
                                m_size: info.effective_m.clone(),
                                n_size: info.n_size.clone(),
                                batch_size: if info.b_has_batch {
                                    info.batch_count.clone()
                                } else {
                                    Dim::Lit(1)
                                },
                            });
                            continue;
                        }
                    }
                }

                let shader = wgsl::emit_loop_shader(
                    *buf,
                    shape,
                    reduce.as_ref(),
                    body,
                    *result,
                    &dim_params,
                );
                let shader_idx = shaders.len();
                shaders.push(shader);
                let output_size = Dim::product(shape);
                steps.push(GpuStep::Dispatch {
                    shader_idx,
                    output_buf: *buf,
                    output_size,
                });
            }
            Stmt::Pad { buf, input_buf, output_shape, input_shape, padding } => {
                let shader = wgsl::emit_pad_shader(
                    *buf,
                    *input_buf,
                    output_shape,
                    input_shape,
                    padding,
                    &dim_params,
                );
                let shader_idx = shaders.len();
                shaders.push(shader);
                let output_size = Dim::product(output_shape);
                steps.push(GpuStep::Pad {
                    buf: *buf,
                    input_buf: *input_buf,
                    output_size,
                    shader_idx,
                });
            }
        }
    }

    let output_bufs = if let Some(outputs) = multi_outputs {
        outputs
            .iter()
            .map(|id| {
                let size = Dim::product(&graph.nodes[id.0].shape);
                (id.0, size)
            })
            .collect()
    } else {
        let last = graph.nodes.len() - 1;
        let size = Dim::product(&graph.nodes[last].shape);
        vec![(last, size)]
    };

    GpuPlan {
        steps,
        shaders,
        dim_params,
        inputs,
        output_bufs,
    }
}

fn collect_params(dim: &Dim, params: &mut Vec<String>) {
    match dim {
        Dim::Param(name) => {
            if !params.contains(name) {
                params.push(name.clone());
            }
        }
        Dim::Add(a, b) | Dim::Mul(a, b) | Dim::Div(a, b) | Dim::Sub(a, b) => {
            collect_params(a, params);
            collect_params(b, params);
        }
        Dim::Lit(_) => {}
    }
}
