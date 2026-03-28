use std::collections::HashMap;

use tensor_lang_graph::{Dim, Graph, Op};
use crate::Backend;
use crate::loop_ir::{self, Stmt, Inst, Index, ReduceOp};

type BufNames = HashMap<usize, String>;

const KERNEL_OUT_BUF: usize = usize::MAX;

pub struct AssemblyScriptBackend;

/// Collect all Param names from a Dim expression.
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

fn dim_shape_to_usize(shape: &[Dim]) -> Vec<usize> {
    shape.iter().map(|d| d.as_usize().expect("symbolic dim not yet supported in AS backend")).collect()
}

fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() { 1 } else { shape.iter().product() }
}

fn format_f64(v: f64) -> String {
    if v == v.floor() && v.is_finite() {
        format!("{v}.0")
    } else {
        format!("{v}")
    }
}

/// Compute strides for a shape (row-major).
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

impl Backend for AssemblyScriptBackend {
    fn emit(&self, graph: &Graph) -> String {
        let mut out = String::new();

        // Export the Float32Array type ID so the host can create arrays
        out.push_str("export const Float32Array_ID = idof<Float32Array>();\n\n");

        // Function signature
        out.push_str("export function execute(");

        let inputs: Vec<_> = graph.nodes.iter().enumerate()
            .filter_map(|(i, n)| {
                if let Op::Input { name } = &n.op {
                    Some((i, name.clone()))
                } else {
                    None
                }
            })
            .collect();

        for (idx, (_i, name)) in inputs.iter().enumerate() {
            if idx > 0 { out.push_str(", "); }
            out.push_str(&format!("{name}: Float32Array"));
        }
        out.push_str("): Float32Array {\n");

        // Allocate buffers using known shapes
        for (i, node) in graph.nodes.iter().enumerate() {
            match &node.op {
                Op::Input { name } => {
                    out.push_str(&format!("  const buf{i} = {name};\n"));
                }
                Op::Constant(v) => {
                    out.push_str(&format!(
                        "  const buf{i} = new Float32Array(1);\n  buf{i}[0] = f32({});\n",
                        format_f64(*v)
                    ));
                }
                Op::Arange { size } => {
                    let sz = size.as_usize().expect("symbolic dim not yet supported in AS backend");
                    out.push_str(&format!(
                        "  const buf{i} = new Float32Array({sz});\n  for (let i: i32 = 0; i < {sz}; i++) buf{i}[i] = f32(i);\n"
                    ));
                }
                _ => {
                    let size = shape_size(&dim_shape_to_usize(&node.shape));
                    out.push_str(&format!("  const buf{i} = new Float32Array({size});\n"));
                }
            }
        }

        out.push_str("\n");

        // Execute each node
        for (i, node) in graph.nodes.iter().enumerate() {
            let usize_shape = dim_shape_to_usize(&node.shape);
            let size = shape_size(&usize_shape);
            match &node.op {
                Op::Input { .. } | Op::Constant(_) | Op::Arange { .. } => {}

                // Unary elementwise
                Op::Neg => {
                    let a = node.inputs[0].0;
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = -buf{a}[i];\n"
                    ));
                }
                Op::Recip => {
                    let a = node.inputs[0].0;
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(1.0) / buf{a}[i];\n"
                    ));
                }
                Op::Exp2 => {
                    let a = node.inputs[0].0;
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(Math.pow(2.0, f64(buf{a}[i])));\n"
                    ));
                }
                Op::Log2 => {
                    let a = node.inputs[0].0;
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(Math.log2(f64(buf{a}[i])));\n"
                    ));
                }
                Op::Sqrt => {
                    let a = node.inputs[0].0;
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(Math.sqrt(f64(buf{a}[i])));\n"
                    ));
                }

                // Binary elementwise with broadcasting
                Op::Add | Op::Mul | Op::Max | Op::CmpLt => {
                    let a = node.inputs[0].0;
                    let b = node.inputs[1].0;
                    let a_shape = dim_shape_to_usize(&graph.nodes[a].shape);
                    let b_shape = dim_shape_to_usize(&graph.nodes[b].shape);

                    let op_expr = match &node.op {
                        Op::Add => "va + vb",
                        Op::Mul => "va * vb",
                        Op::Max => "va > vb ? va : vb",
                        Op::CmpLt => "va < vb ? f32(1.0) : f32(0.0)",
                        _ => unreachable!(),
                    };

                    let a_idx = broadcast_index_expr("idx", &usize_shape, &a_shape);
                    let b_idx = broadcast_index_expr("idx", &usize_shape, &b_shape);

                    out.push_str(&format!("  for (let idx: i32 = 0; idx < {size}; idx++) {{\n"));
                    out.push_str(&format!("    const va: f32 = buf{a}[{a_idx}];\n"));
                    out.push_str(&format!("    const vb: f32 = buf{b}[{b_idx}];\n"));
                    out.push_str(&format!("    buf{i}[idx] = {op_expr};\n"));
                    out.push_str("  }\n");
                }

                // Reduce with keepdim
                Op::ReduceSum { axis } | Op::ReduceMax { axis } => {
                    let a = node.inputs[0].0;
                    let a_shape = dim_shape_to_usize(&graph.nodes[a].shape);
                    let (init, combine) = match &node.op {
                        Op::ReduceSum { .. } => ("f32(0.0)", "acc + val"),
                        Op::ReduceMax { .. } => ("f32(-Infinity)", "val > acc ? val : acc"),
                        _ => unreachable!(),
                    };

                    let axis = *axis;
                    let a_strides = strides(&a_shape);
                    let axis_size = a_shape[axis];
                    let out_strides = strides(&usize_shape);

                    // For each output element, iterate over the reduced axis
                    out.push_str(&format!("  // reduce axis={axis} ({a_shape:?} -> {usize_shape:?})\n"));
                    out.push_str(&format!("  for (let oi: i32 = 0; oi < {size}; oi++) {{\n"));
                    out.push_str(&format!("    let acc: f32 = {init};\n"));

                    // Compute the base index in the input from the output index
                    // We need to map output flat index -> multi-dim coords -> input base index
                    // Then sweep over the axis dimension
                    let ndim = usize_shape.len();
                    // Decompose oi into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (oi / {}) % {};\n",
                                out_strides[d], usize_shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = oi % {};\n",
                                usize_shape[d]
                            ));
                        }
                    }

                    // Compute input base index (with axis dimension = 0)
                    let mut base_parts = Vec::new();
                    for d in 0..ndim {
                        if d == axis {
                            // This dimension will be iterated
                            continue;
                        }
                        base_parts.push(format!("d{d} * {}", a_strides[d]));
                    }
                    let base_expr = if base_parts.is_empty() {
                        "0".to_string()
                    } else {
                        base_parts.join(" + ")
                    };

                    out.push_str(&format!("    const base: i32 = {base_expr};\n"));
                    out.push_str(&format!("    for (let k: i32 = 0; k < {axis_size}; k++) {{\n"));
                    out.push_str(&format!("      const val: f32 = buf{a}[base + k * {}];\n", a_strides[axis]));
                    out.push_str(&format!("      acc = {combine};\n"));
                    out.push_str("    }\n");
                    out.push_str(&format!("    buf{i}[oi] = acc;\n"));
                    out.push_str("  }\n");
                }

                // Movement
                Op::Reshape { .. } => {
                    let a = node.inputs[0].0;
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = buf{a}[i];\n"
                    ));
                }
                Op::Permute { order } => {
                    let a = node.inputs[0].0;
                    let a_shape = dim_shape_to_usize(&graph.nodes[a].shape);
                    let a_strides = strides(&a_shape);
                    let out_strides = strides(&usize_shape);
                    let ndim = usize_shape.len();

                    out.push_str(&format!("  // permute {order:?}\n"));
                    out.push_str(&format!("  for (let oi: i32 = 0; oi < {size}; oi++) {{\n"));

                    // Decompose output index into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (oi / {}) % {};\n",
                                out_strides[d], usize_shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = oi % {};\n",
                                usize_shape[d]
                            ));
                        }
                    }

                    // Map back to input index: output dim d corresponds to input dim order[d]
                    let mut input_parts = Vec::new();
                    for d in 0..ndim {
                        input_parts.push(format!("d{d} * {}", a_strides[order[d]]));
                    }
                    let input_expr = input_parts.join(" + ");

                    out.push_str(&format!("    buf{i}[oi] = buf{a}[{input_expr}];\n"));
                    out.push_str("  }\n");
                }
                Op::Expand { .. } => {
                    let a = node.inputs[0].0;
                    let a_shape = dim_shape_to_usize(&graph.nodes[a].shape);
                    let a_idx = broadcast_index_expr("i", &usize_shape, &a_shape);
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = buf{a}[{a_idx}];\n"
                    ));
                }
                Op::Pad { padding } => {
                    let a = node.inputs[0].0;
                    let a_shape = dim_shape_to_usize(&graph.nodes[a].shape);
                    let a_strides = strides(&a_shape);
                    let out_strides = strides(&usize_shape);
                    let ndim = usize_shape.len();

                    // Zero-fill then copy from input
                    out.push_str(&format!("  // pad {padding:?}\n"));
                    out.push_str(&format!("  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(0.0);\n"));
                    let a_size = shape_size(&a_shape);
                    out.push_str(&format!("  for (let ai: i32 = 0; ai < {a_size}; ai++) {{\n"));

                    // Decompose input index into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (ai / {}) % {};\n",
                                a_strides[d], a_shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = ai % {};\n",
                                a_shape[d]
                            ));
                        }
                    }

                    // Compute output index by adding padding offsets
                    let mut out_parts = Vec::new();
                    for d in 0..ndim {
                        let (lo, _) = padding[d];
                        out_parts.push(format!("(d{d} + {lo}) * {}", out_strides[d]));
                    }
                    let out_expr = out_parts.join(" + ");

                    out.push_str(&format!("    buf{i}[{out_expr}] = buf{a}[ai];\n"));
                    out.push_str("  }\n");
                }
                Op::Shrink { bounds } => {
                    let a = node.inputs[0].0;
                    let a_shape = dim_shape_to_usize(&graph.nodes[a].shape);
                    let a_strides = strides(&a_shape);
                    let out_strides = strides(&usize_shape);
                    let ndim = usize_shape.len();

                    let usize_bounds: Vec<(usize, usize)> = bounds.iter().map(|(lo, hi)| {
                        (lo.as_usize().expect("symbolic dim not yet supported in AS backend"),
                         hi.as_usize().expect("symbolic dim not yet supported in AS backend"))
                    }).collect();

                    out.push_str(&format!("  // shrink {usize_bounds:?}\n"));
                    out.push_str(&format!("  for (let oi: i32 = 0; oi < {size}; oi++) {{\n"));

                    // Decompose output index into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (oi / {}) % {};\n",
                                out_strides[d], usize_shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = oi % {};\n",
                                usize_shape[d]
                            ));
                        }
                    }

                    // Compute input index by adding lo offsets
                    let mut in_parts = Vec::new();
                    for d in 0..ndim {
                        let (lo, _) = usize_bounds[d];
                        in_parts.push(format!("(d{d} + {lo}) * {}", a_strides[d]));
                    }
                    let in_expr = in_parts.join(" + ");

                    out.push_str(&format!("    buf{i}[oi] = buf{a}[{in_expr}];\n"));
                    out.push_str("  }\n");
                }
            }
        }

        let last = graph.nodes.len() - 1;
        out.push_str(&format!("\n  return buf{last};\n"));
        out.push_str("}\n");

        out
    }
}

/// Generate an index expression that maps a flat index in `out_shape` to a flat
/// index in `src_shape`, handling broadcasting (dimensions of size 1 in src).
fn broadcast_index_expr(iter_var: &str, out_shape: &[usize], src_shape: &[usize]) -> String {
    // Scalar source — always index 0
    if src_shape.is_empty() {
        return "0".to_string();
    }

    // Same shape — identity
    if out_shape == src_shape {
        return iter_var.to_string();
    }

    let ndim = out_shape.len();
    let out_strides = strides(out_shape);
    let src_strides = strides(src_shape);

    // Pad src shape with 1s on the left to match ndim
    let offset = ndim - src_shape.len();

    let mut parts = Vec::new();
    for d in 0..ndim {
        let src_d = if d >= offset { d - offset } else { continue };
        if src_shape[src_d] == 1 {
            // Broadcast dimension — contributes 0 to index
            continue;
        }
        let coord = if d < ndim - 1 {
            format!("(({iter_var} / {}) % {})", out_strides[d], out_shape[d])
        } else {
            format!("({iter_var} % {})", out_shape[d])
        };
        parts.push(format!("{coord} * {}", src_strides[src_d]));
    }

    if parts.is_empty() {
        "0".to_string()
    } else if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        parts.join(" + ")
    }
}

// ---------------------------------------------------------------------------
// Kernel signature and deduplication
// ---------------------------------------------------------------------------

/// Normalized instruction for signature comparison — buf IDs replaced with param ordinals.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum NormalizedInst {
    Load { param_idx: usize, index: Index },
    Const(u64), // f64 bits for Eq/Hash
    DimVar(usize),
    Neg(usize), Recip(usize), Exp2(usize), Log2(usize), Sqrt(usize),
    Add(usize, usize), Mul(usize, usize), Max(usize, usize), CmpLt(usize, usize),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct LoopSignature {
    shape: Vec<Dim>,
    reduce: Option<(usize, ReduceOp, Dim)>, // axis, op, size
    body: Vec<NormalizedInst>,
    result: usize,
    tile: Option<Vec<Dim>>,
}

fn normalize_body(body: &[Inst]) -> (Vec<NormalizedInst>, Vec<usize>) {
    let mut buf_map: HashMap<usize, usize> = HashMap::new();
    let mut buf_ordering: Vec<usize> = Vec::new();

    let normalized = body.iter().map(|inst| match inst {
        Inst::Load { buf, index } => {
            let param_idx = if let Some(&idx) = buf_map.get(buf) {
                idx
            } else {
                let idx = buf_ordering.len();
                buf_map.insert(*buf, idx);
                buf_ordering.push(*buf);
                idx
            };
            NormalizedInst::Load { param_idx, index: index.clone() }
        }
        Inst::Const(v) => NormalizedInst::Const(v.to_bits()),
        Inst::DimVar(d) => NormalizedInst::DimVar(*d),
        Inst::Neg(a) => NormalizedInst::Neg(*a),
        Inst::Recip(a) => NormalizedInst::Recip(*a),
        Inst::Exp2(a) => NormalizedInst::Exp2(*a),
        Inst::Log2(a) => NormalizedInst::Log2(*a),
        Inst::Sqrt(a) => NormalizedInst::Sqrt(*a),
        Inst::Add(a, b) => NormalizedInst::Add(*a, *b),
        Inst::Mul(a, b) => NormalizedInst::Mul(*a, *b),
        Inst::Max(a, b) => NormalizedInst::Max(*a, *b),
        Inst::CmpLt(a, b) => NormalizedInst::CmpLt(*a, *b),
    }).collect();

    (normalized, buf_ordering)
}

fn compute_loop_signature(
    shape: &[Dim], reduce: Option<&loop_ir::ReduceDesc>,
    body: &[Inst], result: usize, tile: Option<&loop_ir::TileConfig>,
) -> (LoopSignature, Vec<usize>) {
    let (norm_body, buf_ordering) = normalize_body(body);
    let sig = LoopSignature {
        shape: shape.to_vec(),
        reduce: reduce.map(|r| (r.axis, r.op, r.size.clone())),
        body: norm_body,
        result,
        tile: tile.map(|t| t.tiles.clone()),
    };
    (sig, buf_ordering)
}

// ---------------------------------------------------------------------------
// Fused codegen: lower graph to loop IR, then emit AS code
// ---------------------------------------------------------------------------

impl AssemblyScriptBackend {
    /// Emit AssemblyScript from a graph using the fused loop IR.
    pub fn emit_fused(&self, graph: &Graph) -> String {
        self.emit_fused_inner(graph, false, None)
    }

    /// Emit AssemblyScript with bounds-checking instrumentation for debugging.
    pub fn emit_fused_debug(&self, graph: &Graph) -> String {
        self.emit_fused_inner(graph, true, None)
    }

    /// Emit AssemblyScript that returns multiple output buffers concatenated.
    /// The returned Float32Array contains [output_0, output_1, ...] flattened.
    /// Use the graph node shapes to split the result on the Rust side.
    pub fn emit_fused_multi_output(
        &self,
        graph: &Graph,
        outputs: &[tensor_lang_graph::NodeId],
    ) -> String {
        self.emit_fused_inner(graph, false, Some(outputs))
    }

    /// Like emit_fused_multi_output but skips tiling (more reliable for training).
    pub fn emit_fused_multi_output_no_tile(
        &self,
        graph: &Graph,
        outputs: &[tensor_lang_graph::NodeId],
    ) -> String {
        let stmts = loop_ir::lower_with_outputs(graph, outputs);
        // Don't tile — skip tile_reduce_loops
        self.emit_fused_from_stmts(graph, &stmts, false, Some(outputs))
    }

    fn emit_fused_inner(
        &self,
        graph: &Graph,
        debug_bounds: bool,
        multi_outputs: Option<&[tensor_lang_graph::NodeId]>,
    ) -> String {
        let mut stmts = if let Some(outputs) = multi_outputs {
            loop_ir::lower_with_outputs(graph, outputs)
        } else {
            loop_ir::lower(graph)
        };
        loop_ir::tile_reduce_loops(&mut stmts);
        self.emit_fused_from_stmts(graph, &stmts, debug_bounds, multi_outputs)
    }

    fn emit_fused_from_stmts(
        &self,
        graph: &Graph,
        stmts: &[loop_ir::Stmt],
        debug_bounds: bool,
        multi_outputs: Option<&[tensor_lang_graph::NodeId]>,
    ) -> String {
        let mut out = String::new();

        // Export Float32Array type ID
        out.push_str("export const Float32Array_ID = idof<Float32Array>();\n\n");

        // Collect symbolic dimension parameters from graph shapes
        let mut dim_params: Vec<String> = Vec::new();
        for node in &graph.nodes {
            for d in &node.shape {
                collect_params(d, &mut dim_params);
            }
        }
        dim_params.sort();
        dim_params.dedup();

        // Pass 1: Register all loops, collect unique kernels
        struct KernelDef {
            shape: Vec<Dim>,
            reduce: Option<loop_ir::ReduceDesc>,
            body: Vec<Inst>,
            result: usize,
            tile: Option<loop_ir::TileConfig>,
            buf_ordering: Vec<usize>,
        }
        let mut kernel_sigs: HashMap<LoopSignature, usize> = HashMap::new();
        let mut kernel_defs: Vec<KernelDef> = Vec::new();
        // Per-stmt: Some((kernel_id, buf_ordering, out_buf)) for Loop stmts, None otherwise
        let mut loop_kernel_map: Vec<Option<(usize, Vec<usize>, usize)>> = Vec::new();

        for stmt in stmts {
            match stmt {
                Stmt::Loop { buf, shape, reduce, body, result, tile } => {
                    let (sig, buf_ordering) = compute_loop_signature(
                        shape, reduce.as_ref(), body, *result, tile.as_ref(),
                    );
                    let kernel_id = if let Some(&id) = kernel_sigs.get(&sig) {
                        id
                    } else {
                        let id = kernel_defs.len();
                        kernel_sigs.insert(sig, id);
                        kernel_defs.push(KernelDef {
                            shape: shape.clone(),
                            reduce: reduce.clone(),
                            body: body.clone(),
                            result: *result,
                            tile: tile.clone(),
                            buf_ordering: buf_ordering.clone(),
                        });
                        id
                    };
                    loop_kernel_map.push(Some((kernel_id, buf_ordering, *buf)));
                }
                _ => {
                    loop_kernel_map.push(None);
                }
            }
        }

        // Emit kernel functions BEFORE the execute function
        for (kid, kdef) in kernel_defs.iter().enumerate() {
            emit_kernel_function(
                &mut out,
                kid,
                &kdef.shape,
                kdef.reduce.as_ref(),
                &kdef.body,
                kdef.result,
                kdef.tile.as_ref(),
                kdef.buf_ordering.len(),
                &kdef.buf_ordering,
                &dim_params,
                debug_bounds,
            );
        }

        // Function signature: dim params as i32, then Float32Array inputs
        out.push_str("export function execute(");
        let mut first = true;
        for param in &dim_params {
            if !first { out.push_str(", "); }
            out.push_str(&format!("{param}: i32"));
            first = false;
        }
        let inputs: Vec<_> = graph
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
        for (_, name) in &inputs {
            if !first { out.push_str(", "); }
            out.push_str(&format!("{name}: Float32Array"));
            first = false;
        }
        out.push_str("): Float32Array {\n");

        // Debug: assert input sizes match expected
        if debug_bounds {
            for &(node_id, ref name) in &inputs {
                let expected = Dim::product(&graph.nodes[node_id].shape).to_code();
                out.push_str(&format!(
                    "  if ({name}.length != {expected}) {{ throw new Error(\"INPUT SIZE: {name} expected {expected} got \" + {name}.length.toString()); }}\n"
                ));
            }
        }

        // Emit each statement
        let mut stmt_idx = 0;
        for stmt in stmts {
            match stmt {
                Stmt::Alloc { buf, size } => {
                    // Check if this is an Input node — use the parameter directly
                    if let Op::Input { name } = &graph.nodes[*buf].op {
                        out.push_str(&format!("  const buf{buf} = {name};\n"));
                    } else {
                        out.push_str(&format!("  const buf{buf} = new Float32Array({});\n", size.to_code()));
                    }
                }
                Stmt::Fill { buf, value } => {
                    out.push_str(&format!(
                        "  buf{buf}[0] = f32({});\n",
                        format_f64(*value)
                    ));
                }
                Stmt::FillArange { buf, size } => {
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {}; i++) buf{buf}[i] = f32(i);\n",
                        size.to_code()
                    ));
                }
                Stmt::Loop { buf: _, shape, reduce, .. } => {
                    if let Some((kernel_id, ref buf_ordering, out_buf)) = loop_kernel_map[stmt_idx] {
                        emit_kernel_call(
                            &mut out,
                            kernel_id,
                            out_buf,
                            buf_ordering,
                            shape,
                            reduce.as_ref(),
                            &dim_params,
                        );
                    }
                }
                Stmt::Pad {
                    buf,
                    input_buf,
                    output_shape,
                    input_shape,
                    padding,
                } => {
                    let buf_names = default_buf_names();
                    emit_pad(&mut out, *buf, *input_buf, output_shape, input_shape, padding, &buf_names);
                }
            }
            stmt_idx += 1;
        }

        if let Some(outputs) = multi_outputs {
            // Multi-output: allocate result buffer and copy each output into it
            let size_parts: Vec<String> = outputs
                .iter()
                .map(|id| Dim::product(&graph.nodes[id.0].shape).to_code())
                .collect();
            let total_size = size_parts.join(" + ");
            out.push_str(&format!("\n  const __out = new Float32Array({total_size});\n"));
            out.push_str("  let __off: i32 = 0;\n");
            for id in outputs {
                let size_expr = Dim::product(&graph.nodes[id.0].shape).to_code();
                out.push_str(&format!(
                    "  for (let __i: i32 = 0; __i < {size_expr}; __i++) __out[__off + __i] = buf{}[__i];\n",
                    id.0
                ));
                out.push_str(&format!("  __off += {size_expr};\n"));
            }
            out.push_str("  return __out;\n");
        } else {
            let last = graph.nodes.len() - 1;
            out.push_str(&format!("\n  return buf{last};\n"));
        }
        out.push_str("}\n");
        out
    }
}

fn emit_loop(
    out: &mut String,
    buf: usize,
    shape: &[Dim],
    reduce: Option<&loop_ir::ReduceDesc>,
    body: &[Inst],
    result: usize,
    debug_bounds: bool,
    buf_names: &BufNames,
) {
    let out_size = Dim::product(shape).to_code();
    let out_strides_dim = Dim::strides(shape);
    let out_name = buf_name(buf, buf_names);

    if let Some(reduce) = reduce {
        // Reduce loop: outer loop over output elements, inner loop over reduce axis
        let ndim = shape.len();
        let (init, combine_start, combine_end) = match reduce.op {
            ReduceOp::Sum => ("f32(0.0)", "acc = acc + ", ";"),
            ReduceOp::Max => ("f32(-Infinity)", "{ const _rv: f32 = ", "; acc = _rv > acc ? _rv : acc; }"),
        };

        out.push_str(&format!("  for (let oi: i32 = 0; oi < {out_size}; oi++) {{\n"));
        out.push_str(&format!("    let acc: f32 = {init};\n"));

        // Decompose output flat index into dimension variables
        for d in 0..ndim {
            if d < ndim - 1 {
                out.push_str(&format!(
                    "    const d{d}: i32 = (oi / {}) % {};\n",
                    out_strides_dim[d].to_code(), shape[d].to_code()
                ));
            } else {
                out.push_str(&format!(
                    "    const d{d}: i32 = oi % {};\n",
                    shape[d].to_code()
                ));
            }
        }

        // Inner reduce loop — the reduce dim index is ndim (virtual dim)
        let reduce_dim = ndim; // convention: reduce dim is after output dims
        out.push_str(&format!(
            "    for (let d{reduce_dim}: i32 = 0; d{reduce_dim} < {}; d{reduce_dim}++) {{\n",
            reduce.size.to_code()
        ));

        // Emit body instructions
        if debug_bounds {
            emit_body(out, body, "      ", true, buf_names);
        } else {
            emit_body_unchecked(out, body, "      ", buf_names);
        }

        // Accumulate
        let result_var = format!("t{result}");
        out.push_str(&format!("      {combine_start}{result_var}{combine_end}\n"));
        out.push_str("    }\n");
        out.push_str(&format!("    {out_name}[oi] = acc;\n"));
        out.push_str("  }\n");
    } else {
        // Simple elementwise loop
        out.push_str(&format!("  for (let oi: i32 = 0; oi < {out_size}; oi++) {{\n"));

        // Decompose output flat index into dimension variables
        let ndim = shape.len();
        if ndim > 0 {
            for d in 0..ndim {
                if d < ndim - 1 {
                    out.push_str(&format!(
                        "    const d{d}: i32 = (oi / {}) % {};\n",
                        out_strides_dim[d].to_code(), shape[d].to_code()
                    ));
                } else {
                    out.push_str(&format!(
                        "    const d{d}: i32 = oi % {};\n",
                        shape[d].to_code()
                    ));
                }
            }
        }

        // Emit body instructions
        if debug_bounds {
            emit_body(out, body, "    ", true, buf_names);
        } else {
            emit_body_unchecked(out, body, "    ", buf_names);
        }

        // Store result
        let result_var = format!("t{result}");
        out.push_str(&format!("    {out_name}[oi] = {result_var};\n"));
        out.push_str("  }\n");
    }
}

fn emit_tiled_loop(
    out: &mut String,
    buf: usize,
    shape: &[Dim],
    reduce: &loop_ir::ReduceDesc,
    body: &[Inst],
    result: usize,
    tile_cfg: &loop_ir::TileConfig,
    _debug_bounds: bool,
    buf_names: &BufNames,
) {
    let ndim = shape.len();
    let tiles = &tile_cfg.tiles;
    // Tile sizes are always concrete (set by tile_reduce_loops)
    let tk = tiles[ndim].as_usize().expect("tile sizes must be concrete");
    let reduce_dim = ndim;

    let batch_dims = ndim.saturating_sub(2);
    let m_dim = ndim - 2;
    let n_dim = ndim - 1;
    // Dimension sizes may be symbolic (Param) or concrete (Lit)
    let m_size_code = shape[m_dim].to_code();
    let n_size_code = shape[n_dim].to_code();
    let k_size_code = reduce.size.to_code();
    let tm = tiles[m_dim].as_usize().expect("tile sizes must be concrete");
    let tn = tiles[n_dim].as_usize().expect("tile sizes must be concrete");

    // Block counts: pre-compute as literals when dims are concrete, else runtime expr
    let m_blocks_code = match shape[m_dim].as_usize() {
        Some(m) => format!("{}", (m + tm - 1) / tm),
        None => format!("(({m_size_code} + {}) / {})", tm - 1, tm),
    };
    let k_blocks_code = match reduce.size.as_usize() {
        Some(k) => format!("{}", (k + tk - 1) / tk),
        None => format!("(({k_size_code} + {}) / {})", tk - 1, tk),
    };

    // Unroll factor: always concrete, capped at tile size
    let unroll: usize = 32_usize.min(tn);

    let init = match reduce.op {
        ReduceOp::Sum => "f32(0.0)",
        ReduceOp::Max => "f32(-Infinity)",
    };

    let batch_strides = Dim::strides(shape);

    let mut indent = "  ".to_string();

    // Batch loops
    for d in 0..batch_dims {
        out.push_str(&format!(
            "{indent}for (let d{d}: i32 = 0; d{d} < {}; d{d}++) {{\n",
            shape[d].to_code()
        ));
        indent.push_str("  ");
    }

    // M block loop
    out.push_str(&format!(
        "{indent}for (let m_blk: i32 = 0; m_blk < {m_blocks_code}; m_blk++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const m_end: i32 = {m_size_code} - m_blk * {tm} < {tm} ? {m_size_code} - m_blk * {tm} : {tm};\n"
    ));

    // mi loop
    out.push_str(&format!(
        "{indent}for (let mi: i32 = 0; mi < m_end; mi++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const d{m_dim}: i32 = m_blk * {tm} + mi;\n"
    ));
    // Precompute row base
    let mut base_parts = Vec::new();
    for d in 0..batch_dims {
        base_parts.push(format!("d{d} * {}", batch_strides[d].to_code()));
    }
    base_parts.push(format!("d{m_dim} * {n_size_code}"));
    let row_base = base_parts.join(" + ");
    out.push_str(&format!(
        "{indent}const _row_base: i32 = {row_base};\n"
    ));

    // --- Emit unrolled N tile blocks ---
    let n_blocks_code = match shape[n_dim].as_usize() {
        Some(n) => format!("{}", (n + tn - 1) / tn),
        None => format!("(({n_size_code} + {}) / {})", tn - 1, tn),
    };
    out.push_str(&format!(
        "{indent}for (let n_blk: i32 = 0; n_blk < {n_blocks_code}; n_blk++) {{\n"
    ));
    indent.push_str("  ");
    // Compute effective tile size for this block (handles remainder at runtime)
    out.push_str(&format!(
        "{indent}const n_tile: i32 = {n_size_code} - n_blk * {tn} < {tn} ? {n_size_code} - n_blk * {tn} : {tn};\n"
    ));

    // Unrolled portion: process unroll elements at a time (always use dynamic bound)
    let unrolled_count = format!("(n_tile / {unroll})");
    out.push_str(&format!(
        "{indent}for (let ni_grp: i32 = 0; ni_grp < {unrolled_count}; ni_grp++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const ni_base: i32 = n_blk * {tn} + ni_grp * {unroll};\n"
    ));

    // Declare scalar accumulators
    for u in 0..unroll {
        out.push_str(&format!(
            "{indent}let _acc{u}: f32 = {init};\n"
        ));
    }

    // K block loop
    out.push_str(&format!(
        "{indent}for (let k_blk: i32 = 0; k_blk < {k_blocks_code}; k_blk++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const k_end: i32 = {k_size_code} - k_blk * {tk} < {tk} ? {k_size_code} - k_blk * {tk} : {tk};\n"
    ));

    // ki loop
    out.push_str(&format!(
        "{indent}for (let ki: i32 = 0; ki < k_end; ki++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const d{reduce_dim}: i32 = k_blk * {tk} + ki;\n"
    ));

    // Analyze which body instructions are invariant w.r.t. d{n_dim}
    let depends_on_n = compute_n_dependence(body, n_dim);

    // Emit hoisted (n-invariant) instructions once
    for (j, inst) in body.iter().enumerate() {
        if !depends_on_n[j] {
            let expr = match inst {
                Inst::Load { buf: b, index } => {
                    let idx = emit_index(index);
                    let name = buf_name(*b, buf_names);
                    format!("unchecked({name}[{idx}])")
                }
                _ => emit_inst(inst, buf_names),
            };
            out.push_str(&format!("{indent}const t{j}: f32 = {expr};\n"));
        }
    }

    // Emit n-dependent instructions unroll times, each in a block scope
    for u in 0..unroll {
        out.push_str(&format!("{indent}{{\n"));
        let inner_indent = format!("{indent}  ");
        out.push_str(&format!(
            "{inner_indent}const d{n_dim}: i32 = ni_base + {u};\n"
        ));
        // Only emit instructions that depend on n
        for (j, inst) in body.iter().enumerate() {
            if depends_on_n[j] {
                let expr = match inst {
                    Inst::Load { buf: b, index } => {
                        let idx = emit_index(index);
                        let name = buf_name(*b, buf_names);
                        format!("unchecked({name}[{idx}])")
                    }
                    _ => emit_inst(inst, buf_names),
                };
                out.push_str(&format!("{inner_indent}const t{j}: f32 = {expr};\n"));
            }
        }
        let result_var = format!("t{result}");
        match reduce.op {
            ReduceOp::Sum => {
                out.push_str(&format!(
                    "{inner_indent}_acc{u} = _acc{u} + {result_var};\n"
                ));
            }
            ReduceOp::Max => {
                out.push_str(&format!(
                    "{inner_indent}_acc{u} = {result_var} > _acc{u} ? {result_var} : _acc{u};\n"
                ));
            }
        }
        out.push_str(&format!("{indent}}}\n"));
    }

    // Close ki, k_blk loops
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // ki
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // k_blk

    // Write back scalar accumulators to output buffer
    let out_name = buf_name(buf, buf_names);
    for u in 0..unroll {
        out.push_str(&format!(
            "{indent}unchecked({out_name}[_row_base + ni_base + {u}] = _acc{u});\n"
        ));
    }

    // Close ni_grp loop
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // ni_grp

    // Remainder: handle leftover ni elements with a scalar loop
    out.push_str(&format!(
        "{indent}for (let ni: i32 = (n_tile / {unroll}) * {unroll}; ni < n_tile; ni++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const d{n_dim}: i32 = n_blk * {tn} + ni;\n"
    ));
    out.push_str(&format!(
        "{indent}let _acc_r: f32 = {init};\n"
    ));
    out.push_str(&format!(
        "{indent}for (let k_blk: i32 = 0; k_blk < {k_blocks_code}; k_blk++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const k_end: i32 = {k_size_code} - k_blk * {tk} < {tk} ? {k_size_code} - k_blk * {tk} : {tk};\n"
    ));
    out.push_str(&format!(
        "{indent}for (let ki: i32 = 0; ki < k_end; ki++) {{\n"
    ));
    indent.push_str("  ");
    out.push_str(&format!(
        "{indent}const d{reduce_dim}: i32 = k_blk * {tk} + ki;\n"
    ));
    emit_body_unchecked(out, body, &indent, buf_names);
    let result_var = format!("t{result}");
    match reduce.op {
        ReduceOp::Sum => {
            out.push_str(&format!(
                "{indent}_acc_r = _acc_r + {result_var};\n"
            ));
        }
        ReduceOp::Max => {
            out.push_str(&format!(
                "{indent}_acc_r = {result_var} > _acc_r ? {result_var} : _acc_r;\n"
            ));
        }
    }
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // ki
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // k_blk
    out.push_str(&format!(
        "{indent}unchecked({out_name}[_row_base + n_blk * {tn} + ni] = _acc_r);\n"
    ));
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // ni remainder

    // Close n_blk loop
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // n_blk

    // Close mi, m_blk loops
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // mi
    indent.truncate(indent.len() - 2);
    out.push_str(&format!("{indent}}}\n")); // m_blk

    // Close batch loops
    for _ in 0..batch_dims {
        indent.truncate(indent.len() - 2);
        out.push_str(&format!("{indent}}}\n"));
    }
}

/// Determine which body instructions are invariant w.r.t. the N dimension (d{n_dim}).
/// Returns a bool per instruction: true = depends on n_dim, false = invariant.
fn compute_n_dependence(body: &[Inst], n_dim: usize) -> Vec<bool> {
    let mut depends_on_n = vec![false; body.len()];
    for (j, inst) in body.iter().enumerate() {
        depends_on_n[j] = match inst {
            Inst::Load { index, .. } => match index {
                Index::Strided { parts, .. } => parts.iter().any(|(dim, _)| *dim == n_dim),
                Index::Flat => true, // conservative
            },
            Inst::DimVar(d) => *d == n_dim,
            Inst::Const(_) => false,
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a) | Inst::Log2(a) | Inst::Sqrt(a) => {
                depends_on_n[*a]
            }
            Inst::Add(a, b) | Inst::Mul(a, b) | Inst::Max(a, b) | Inst::CmpLt(a, b) => {
                depends_on_n[*a] || depends_on_n[*b]
            }
        };
    }
    depends_on_n
}

fn emit_pad(
    out: &mut String,
    buf: usize,
    input_buf: usize,
    output_shape: &[Dim],
    input_shape: &[Dim],
    padding: &[(usize, usize)],
    buf_names: &BufNames,
) {
    let out_size = Dim::product(output_shape).to_code();
    let in_size = Dim::product(input_shape).to_code();
    let in_strides = Dim::strides(input_shape);
    let out_strides = Dim::strides(output_shape);
    let ndim = output_shape.len();
    let out_name = buf_name(buf, buf_names);
    let in_name = buf_name(input_buf, buf_names);

    // Zero-fill
    out.push_str(&format!(
        "  for (let i: i32 = 0; i < {out_size}; i++) {out_name}[i] = f32(0.0);\n"
    ));
    // Copy input with padding offsets
    out.push_str(&format!("  for (let ai: i32 = 0; ai < {in_size}; ai++) {{\n"));
    for d in 0..ndim {
        if d < ndim - 1 {
            out.push_str(&format!(
                "    const d{d}: i32 = (ai / {}) % {};\n",
                in_strides[d].to_code(), input_shape[d].to_code()
            ));
        } else {
            out.push_str(&format!(
                "    const d{d}: i32 = ai % {};\n",
                input_shape[d].to_code()
            ));
        }
    }
    let mut out_parts = Vec::new();
    for d in 0..ndim {
        let (lo, _) = padding[d];
        out_parts.push(format!("(d{d} + {lo}) * {}", out_strides[d].to_code()));
    }
    let out_expr = out_parts.join(" + ");
    out.push_str(&format!(
        "    {out_name}[{out_expr}] = {in_name}[ai];\n"
    ));
    out.push_str("  }\n");
}

fn emit_body(out: &mut String, body: &[Inst], indent: &str, debug_bounds: bool, buf_names: &BufNames) {
    for (j, inst) in body.iter().enumerate() {
        // Emit bounds check before load instructions
        if debug_bounds {
            if let Inst::Load { buf, index } = inst {
                let idx_expr = emit_index(index);
                let name = buf_name(*buf, buf_names);
                out.push_str(&format!(
                    "{indent}if ({idx_expr} < 0 || {idx_expr} >= {name}.length) {{ throw new Error(\"OOB: {name}[\" + ({idx_expr}).toString() + \"] len=\" + {name}.length.toString()); }}\n"
                ));
            }
        }
        let expr = emit_inst(inst, buf_names);
        out.push_str(&format!("{indent}const t{j}: f32 = {expr};\n"));
    }
}

/// Like emit_body but wraps Load instructions with unchecked() for perf in tiled loops.
fn emit_body_unchecked(out: &mut String, body: &[Inst], indent: &str, buf_names: &BufNames) {
    for (j, inst) in body.iter().enumerate() {
        let expr = match inst {
            Inst::Load { buf, index } => {
                let idx = emit_index(index);
                let name = buf_name(*buf, buf_names);
                format!("unchecked({name}[{idx}])")
            }
            _ => emit_inst(inst, buf_names),
        };
        out.push_str(&format!("{indent}const t{j}: f32 = {expr};\n"));
    }
}

fn emit_inst(inst: &Inst, buf_names: &BufNames) -> String {
    match inst {
        Inst::Load { buf, index } => {
            let idx = emit_index(index);
            let name = buf_name(*buf, buf_names);
            format!("{name}[{idx}]")
        }
        Inst::Const(v) => format!("f32({})", format_f64(*v)),
        Inst::DimVar(d) => format!("f32(d{d})"),
        Inst::Neg(a) => format!("(-t{a})"),
        Inst::Recip(a) => format!("(f32(1.0) / t{a})"),
        Inst::Exp2(a) => format!("f32(Math.pow(2.0, f64(t{a})))"),
        Inst::Log2(a) => format!("f32(Math.log2(f64(t{a})))"),
        Inst::Sqrt(a) => format!("f32(Math.sqrt(f64(t{a})))"),
        Inst::Add(a, b) => format!("(t{a} + t{b})"),
        Inst::Mul(a, b) => format!("(t{a} * t{b})"),
        Inst::Max(a, b) => format!("(t{a} > t{b} ? t{a} : t{b})"),
        Inst::CmpLt(a, b) => format!("(t{a} < t{b} ? f32(1.0) : f32(0.0))"),
    }
}

/// Look up a buffer name. If in buf_names, use that; otherwise default to `buf{id}`.
fn buf_name(id: usize, buf_names: &BufNames) -> String {
    if let Some(name) = buf_names.get(&id) {
        name.clone()
    } else {
        format!("buf{id}")
    }
}

/// Returns an empty BufNames (all buffers use default `buf{N}` names).
fn default_buf_names() -> BufNames {
    HashMap::new()
}

fn emit_kernel_function(
    out: &mut String,
    kernel_id: usize,
    shape: &[Dim],
    reduce: Option<&loop_ir::ReduceDesc>,
    body: &[Inst],
    result: usize,
    tile: Option<&loop_ir::TileConfig>,
    n_input_bufs: usize,
    first_buf_ordering: &[usize],
    dim_params: &[String],
    debug_bounds: bool,
) {
    // Function signature
    out.push_str(&format!("function kernel_{kernel_id}("));
    let mut first = true;
    for dp in dim_params {
        if !first { out.push_str(", "); }
        out.push_str(&format!("{dp}: i32"));
        first = false;
    }
    // Output buffer
    if !first { out.push_str(", "); }
    out.push_str("_out: Float32Array");
    // Input buffers
    for i in 0..n_input_bufs {
        out.push_str(&format!(", _in{i}: Float32Array"));
    }
    out.push_str("): void {\n");

    // Build buf_names mapping for this kernel:
    // - The output buf (from the first instance) maps to "_out"
    // - Each input buf maps to "_in{i}" based on its position in buf_ordering
    let mut buf_names: BufNames = HashMap::new();
    buf_names.insert(KERNEL_OUT_BUF, "_out".to_string());
    for (i, &orig_buf) in first_buf_ordering.iter().enumerate() {
        buf_names.insert(orig_buf, format!("_in{i}"));
    }

    // Call emit_loop or emit_tiled_loop with KERNEL_OUT_BUF as the output buf
    if let (Some(reduce), Some(tile_cfg)) = (reduce, tile) {
        emit_tiled_loop(out, KERNEL_OUT_BUF, shape, reduce, body, result, tile_cfg, debug_bounds, &buf_names);
    } else {
        emit_loop(out, KERNEL_OUT_BUF, shape, reduce, body, result, debug_bounds, &buf_names);
    }

    out.push_str("}\n\n");
}

fn emit_kernel_call(
    out: &mut String,
    kernel_id: usize,
    out_buf: usize,
    buf_ordering: &[usize],
    _shape: &[Dim],
    _reduce: Option<&loop_ir::ReduceDesc>,
    dim_params: &[String],
) {
    out.push_str(&format!("  kernel_{kernel_id}("));
    let mut first = true;
    // Dimension params
    for dp in dim_params {
        if !first { out.push_str(", "); }
        out.push_str(dp);
        first = false;
    }
    // Output buffer
    if !first { out.push_str(", "); }
    out.push_str(&format!("buf{out_buf}"));
    // Input buffers
    for &buf_id in buf_ordering {
        out.push_str(&format!(", buf{buf_id}"));
    }
    out.push_str(");\n");
}

fn emit_index(index: &Index) -> String {
    match index {
        Index::Flat => "oi".to_string(),
        Index::Strided { parts, offset } => {
            if parts.is_empty() {
                return if !offset.is_zero() {
                    offset.to_code()
                } else {
                    "0".to_string()
                };
            }
            let terms: Vec<String> = parts
                .iter()
                .filter(|(_, stride)| !stride.is_zero())
                .map(|(dim, stride)| {
                    if stride.is_one() {
                        format!("d{dim}")
                    } else {
                        format!("(d{dim} * {})", stride.to_code())
                    }
                })
                .collect();
            let base = if terms.is_empty() {
                "0".to_string()
            } else {
                terms.join(" + ")
            };
            if !offset.is_zero() {
                format!("{base} + {}", offset.to_code())
            } else {
                base
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_lang_graph::compile;

    #[test]
    fn test_emit_simple_add() {
        let g = compile("let x = load([10]) let y = load([10]) let z = add(x, y)");
        let backend = AssemblyScriptBackend;
        let code = backend.emit(&g);
        println!("{code}");
        assert!(code.contains("export function execute("));
        assert!(code.contains("new Float32Array(10)"));
        assert!(code.contains("va + vb"));
    }

    #[test]
    fn test_emit_softmax() {
        let input = r#"
            fn softmax(x) {
                let m = max(x, axis: 1)
                let e = exp(sub(x, m))
                let s = sum(e, axis: 1)
                mul(recip(s), e)
            }
            let x = load([4, 10])
            let y = softmax(x)
        "#;
        let g = compile(input);
        let backend = AssemblyScriptBackend;
        let code = backend.emit(&g);
        println!("{code}");
        // Check buffer sizes are correct
        assert!(code.contains("new Float32Array(4)"));   // reduce output [4,1] = 4 elements
        assert!(code.contains("new Float32Array(40)"));  // full [4,10] = 40 elements
        assert!(code.contains("return buf"));
    }

    #[test]
    fn test_emit_broadcast_shapes() {
        // mul(tensor[4,10], reduced[4,1]) should generate proper broadcast indexing
        let input = r#"
            let x = load([4, 10])
            let s = sum(x, axis: 1)
            let y = mul(x, s)
        "#;
        let g = compile(input);
        let backend = AssemblyScriptBackend;
        let code = backend.emit(&g);
        println!("{code}");
        // The broadcast index for s ([4,1]) into [4,10] should not just be "idx"
        // It should compute the row index only
        assert!(code.contains("va * vb"));
    }

    #[test]
    fn test_emit_fused_symbolic() {
        // A simple symbolic program: add two tensors with symbolic T
        let input = "dim T\nlet x = load([1, T, 768])\nlet y = neg(x)";
        let g = compile(input);
        let backend = AssemblyScriptBackend;
        let code = backend.emit_fused(&g);
        println!("{code}");
        // The code should contain T as a runtime variable in loop bounds
        // Buffer size for [1, T, 768] = T * 768
        assert!(code.contains("(T * 768)"), "expected T*768 in buffer size");
        // Loop bound should reference T
        assert!(code.contains("T"), "expected T in generated code");
    }
}
