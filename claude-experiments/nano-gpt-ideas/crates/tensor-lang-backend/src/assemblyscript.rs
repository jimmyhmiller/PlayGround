use tensor_lang_graph::{Graph, Op};
use crate::Backend;
use crate::loop_ir::{self, Stmt, Inst, Index, ReduceOp};

pub struct AssemblyScriptBackend;

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
                    out.push_str(&format!(
                        "  const buf{i} = new Float32Array({size});\n  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(i);\n"
                    ));
                }
                _ => {
                    let size = shape_size(&node.shape);
                    out.push_str(&format!("  const buf{i} = new Float32Array({size});\n"));
                }
            }
        }

        out.push_str("\n");

        // Execute each node
        for (i, node) in graph.nodes.iter().enumerate() {
            let size = shape_size(&node.shape);
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
                    let a_shape = &graph.nodes[a].shape;
                    let b_shape = &graph.nodes[b].shape;
                    let out_shape = &node.shape;

                    let op_expr = match &node.op {
                        Op::Add => "va + vb",
                        Op::Mul => "va * vb",
                        Op::Max => "va > vb ? va : vb",
                        Op::CmpLt => "va < vb ? f32(1.0) : f32(0.0)",
                        _ => unreachable!(),
                    };

                    let a_idx = broadcast_index_expr("idx", out_shape, a_shape);
                    let b_idx = broadcast_index_expr("idx", out_shape, b_shape);

                    out.push_str(&format!("  for (let idx: i32 = 0; idx < {size}; idx++) {{\n"));
                    out.push_str(&format!("    const va: f32 = buf{a}[{a_idx}];\n"));
                    out.push_str(&format!("    const vb: f32 = buf{b}[{b_idx}];\n"));
                    out.push_str(&format!("    buf{i}[idx] = {op_expr};\n"));
                    out.push_str("  }\n");
                }

                // Reduce with keepdim
                Op::ReduceSum { axis } | Op::ReduceMax { axis } => {
                    let a = node.inputs[0].0;
                    let a_shape = &graph.nodes[a].shape;
                    let (init, combine) = match &node.op {
                        Op::ReduceSum { .. } => ("f32(0.0)", "acc + val"),
                        Op::ReduceMax { .. } => ("f32(-Infinity)", "val > acc ? val : acc"),
                        _ => unreachable!(),
                    };

                    let axis = *axis;
                    let a_strides = strides(a_shape);
                    let axis_size = a_shape[axis];
                    let out_strides = strides(&node.shape);

                    // For each output element, iterate over the reduced axis
                    out.push_str(&format!("  // reduce axis={axis} ({a_shape:?} -> {:?})\n", node.shape));
                    out.push_str(&format!("  for (let oi: i32 = 0; oi < {size}; oi++) {{\n"));
                    out.push_str(&format!("    let acc: f32 = {init};\n"));

                    // Compute the base index in the input from the output index
                    // We need to map output flat index -> multi-dim coords -> input base index
                    // Then sweep over the axis dimension
                    let ndim = node.shape.len();
                    // Decompose oi into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (oi / {}) % {};\n",
                                out_strides[d], node.shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = oi % {};\n",
                                node.shape[d]
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
                    let a_shape = &graph.nodes[a].shape;
                    let a_strides = strides(a_shape);
                    let out_strides = strides(&node.shape);
                    let ndim = node.shape.len();

                    out.push_str(&format!("  // permute {order:?}\n"));
                    out.push_str(&format!("  for (let oi: i32 = 0; oi < {size}; oi++) {{\n"));

                    // Decompose output index into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (oi / {}) % {};\n",
                                out_strides[d], node.shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = oi % {};\n",
                                node.shape[d]
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
                    let a_shape = &graph.nodes[a].shape;
                    let a_idx = broadcast_index_expr("i", &node.shape, a_shape);
                    out.push_str(&format!(
                        "  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = buf{a}[{a_idx}];\n"
                    ));
                }
                Op::Pad { padding } => {
                    let a = node.inputs[0].0;
                    let a_shape = &graph.nodes[a].shape;
                    let a_strides = strides(a_shape);
                    let out_strides = strides(&node.shape);
                    let ndim = node.shape.len();

                    // Zero-fill then copy from input
                    out.push_str(&format!("  // pad {padding:?}\n"));
                    out.push_str(&format!("  for (let i: i32 = 0; i < {size}; i++) buf{i}[i] = f32(0.0);\n"));
                    let a_size = shape_size(a_shape);
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
                    let a_shape = &graph.nodes[a].shape;
                    let a_strides = strides(a_shape);
                    let out_strides = strides(&node.shape);
                    let ndim = node.shape.len();

                    out.push_str(&format!("  // shrink {bounds:?}\n"));
                    out.push_str(&format!("  for (let oi: i32 = 0; oi < {size}; oi++) {{\n"));

                    // Decompose output index into coordinates
                    for d in 0..ndim {
                        if d < ndim - 1 {
                            out.push_str(&format!(
                                "    const d{d}: i32 = (oi / {}) % {};\n",
                                out_strides[d], node.shape[d]
                            ));
                        } else {
                            out.push_str(&format!(
                                "    const d{d}: i32 = oi % {};\n",
                                node.shape[d]
                            ));
                        }
                    }

                    // Compute input index by adding lo offsets
                    let mut in_parts = Vec::new();
                    for d in 0..ndim {
                        let (lo, _) = bounds[d];
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
// Fused codegen: lower graph to loop IR, then emit AS code
// ---------------------------------------------------------------------------

impl AssemblyScriptBackend {
    /// Emit AssemblyScript from a graph using the fused loop IR.
    pub fn emit_fused(&self, graph: &Graph) -> String {
        self.emit_fused_inner(graph, false)
    }

    /// Emit AssemblyScript with bounds-checking instrumentation for debugging.
    pub fn emit_fused_debug(&self, graph: &Graph) -> String {
        self.emit_fused_inner(graph, true)
    }

    fn emit_fused_inner(&self, graph: &Graph, debug_bounds: bool) -> String {
        let stmts = loop_ir::lower(graph);
        let mut out = String::new();

        // Export Float32Array type ID
        out.push_str("export const Float32Array_ID = idof<Float32Array>();\n\n");

        // Function signature
        out.push_str("export function execute(");
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
        for (idx, (_, name)) in inputs.iter().enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("{name}: Float32Array"));
        }
        out.push_str("): Float32Array {\n");

        // Debug: assert input sizes match expected
        if debug_bounds {
            for &(node_id, ref name) in &inputs {
                let expected: usize = shape_size(&graph.nodes[node_id].shape);
                out.push_str(&format!(
                    "  if ({name}.length != {expected}) {{ throw new Error(\"INPUT SIZE: {name} expected {expected} got \" + {name}.length.toString()); }}\n"
                ));
            }
        }

        // Emit each statement
        for stmt in &stmts {
            match stmt {
                Stmt::Alloc { buf, size } => {
                    // Check if this is an Input node — use the parameter directly
                    if let Op::Input { name } = &graph.nodes[*buf].op {
                        out.push_str(&format!("  const buf{buf} = {name};\n"));
                    } else {
                        out.push_str(&format!("  const buf{buf} = new Float32Array({size});\n"));
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
                        "  for (let i: i32 = 0; i < {size}; i++) buf{buf}[i] = f32(i);\n"
                    ));
                }
                Stmt::Loop {
                    buf,
                    shape,
                    reduce,
                    body,
                    result,
                } => {
                    emit_loop(&mut out, *buf, shape, reduce.as_ref(), body, *result, debug_bounds);
                }
                Stmt::Pad {
                    buf,
                    input_buf,
                    output_shape,
                    input_shape,
                    padding,
                } => {
                    emit_pad(&mut out, *buf, *input_buf, output_shape, input_shape, padding);
                }
            }
        }

        let last = graph.nodes.len() - 1;
        out.push_str(&format!("\n  return buf{last};\n"));
        out.push_str("}\n");
        out
    }
}

fn emit_loop(
    out: &mut String,
    buf: usize,
    shape: &[usize],
    reduce: Option<&loop_ir::ReduceDesc>,
    body: &[Inst],
    result: usize,
    debug_bounds: bool,
) {
    let out_size: usize = if shape.is_empty() { 1 } else { shape.iter().product() };

    if let Some(reduce) = reduce {
        // Reduce loop: outer loop over output elements, inner loop over reduce axis
        let out_strides = strides(shape);
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
                    out_strides[d], shape[d]
                ));
            } else {
                out.push_str(&format!(
                    "    const d{d}: i32 = oi % {};\n",
                    shape[d]
                ));
            }
        }

        // Inner reduce loop — the reduce dim index is ndim (virtual dim)
        let reduce_dim = ndim; // convention: reduce dim is after output dims
        out.push_str(&format!(
            "    for (let d{reduce_dim}: i32 = 0; d{reduce_dim} < {}; d{reduce_dim}++) {{\n",
            reduce.size
        ));

        // Emit body instructions
        emit_body(out, body, "      ", debug_bounds);

        // Accumulate
        let result_var = format!("t{result}");
        out.push_str(&format!("      {combine_start}{result_var}{combine_end}\n"));
        out.push_str("    }\n");
        out.push_str(&format!("    buf{buf}[oi] = acc;\n"));
        out.push_str("  }\n");
    } else {
        // Simple elementwise loop
        out.push_str(&format!("  for (let oi: i32 = 0; oi < {out_size}; oi++) {{\n"));

        // Decompose output flat index into dimension variables
        let ndim = shape.len();
        if ndim > 0 {
            let out_strides = strides(shape);
            for d in 0..ndim {
                if d < ndim - 1 {
                    out.push_str(&format!(
                        "    const d{d}: i32 = (oi / {}) % {};\n",
                        out_strides[d], shape[d]
                    ));
                } else {
                    out.push_str(&format!(
                        "    const d{d}: i32 = oi % {};\n",
                        shape[d]
                    ));
                }
            }
        }

        // Emit body instructions
        emit_body(out, body, "    ", debug_bounds);

        // Store result
        let result_var = format!("t{result}");
        out.push_str(&format!("    buf{buf}[oi] = {result_var};\n"));
        out.push_str("  }\n");
    }
}

fn emit_pad(
    out: &mut String,
    buf: usize,
    input_buf: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    padding: &[(usize, usize)],
) {
    let out_size: usize = output_shape.iter().product();
    let in_size: usize = input_shape.iter().product();
    let in_strides = strides(input_shape);
    let out_strides = strides(output_shape);
    let ndim = output_shape.len();

    // Zero-fill
    out.push_str(&format!(
        "  for (let i: i32 = 0; i < {out_size}; i++) buf{buf}[i] = f32(0.0);\n"
    ));
    // Copy input with padding offsets
    out.push_str(&format!("  for (let ai: i32 = 0; ai < {in_size}; ai++) {{\n"));
    for d in 0..ndim {
        if d < ndim - 1 {
            out.push_str(&format!(
                "    const d{d}: i32 = (ai / {}) % {};\n",
                in_strides[d], input_shape[d]
            ));
        } else {
            out.push_str(&format!(
                "    const d{d}: i32 = ai % {};\n",
                input_shape[d]
            ));
        }
    }
    let mut out_parts = Vec::new();
    for d in 0..ndim {
        let (lo, _) = padding[d];
        out_parts.push(format!("(d{d} + {lo}) * {}", out_strides[d]));
    }
    let out_expr = out_parts.join(" + ");
    out.push_str(&format!(
        "    buf{buf}[{out_expr}] = buf{input_buf}[ai];\n"
    ));
    out.push_str("  }\n");
}

fn emit_body(out: &mut String, body: &[Inst], indent: &str, debug_bounds: bool) {
    for (j, inst) in body.iter().enumerate() {
        // Emit bounds check before load instructions
        if debug_bounds {
            if let Inst::Load { buf, index } = inst {
                let idx_expr = emit_index(index);
                out.push_str(&format!(
                    "{indent}if ({idx_expr} < 0 || {idx_expr} >= buf{buf}.length) {{ throw new Error(\"OOB: buf{buf}[\" + ({idx_expr}).toString() + \"] len=\" + buf{buf}.length.toString()); }}\n"
                ));
            }
        }
        let expr = emit_inst(inst);
        out.push_str(&format!("{indent}const t{j}: f32 = {expr};\n"));
    }
}

fn emit_inst(inst: &Inst) -> String {
    match inst {
        Inst::Load { buf, index } => {
            let idx = emit_index(index);
            format!("buf{buf}[{idx}]")
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

fn emit_index(index: &Index) -> String {
    match index {
        Index::Flat => "oi".to_string(),
        Index::Strided { parts, offset } => {
            if parts.is_empty() {
                return if *offset > 0 {
                    format!("{offset}")
                } else {
                    "0".to_string()
                };
            }
            let terms: Vec<String> = parts
                .iter()
                .filter(|(_, stride)| *stride != 0)
                .map(|(dim, stride)| {
                    if *stride == 1 {
                        format!("d{dim}")
                    } else {
                        format!("d{dim} * {stride}")
                    }
                })
                .collect();
            let base = if terms.is_empty() {
                "0".to_string()
            } else {
                terms.join(" + ")
            };
            if *offset > 0 {
                format!("{base} + {offset}")
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
}
