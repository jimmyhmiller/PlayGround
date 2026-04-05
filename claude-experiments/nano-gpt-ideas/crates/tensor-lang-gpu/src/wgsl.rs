//! WGSL compute shader generation from Loop IR.
//!
//! Each `Stmt::Loop` becomes one compute shader. Elementwise loops dispatch
//! one thread per output element. Reduce loops dispatch one thread per output
//! element with a sequential inner loop over the reduce dimension.

use std::collections::BTreeSet;
use std::fmt::Write;

use tensor_lang_backend::loop_ir::{Inst, InstRef, Index, ReduceDesc, ReduceOp};
use tensor_lang_graph::Dim;

const WORKGROUP_SIZE: u32 = 256;

/// A generated WGSL shader with metadata about its buffer bindings.
#[derive(Debug, Clone)]
pub struct Shader {
    /// The WGSL source text.
    pub source: String,
    /// Buffer bindings: (binding_index, buf_id, writable).
    pub bindings: Vec<(u32, usize, bool)>,
    /// Whether this shader uses the dims uniform buffer.
    pub uses_dims: bool,
    /// Workgroup size declared in the shader.
    pub workgroup_size: u32,
}

/// Generate a WGSL compute shader for a `Stmt::Loop`.
pub fn emit_loop_shader(
    buf: usize,
    shape: &[Dim],
    reduce: Option<&ReduceDesc>,
    body: &[Inst],
    result: InstRef,
    dim_params: &[String],
) -> Shader {
    let mut src = String::new();

    // Collect all buffer IDs referenced by Load instructions
    let mut read_bufs: BTreeSet<usize> = BTreeSet::new();
    for inst in body {
        if let Inst::Load { buf: b, .. } = inst {
            read_bufs.insert(*b);
        }
    }
    // Output buffer should not appear in read set (it's write-only)
    read_bufs.remove(&buf);

    let uses_dims = !dim_params.is_empty() && shape_or_body_uses_params(shape, reduce, body);

    // Emit dim params struct if needed
    if uses_dims {
        writeln!(src, "struct Dims {{").unwrap();
        for name in dim_params {
            writeln!(src, "    {name} : u32,").unwrap();
        }
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();
    }

    // Emit bindings
    let mut bindings: Vec<(u32, usize, bool)> = Vec::new();
    let mut binding_idx = 0u32;

    // Read buffers
    let read_bufs_vec: Vec<usize> = read_bufs.into_iter().collect();
    for &b in &read_bufs_vec {
        writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read> buf_{b} : array<f32>;").unwrap();
        bindings.push((binding_idx, b, false));
        binding_idx += 1;
    }

    // Output buffer (read_write for reductions that accumulate)
    writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read_write> buf_{buf} : array<f32>;").unwrap();
    bindings.push((binding_idx, buf, true));
    binding_idx += 1;

    // Dims uniform
    if uses_dims {
        writeln!(src, "@group(0) @binding({binding_idx}) var<uniform> dims : Dims;").unwrap();
        binding_idx += 1;
    }
    let _ = binding_idx; // suppress unused warning

    writeln!(src).unwrap();

    // Compute shader entry
    writeln!(src, "@compute @workgroup_size({WORKGROUP_SIZE})").unwrap();
    writeln!(src, "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{").unwrap();
    writeln!(src, "    let oi = gid.x;").unwrap();

    // Compute output size (product of shape)
    let out_size = dim_product_wgsl(shape);
    writeln!(src, "    let out_size = {out_size};").unwrap();
    writeln!(src, "    if (oi >= out_size) {{ return; }}").unwrap();
    writeln!(src).unwrap();

    // Decompose oi into per-dimension variables
    let strides = Dim::strides(shape);
    for (d, dim) in shape.iter().enumerate() {
        let stride = dim_to_wgsl(&strides[d]);
        let size = dim_to_wgsl(dim);
        writeln!(src, "    let d_{d} = (oi / {stride}) % {size};").unwrap();
    }
    writeln!(src).unwrap();

    if let Some(rd) = reduce {
        // Reduce: sequential loop over reduce dimension
        let reduce_dim = shape.len(); // virtual dim index
        let reduce_size = dim_to_wgsl(&rd.size);
        let init = match rd.op {
            ReduceOp::Sum => "0.0".to_string(),
            ReduceOp::Max => "(-3.402823466e+38)".to_string(), // -FLT_MAX
        };
        writeln!(src, "    var acc : f32 = {init};").unwrap();
        writeln!(src, "    for (var d_{reduce_dim} : u32 = 0u; d_{reduce_dim} < {reduce_size}; d_{reduce_dim} = d_{reduce_dim} + 1u) {{").unwrap();

        // Emit body instructions inside reduce loop
        emit_body_instructions(&mut src, body, result, "        ");

        // Accumulate
        let result_var = format!("t_{result}");
        match rd.op {
            ReduceOp::Sum => writeln!(src, "        acc = acc + {result_var};").unwrap(),
            ReduceOp::Max => writeln!(src, "        acc = max(acc, {result_var});").unwrap(),
        }
        writeln!(src, "    }}").unwrap();
        writeln!(src, "    buf_{buf}[oi] = acc;").unwrap();
    } else {
        // Elementwise: compute body and store
        emit_body_instructions(&mut src, body, result, "    ");
        writeln!(src, "    buf_{buf}[oi] = t_{result};").unwrap();
    }

    writeln!(src, "}}").unwrap();

    Shader {
        source: src,
        bindings,
        uses_dims,
        workgroup_size: WORKGROUP_SIZE,
    }
}

/// Generate a WGSL shader for FillConstant.
pub fn emit_fill_constant_shader(buf: usize, value: f32) -> Shader {
    let val_str = format_f32(value);
    let source = format!(
        r#"@group(0) @binding(0) var<storage, read_write> buf_{buf} : array<f32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&buf_{buf})) {{ return; }}
    buf_{buf}[i] = {val_str};
}}
"#
    );
    Shader {
        source,
        bindings: vec![(0, buf, true)],
        uses_dims: false,
        workgroup_size: WORKGROUP_SIZE,
    }
}

/// Generate a WGSL shader for FillArange.
pub fn emit_fill_arange_shader(buf: usize) -> Shader {
    let source = format!(
        r#"@group(0) @binding(0) var<storage, read_write> buf_{buf} : array<f32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&buf_{buf})) {{ return; }}
    buf_{buf}[i] = f32(i);
}}
"#
    );
    Shader {
        source,
        bindings: vec![(0, buf, true)],
        uses_dims: false,
        workgroup_size: WORKGROUP_SIZE,
    }
}

/// Generate a WGSL shader for Pad (zero-fill + copy with offsets).
pub fn emit_pad_shader(
    buf: usize,
    input_buf: usize,
    output_shape: &[Dim],
    input_shape: &[Dim],
    padding: &[(usize, usize)],
    dim_params: &[String],
) -> Shader {
    let mut src = String::new();
    let uses_dims = !dim_params.is_empty()
        && (output_shape.iter().any(|d| d.is_symbolic())
            || input_shape.iter().any(|d| d.is_symbolic()));

    if uses_dims {
        writeln!(src, "struct Dims {{").unwrap();
        for name in dim_params {
            writeln!(src, "    {name} : u32,").unwrap();
        }
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();
    }

    let mut bindings = Vec::new();
    let mut binding_idx = 0u32;

    writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read> buf_{input_buf} : array<f32>;").unwrap();
    bindings.push((binding_idx, input_buf, false));
    binding_idx += 1;

    writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read_write> buf_{buf} : array<f32>;").unwrap();
    bindings.push((binding_idx, buf, true));
    binding_idx += 1;

    if uses_dims {
        writeln!(src, "@group(0) @binding({binding_idx}) var<uniform> dims : Dims;").unwrap();
        binding_idx += 1;
    }
    let _ = binding_idx;

    writeln!(src).unwrap();

    let out_size = dim_product_wgsl(output_shape);
    let out_strides = Dim::strides(output_shape);
    let in_strides = Dim::strides(input_shape);

    writeln!(src, "@compute @workgroup_size({WORKGROUP_SIZE})").unwrap();
    writeln!(src, "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{").unwrap();
    writeln!(src, "    let oi = gid.x;").unwrap();
    writeln!(src, "    let out_size = {out_size};").unwrap();
    writeln!(src, "    if (oi >= out_size) {{ return; }}").unwrap();
    writeln!(src).unwrap();

    // Decompose into output dims
    for (d, dim) in output_shape.iter().enumerate() {
        let stride = dim_to_wgsl(&out_strides[d]);
        let size = dim_to_wgsl(dim);
        writeln!(src, "    let d_{d} = (oi / {stride}) % {size};").unwrap();
    }
    writeln!(src).unwrap();

    // Check if within padded region
    let mut conditions = Vec::new();
    for (d, (lo, _hi)) in padding.iter().enumerate() {
        let in_size = dim_to_wgsl(&input_shape[d]);
        conditions.push(format!("d_{d} >= {lo}u && d_{d} < ({lo}u + {in_size})"));
    }

    if conditions.is_empty() {
        // No padding — straight copy
        writeln!(src, "    buf_{buf}[oi] = buf_{input_buf}[oi];").unwrap();
    } else {
        let cond = conditions.join(" && ");
        writeln!(src, "    if ({cond}) {{").unwrap();

        // Compute input index
        let mut idx_parts = Vec::new();
        for (d, (lo, _hi)) in padding.iter().enumerate() {
            let in_stride = dim_to_wgsl(&in_strides[d]);
            idx_parts.push(format!("(d_{d} - {lo}u) * {in_stride}"));
        }
        let idx_expr = if idx_parts.is_empty() {
            "0u".to_string()
        } else {
            idx_parts.join(" + ")
        };
        writeln!(src, "        buf_{buf}[oi] = buf_{input_buf}[{idx_expr}];").unwrap();
        writeln!(src, "    }} else {{").unwrap();
        writeln!(src, "        buf_{buf}[oi] = 0.0;").unwrap();
        writeln!(src, "    }}").unwrap();
    }

    writeln!(src, "}}").unwrap();

    Shader {
        source: src,
        bindings,
        uses_dims,
        workgroup_size: WORKGROUP_SIZE,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Emit Loop IR instructions as WGSL `let` statements.
fn emit_body_instructions(src: &mut String, body: &[Inst], _result: InstRef, indent: &str) {
    for (i, inst) in body.iter().enumerate() {
        let expr = match inst {
            Inst::Load { buf, index } => {
                let idx = index_to_wgsl(index);
                format!("buf_{buf}[{idx}]")
            }
            Inst::Const(v) => format_f32(*v as f32),
            Inst::DimVar(d) => format!("f32(d_{d})"),
            Inst::Neg(a) => format!("(-t_{a})"),
            Inst::Recip(a) => format!("(1.0 / t_{a})"),
            Inst::Exp2(a) => format!("exp2(t_{a})"),
            Inst::Log2(a) => format!("log2(t_{a})"),
            Inst::Sqrt(a) => format!("sqrt(t_{a})"),
            Inst::Add(a, b) => format!("(t_{a} + t_{b})"),
            Inst::Mul(a, b) => format!("(t_{a} * t_{b})"),
            Inst::Max(a, b) => format!("max(t_{a}, t_{b})"),
            Inst::CmpLt(a, b) => format!("select(0.0, 1.0, t_{a} < t_{b})"),
        };
        writeln!(src, "{indent}let t_{i} = {expr};").unwrap();
    }
}

/// Convert a Dim expression to WGSL (u32 arithmetic).
pub fn dim_to_wgsl(dim: &Dim) -> String {
    match dim {
        Dim::Lit(n) => format!("{n}u"),
        Dim::Param(name) => format!("dims.{name}"),
        Dim::Add(a, b) => format!("({} + {})", dim_to_wgsl(a), dim_to_wgsl(b)),
        Dim::Sub(a, b) => format!("({} - {})", dim_to_wgsl(a), dim_to_wgsl(b)),
        Dim::Mul(a, b) => {
            if a.is_one() {
                return dim_to_wgsl(b);
            }
            if b.is_one() {
                return dim_to_wgsl(a);
            }
            format!("({} * {})", dim_to_wgsl(a), dim_to_wgsl(b))
        }
        Dim::Div(a, b) => format!("({} / {})", dim_to_wgsl(a), dim_to_wgsl(b)),
    }
}

/// Convert a loop IR Index to a WGSL index expression.
fn index_to_wgsl(index: &Index) -> String {
    match index {
        Index::Flat => "oi".to_string(),
        Index::Strided { parts, offset } => {
            let mut terms: Vec<String> = Vec::new();
            if !offset.is_zero() {
                terms.push(dim_to_wgsl(offset));
            }
            for (dim, stride) in parts {
                if stride.is_zero() {
                    continue;
                }
                if stride.is_one() {
                    terms.push(format!("d_{dim}"));
                } else {
                    terms.push(format!("(d_{dim} * {})", dim_to_wgsl(stride)));
                }
            }
            if terms.is_empty() {
                "0u".to_string()
            } else {
                terms.join(" + ")
            }
        }
    }
}

/// Product of a shape as a WGSL expression.
fn dim_product_wgsl(shape: &[Dim]) -> String {
    let product = Dim::product(shape);
    dim_to_wgsl(&product)
}

/// Format an f32 as a WGSL literal.
fn format_f32(v: f32) -> String {
    if v == 0.0 {
        "0.0".to_string()
    } else if v == 1.0 {
        "1.0".to_string()
    } else if v == -1.0 {
        "-1.0".to_string()
    } else if v.fract() == 0.0 {
        format!("{v}.0")
    } else {
        format!("{v}")
    }
}

/// Check if the shape, reduce, or body reference any symbolic dim params.
fn shape_or_body_uses_params(shape: &[Dim], reduce: Option<&ReduceDesc>, body: &[Inst]) -> bool {
    if shape.iter().any(|d| d.is_symbolic()) {
        return true;
    }
    if let Some(rd) = reduce {
        if rd.size.is_symbolic() {
            return true;
        }
    }
    for inst in body {
        if let Inst::Load { index: Index::Strided { parts, offset }, .. } = inst {
            if offset.is_symbolic() {
                return true;
            }
            for (_, stride) in parts {
                if stride.is_symbolic() {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_to_wgsl_lit() {
        assert_eq!(dim_to_wgsl(&Dim::Lit(42)), "42u");
    }

    #[test]
    fn test_dim_to_wgsl_param() {
        assert_eq!(dim_to_wgsl(&Dim::Param("T".into())), "dims.T");
    }

    #[test]
    fn test_dim_to_wgsl_mul_identity() {
        let d = Dim::Mul(Box::new(Dim::Lit(1)), Box::new(Dim::Param("T".into())));
        assert_eq!(dim_to_wgsl(&d), "dims.T");
    }

    #[test]
    fn test_format_f32() {
        assert_eq!(format_f32(0.0), "0.0");
        assert_eq!(format_f32(1.0), "1.0");
        assert_eq!(format_f32(3.0), "3.0");
    }

    #[test]
    fn test_index_flat() {
        assert_eq!(index_to_wgsl(&Index::Flat), "oi");
    }

    #[test]
    fn test_index_strided() {
        let idx = Index::Strided {
            parts: vec![(0, Dim::Lit(10)), (1, Dim::Lit(1))],
            offset: Dim::Lit(0),
        };
        assert_eq!(index_to_wgsl(&idx), "(d_0 * 10u) + d_1");
    }
}
