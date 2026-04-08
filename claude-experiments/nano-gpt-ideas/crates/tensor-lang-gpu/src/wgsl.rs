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

        // K-invariant hoisting: compute which instructions depend on reduce dim
        let depends_on_k = compute_dim_dependence(body, reduce_dim);

        // Emit K-invariant instructions before the loop
        let mut any_hoisted = false;
        for (i, inst) in body.iter().enumerate() {
            if !depends_on_k[i] {
                let expr = inst_to_wgsl(inst);
                writeln!(src, "    let t_{i} = {expr};").unwrap();
                any_hoisted = true;
            }
        }
        if any_hoisted {
            writeln!(src).unwrap();
        }

        writeln!(src, "    var acc : f32 = {init};").unwrap();
        writeln!(src, "    for (var d_{reduce_dim} : u32 = 0u; d_{reduce_dim} < {reduce_size}; d_{reduce_dim} = d_{reduce_dim} + 1u) {{").unwrap();

        // Emit only K-dependent instructions inside the loop
        for (i, inst) in body.iter().enumerate() {
            if depends_on_k[i] {
                let expr = inst_to_wgsl(inst);
                writeln!(src, "        let t_{i} = {expr};").unwrap();
            }
        }

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

/// Information about a detected matmul pattern for tiled shader emission.
/// Handles batch dimension collapsing: effective_M = product(batch_dims) × shape[m_dim].
#[derive(Debug)]
pub struct MatmulInfo {
    pub a_buf: usize,
    pub b_buf: usize,
    /// Effective M dimension (collapsed batch + m_dim)
    pub effective_m: Dim,
    /// N dimension size
    pub n_size: Dim,
    /// K dimension size (reduce)
    pub k_size: Dim,
    /// A row stride: stride to advance one row in the collapsed M space
    pub a_row_stride: Dim,
    /// A K stride
    pub a_k_stride: Dim,
    /// A base offset
    pub a_offset: Dim,
    /// B K stride
    pub b_k_stride: Dim,
    /// B N stride
    pub b_n_stride: Dim,
    /// B base offset (batch part of B, usually 0 for weights)
    pub b_offset: Dim,
    /// Whether B has batch dims (for attention matmuls)
    pub b_has_batch: bool,
    /// B batch parts for computing B offset per batch
    pub b_batch_parts: Vec<(usize, Dim)>,
    /// Output row stride = shape[n_dim]
    pub out_n_stride: Dim,
    /// Batch dimensions (for decomposing wg_id.z when B has batch parts)
    pub batch_dims: Vec<Dim>,
    /// Batch stride parts for B
    pub batch_count: Dim,
}

const TILE_M: u32 = 16;
const TILE_N: u32 = 16;
const TILE_K: u32 = 64;
const MATMUL_WG_SIZE: u32 = TILE_M * TILE_N; // 256 threads per workgroup

/// Try to detect a matmul pattern in a reduce loop body.
/// Returns Some(MatmulInfo) if the body is [Load A, Load B, Mul(0,1)] with Sum reduce.
/// Handles batch dimension collapsing (e.g., GPT-2's [1, T, 1, N] shape).
pub fn detect_matmul(
    body: &[Inst],
    result: InstRef,
    reduce: &ReduceDesc,
    shape: &[Dim],
) -> Option<MatmulInfo> {
    if body.len() != 3 { return None; }
    if reduce.op != ReduceOp::Sum { return None; }
    if shape.len() < 2 { return None; }

    // result must be Mul(0, 1)
    match &body[result] {
        Inst::Mul(a, b) if *a == 0 && *b == 1 => {}
        _ => return None,
    }

    let ndim = shape.len();
    let n_dim = ndim - 1;
    let k_dim = ndim; // reduce dim (virtual)

    let (a_buf, a_parts, a_offset) = match &body[0] {
        Inst::Load { buf, index: Index::Strided { parts, offset } } => (*buf, parts, offset),
        _ => return None,
    };
    let (b_buf, b_parts, b_offset) = match &body[1] {
        Inst::Load { buf, index: Index::Strided { parts, offset } } => (*buf, parts, offset),
        _ => return None,
    };

    // A must reference k_dim, B must reference k_dim and n_dim
    let a_has_k = a_parts.iter().any(|(d, _)| *d == k_dim);
    let b_has_k = b_parts.iter().any(|(d, _)| *d == k_dim);
    let b_has_n = b_parts.iter().any(|(d, _)| *d == n_dim);
    if !a_has_k || !b_has_k || !b_has_n { return None; }

    // Extract A's K stride and find A's row stride (for collapsed batch+M)
    let mut a_k_stride = Dim::Lit(0);
    let mut a_non_k_parts: Vec<(usize, Dim)> = Vec::new();
    for (d, s) in a_parts {
        if *d == k_dim { a_k_stride = s.clone(); }
        else { a_non_k_parts.push((*d, s.clone())); }
    }

    // A row stride: find the rightmost non-zero stride in batch+m dims.
    // For [1, T, 1, N], A has parts for dims 0,1 (batch) with strides.
    // The stride for dim 1 (T) is the row stride.
    let a_row_stride = {
        let mut stride = Dim::Lit(0);
        // Walk dims right to left (m_dim first, then batch dims)
        for d in (0..n_dim).rev() {
            if let Some((_, s)) = a_non_k_parts.iter().find(|(dd, _)| *dd == d) {
                if !s.is_zero() {
                    stride = s.clone();
                    break;
                }
            }
        }
        stride
    };
    if a_row_stride.is_zero() { return None; }

    // Effective M = product of all dims before n_dim
    // For [1, T, 1, N]: effective_M = 1 * T * 1 = T
    let effective_m = Dim::product(&shape[..n_dim]);

    // Extract B's strides
    let mut b_k_stride = Dim::Lit(0);
    let mut b_n_stride = Dim::Lit(0);
    let mut b_batch_parts = Vec::new();
    for (d, s) in b_parts {
        if *d == k_dim { b_k_stride = s.clone(); }
        else if *d == n_dim { b_n_stride = s.clone(); }
        else { b_batch_parts.push((*d, s.clone())); }
    }
    let b_has_batch = !b_batch_parts.is_empty();

    // Batch dimensions for B decomposition
    let batch_dims: Vec<Dim> = shape[..n_dim].to_vec();
    let batch_count = if b_has_batch {
        // For attention matmuls: B has batch parts, dispatch batch_count workgroups
        // batch_count = product(batch_dims that B references)
        Dim::product(&shape[..n_dim])
    } else {
        Dim::Lit(1)
    };

    Some(MatmulInfo {
        a_buf, b_buf,
        effective_m,
        n_size: shape[n_dim].clone(),
        k_size: reduce.size.clone(),
        a_row_stride, a_k_stride,
        a_offset: a_offset.clone(),
        b_k_stride, b_n_stride,
        b_offset: b_offset.clone(),
        b_has_batch,
        b_batch_parts,
        out_n_stride: shape[n_dim].clone(),
        batch_dims,
        batch_count,
    })
}

/// Generate a tiled matmul WGSL shader using workgroup shared memory.
/// Each workgroup computes a TILE_M × TILE_N output tile.
/// Threads cooperatively load TILE_M×TILE_K of A and TILE_K×TILE_N of B
/// into shared memory, then compute partial products.
///
/// Uses collapsed batch dimensions: the output is treated as [effective_M, N]
/// where effective_M = product(batch_dims) × shape[m_dim].
/// For GPT-2's [1, T, 1, N], effective_M = T.
pub fn emit_tiled_matmul_shader(
    buf: usize,
    shape: &[Dim],
    reduce: &ReduceDesc,
    info: &MatmulInfo,
    dim_params: &[String],
) -> Shader {
    let mut src = String::new();

    let uses_dims = !dim_params.is_empty() && {
        shape.iter().any(|d| d.is_symbolic()) || reduce.size.is_symbolic()
    };

    if uses_dims {
        writeln!(src, "struct Dims {{").unwrap();
        for name in dim_params {
            writeln!(src, "    {name} : u32,").unwrap();
        }
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();
    }

    let mut bindings: Vec<(u32, usize, bool)> = Vec::new();
    let mut binding_idx = 0u32;

    // A buffer (read)
    writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read> buf_{} : array<f32>;", info.a_buf).unwrap();
    bindings.push((binding_idx, info.a_buf, false));
    binding_idx += 1;

    // B buffer (read) — only add if different from A
    if info.b_buf != info.a_buf {
        writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read> buf_{} : array<f32>;", info.b_buf).unwrap();
        bindings.push((binding_idx, info.b_buf, false));
        binding_idx += 1;
    }

    // Output buffer (read_write)
    writeln!(src, "@group(0) @binding({binding_idx}) var<storage, read_write> buf_{buf} : array<f32>;").unwrap();
    bindings.push((binding_idx, buf, true));
    binding_idx += 1;

    if uses_dims {
        writeln!(src, "@group(0) @binding({binding_idx}) var<uniform> dims : Dims;").unwrap();
        binding_idx += 1;
    }
    let _ = binding_idx;

    writeln!(src).unwrap();

    // Shared memory tiles
    writeln!(src, "var<workgroup> tile_a : array<f32, {}>; // {}x{}", TILE_M * TILE_K, TILE_M, TILE_K).unwrap();
    writeln!(src, "var<workgroup> tile_b : array<f32, {}>; // {}x{}", TILE_K * TILE_N, TILE_K, TILE_N).unwrap();
    writeln!(src).unwrap();

    let eff_m = dim_to_wgsl(&info.effective_m);
    let n_size = dim_to_wgsl(&info.n_size);
    let k_size = dim_to_wgsl(&info.k_size);
    let a_row_stride = dim_to_wgsl(&info.a_row_stride);
    let a_k_stride = dim_to_wgsl(&info.a_k_stride);
    let b_k_stride = dim_to_wgsl(&info.b_k_stride);
    let b_n_stride = dim_to_wgsl(&info.b_n_stride);

    // Dispatch: (ceil(N/TILE_N), ceil(effective_M/TILE_M), batch_for_B)
    // When B has no batch dims (linear projections), wg_id.z is always 0.
    // When B has batch dims (attention), wg_id.z indexes the batch.
    writeln!(src, "@compute @workgroup_size({TILE_N}, {TILE_M}, 1)").unwrap();
    writeln!(src, "fn main(").unwrap();
    writeln!(src, "    @builtin(workgroup_id) wg_id : vec3<u32>,").unwrap();
    writeln!(src, "    @builtin(local_invocation_id) lid : vec3<u32>,").unwrap();
    writeln!(src, ") {{").unwrap();

    writeln!(src, "    let row = lid.y;").unwrap();
    writeln!(src, "    let col = lid.x;").unwrap();
    writeln!(src, "    let local_idx = row * {TILE_N}u + col;").unwrap();
    writeln!(src).unwrap();

    // Global M,N coordinates
    writeln!(src, "    let global_m = wg_id.y * {TILE_M}u + row;").unwrap();
    writeln!(src, "    let global_n = wg_id.x * {TILE_N}u + col;").unwrap();
    writeln!(src).unwrap();

    // A base: offset + global_m is the collapsed row index
    // A[global_m, k] = A_base + global_m * a_row_stride + k * a_k_stride
    let a_offset = dim_to_wgsl(&info.a_offset);
    writeln!(src, "    let a_base = {a_offset};").unwrap();

    // B base: for weights (no batch), just the offset.
    // For attention matmuls with batch dims, decompose wg_id.z.
    if info.b_has_batch {
        let ndim = shape.len();
        let batch_dims_shape = &shape[..ndim - 1]; // all dims before N
        writeln!(src, "    let batch_idx = wg_id.z;").unwrap();
        // Decompose batch_idx into dimension variables
        let batch_strides = Dim::strides(batch_dims_shape);
        for (d, dim) in batch_dims_shape.iter().enumerate() {
            let stride = dim_to_wgsl(&batch_strides[d]);
            let size = dim_to_wgsl(dim);
            writeln!(src, "    let d_{d} = (batch_idx / {stride}) % {size};").unwrap();
        }
        // Compute B offset from batch parts
        let mut b_terms = Vec::new();
        if !info.b_offset.is_zero() {
            b_terms.push(dim_to_wgsl(&info.b_offset));
        }
        for (d, s) in &info.b_batch_parts {
            if s.is_one() {
                b_terms.push(format!("d_{d}"));
            } else {
                b_terms.push(format!("(d_{d} * {})", dim_to_wgsl(s)));
            }
        }
        let b_base_expr = if b_terms.is_empty() { "0u".to_string() } else { b_terms.join(" + ") };
        writeln!(src, "    let b_base = {b_base_expr};").unwrap();
    } else {
        let b_offset = dim_to_wgsl(&info.b_offset);
        writeln!(src, "    let b_base = {b_offset};").unwrap();
    }
    writeln!(src).unwrap();

    // Accumulator
    writeln!(src, "    var acc : f32 = 0.0;").unwrap();
    writeln!(src).unwrap();

    // Tiled K loop
    writeln!(src, "    let k_tiles = ({k_size} + {TILE_K}u - 1u) / {TILE_K}u;").unwrap();
    writeln!(src, "    for (var kt : u32 = 0u; kt < k_tiles; kt = kt + 1u) {{").unwrap();
    writeln!(src, "        let k_base = kt * {TILE_K}u;").unwrap();
    writeln!(src).unwrap();

    // Cooperative load of A tile
    let a_tile_size = TILE_M * TILE_K;
    let loads_per_thread_a = (a_tile_size + MATMUL_WG_SIZE - 1) / MATMUL_WG_SIZE;
    writeln!(src, "        for (var li : u32 = 0u; li < {loads_per_thread_a}u; li = li + 1u) {{").unwrap();
    writeln!(src, "            let idx = local_idx + li * {MATMUL_WG_SIZE}u;").unwrap();
    writeln!(src, "            if (idx < {}u) {{", a_tile_size).unwrap();
    writeln!(src, "                let tile_r = idx / {TILE_K}u;").unwrap();
    writeln!(src, "                let tile_c = idx % {TILE_K}u;").unwrap();
    writeln!(src, "                let gm = wg_id.y * {TILE_M}u + tile_r;").unwrap();
    writeln!(src, "                let gk = k_base + tile_c;").unwrap();
    writeln!(src, "                if (gm < {eff_m} && gk < {k_size}) {{").unwrap();
    writeln!(src, "                    tile_a[idx] = buf_{}[a_base + gm * {a_row_stride} + gk * {a_k_stride}];", info.a_buf).unwrap();
    writeln!(src, "                }} else {{").unwrap();
    writeln!(src, "                    tile_a[idx] = 0.0;").unwrap();
    writeln!(src, "                }}").unwrap();
    writeln!(src, "            }}").unwrap();
    writeln!(src, "        }}").unwrap();
    writeln!(src).unwrap();

    // Cooperative load of B tile
    let b_tile_size = TILE_K * TILE_N;
    let loads_per_thread_b = (b_tile_size + MATMUL_WG_SIZE - 1) / MATMUL_WG_SIZE;
    writeln!(src, "        for (var li : u32 = 0u; li < {loads_per_thread_b}u; li = li + 1u) {{").unwrap();
    writeln!(src, "            let idx = local_idx + li * {MATMUL_WG_SIZE}u;").unwrap();
    writeln!(src, "            if (idx < {}u) {{", b_tile_size).unwrap();
    writeln!(src, "                let tile_r = idx / {TILE_N}u;").unwrap();
    writeln!(src, "                let tile_c = idx % {TILE_N}u;").unwrap();
    writeln!(src, "                let gk = k_base + tile_r;").unwrap();
    writeln!(src, "                let gn = wg_id.x * {TILE_N}u + tile_c;").unwrap();
    writeln!(src, "                if (gk < {k_size} && gn < {n_size}) {{").unwrap();
    writeln!(src, "                    tile_b[idx] = buf_{}[b_base + gk * {b_k_stride} + gn * {b_n_stride}];", info.b_buf).unwrap();
    writeln!(src, "                }} else {{").unwrap();
    writeln!(src, "                    tile_b[idx] = 0.0;").unwrap();
    writeln!(src, "                }}").unwrap();
    writeln!(src, "            }}").unwrap();
    writeln!(src, "        }}").unwrap();
    writeln!(src).unwrap();

    writeln!(src, "        workgroupBarrier();").unwrap();
    writeln!(src).unwrap();

    // Compute: each thread accumulates one output element
    writeln!(src, "        for (var ki : u32 = 0u; ki < {TILE_K}u; ki = ki + 1u) {{").unwrap();
    writeln!(src, "            acc = acc + tile_a[row * {TILE_K}u + ki] * tile_b[ki * {TILE_N}u + col];").unwrap();
    writeln!(src, "        }}").unwrap();
    writeln!(src).unwrap();

    writeln!(src, "        workgroupBarrier();").unwrap();
    writeln!(src, "    }}").unwrap();
    writeln!(src).unwrap();

    // Write result — output is flat [effective_M × N]
    writeln!(src, "    if (global_m < {eff_m} && global_n < {n_size}) {{").unwrap();
    let out_n = dim_to_wgsl(&info.out_n_stride);
    writeln!(src, "        buf_{buf}[global_m * {out_n} + global_n] = acc;").unwrap();
    writeln!(src, "    }}").unwrap();
    writeln!(src, "}}").unwrap();

    Shader {
        source: src,
        bindings,
        uses_dims,
        workgroup_size: MATMUL_WG_SIZE,
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
        let expr = inst_to_wgsl(inst);
        writeln!(src, "{indent}let t_{i} = {expr};").unwrap();
    }
}

/// Convert a single Loop IR instruction to a WGSL expression string.
fn inst_to_wgsl(inst: &Inst) -> String {
    match inst {
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
    }
}

/// Compute which body instructions depend on a given dimension variable.
fn compute_dim_dependence(body: &[Inst], dim: usize) -> Vec<bool> {
    let mut dep = vec![false; body.len()];
    for (j, inst) in body.iter().enumerate() {
        dep[j] = match inst {
            Inst::Load { index: Index::Strided { parts, .. }, .. } => {
                parts.iter().any(|(d, _)| *d == dim)
            }
            Inst::Load { index: Index::Flat, .. } => true,
            Inst::Const(_) => false,
            Inst::DimVar(d) => *d == dim,
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a) | Inst::Log2(a) | Inst::Sqrt(a) => dep[*a],
            Inst::Add(a, b) | Inst::Mul(a, b) | Inst::Max(a, b) | Inst::CmpLt(a, b) => dep[*a] || dep[*b],
        };
    }
    dep
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
