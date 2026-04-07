//! Loop IR: a flat, fused representation lowered from the dataflow graph.
//!
//! The lowering pass applies fusion by inlining single-consumer nodes into
//! their consumer's loop body. Multi-consumer nodes and reduce outputs are
//! materialized (get their own buffer).

use tensor_lang_graph::{Dim, Graph, NodeId, Op};

// ---------------------------------------------------------------------------
// IR types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Stmt {
    /// Allocate a buffer. Source nodes (Input, Constant, Arange) use this.
    Alloc { buf: usize, size: Dim },
    /// Fill a 1-element buffer with a constant.
    Fill { buf: usize, value: f64 },
    /// Fill a buffer with 0, 1, 2, ..., size-1.
    FillArange { buf: usize, size: Dim },
    /// A loop that computes a materialized buffer.
    Loop {
        buf: usize,
        shape: Vec<Dim>,
        reduce: Option<ReduceDesc>,
        body: Vec<Inst>,
        /// Which instruction in `body` produces the output value.
        result: InstRef,
        /// Optional tiling configuration for cache-friendly execution.
        tile: Option<TileConfig>,
    },
    /// Pad: zero-fill output, then copy input with offsets.
    /// Uses the original unfused approach since pad has special structure.
    Pad {
        buf: usize,
        input_buf: usize,
        output_shape: Vec<Dim>,
        input_shape: Vec<Dim>,
        padding: Vec<(usize, usize)>,
    },
}

#[derive(Debug, Clone)]
pub struct ReduceDesc {
    pub axis: usize,
    pub size: Dim,
    pub op: ReduceOp,
}

#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Tile size for each output dimension + the reduce dimension (last entry).
    pub tiles: Vec<Dim>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
}

pub type InstRef = usize;

#[derive(Debug, Clone)]
pub enum Inst {
    Load { buf: usize, index: Index },
    Const(f64),
    DimVar(usize),
    // Unary
    Neg(InstRef),
    Recip(InstRef),
    Exp2(InstRef),
    Log2(InstRef),
    Sqrt(InstRef),
    // Binary
    Add(InstRef, InstRef),
    Mul(InstRef, InstRef),
    Max(InstRef, InstRef),
    CmpLt(InstRef, InstRef),
}

/// How to compute a flat buffer index from loop dimension variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Index {
    /// Flat index equals the output loop's flat index.
    Flat,
    /// Index = offset + sum of (d_{dim} * stride) for each entry.
    /// First element is the dimension variable index (always concrete),
    /// second is the stride (may be symbolic).
    Strided {
        parts: Vec<(usize, Dim)>,
        offset: Dim,
    },
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

/// Lower a dataflow graph into the loop IR with fusion applied.
pub fn lower(graph: &Graph) -> Vec<Stmt> {
    lower_with_outputs(graph, &[])
}

/// Lower the graph, ensuring all `extra_outputs` nodes are materialized.
pub fn lower_with_outputs(graph: &Graph, extra_outputs: &[NodeId]) -> Vec<Stmt> {
    let n = graph.nodes.len();
    if n == 0 {
        return vec![];
    }

    // Step 1: compute consumer counts
    let mut consumers = vec![0usize; n];
    for node in &graph.nodes {
        for inp in &node.inputs {
            consumers[inp.0] += 1;
        }
    }

    // Step 2: determine materialization set
    let last = n - 1;
    let mut materialized: Vec<bool> = (0..n)
        .map(|i| {
            let node = &graph.nodes[i];
            match &node.op {
                // Sources always materialize
                Op::Input { .. } | Op::Constant(_) | Op::Arange { .. } => true,
                // Pad needs special zero-fill
                Op::Pad { .. } => true,
                // Shrink needs offset support not yet in Index — materialize for now
                Op::Shrink { .. } => true,
                // Multi-consumer: must materialize
                _ if consumers[i] > 1 => true,
                // Final output or requested multi-outputs
                _ if i == last || extra_outputs.contains(&NodeId(i)) => true,
                // Reduce outputs: downstream iterates different shape
                Op::ReduceSum { .. } | Op::ReduceMax { .. } => true,
                // Reshapes that merge/split dims (not pure 1-dim change) must
                // materialize to prevent index map mismatch in downstream ops.
                Op::Reshape { shape } => {
                    let input_shape = &graph.nodes[node.inputs[0].0].shape;
                    input_shape.len() != shape.len()
                        && !is_pure_1dim_change(input_shape, shape)
                }
                _ => false,
            }
        })
        .collect();

    // Also materialize the input of any dim-merge/split reshape, so the
    // reshape loop body is just a flat copy (no need to trace through ops
    // with mismatched dimension counts).
    for i in 0..n {
        let node = &graph.nodes[i];
        if let Op::Reshape { shape } = &node.op {
            let input_id = node.inputs[0].0;
            let input_shape = &graph.nodes[input_id].shape;
            if materialized[i]
                && input_shape.len() != shape.len()
                && !is_pure_1dim_change(input_shape, shape)
            {
                materialized[input_id] = true;
            }
        }
    }

    // Materialize the input of any Pad node, since Pad emits a special
    // copy statement that reads directly from the input buffer.
    for i in 0..n {
        let node = &graph.nodes[i];
        if let Op::Pad { .. } = &node.op {
            if materialized[i] {
                materialized[node.inputs[0].0] = true;
            }
        }
    }

    // Step 3: emit statements
    let mut stmts = Vec::new();

    for (i, node) in graph.nodes.iter().enumerate() {
        match &node.op {
            Op::Input { .. } => {
                let size = shape_size(&node.shape);
                stmts.push(Stmt::Alloc { buf: i, size });
            }
            Op::Constant(v) => {
                stmts.push(Stmt::Alloc { buf: i, size: Dim::Lit(1) });
                stmts.push(Stmt::Fill { buf: i, value: *v });
            }
            Op::Arange { size } => {
                stmts.push(Stmt::Alloc { buf: i, size: size.clone() });
                stmts.push(Stmt::FillArange { buf: i, size: size.clone() });
            }
            Op::Pad { padding } if materialized[i] => {
                let size = shape_size(&node.shape);
                let input_id = node.inputs[0].0;
                stmts.push(Stmt::Alloc { buf: i, size });
                stmts.push(Stmt::Pad {
                    buf: i,
                    input_buf: input_id,
                    output_shape: node.shape.clone(),
                    input_shape: graph.nodes[input_id].shape.clone(),
                    padding: padding.clone(),
                });
            }
            _ if materialized[i] => {
                let size = shape_size(&node.shape);
                stmts.push(Stmt::Alloc { buf: i, size });

                // Build the loop
                let mut ctx = LowerCtx {
                    graph,
                    materialized: &materialized,
                    body: Vec::new(),
                    reduce: None,
                };
                let result = ctx.emit_node(i, &node.shape);
                stmts.push(Stmt::Loop {
                    buf: i,
                    shape: node.shape.clone(),
                    reduce: ctx.reduce,
                    body: ctx.body,
                    result,
                    tile: None,
                });
            }
            _ => {
                // Single-consumer, non-materialized: will be inlined
            }
        }
    }

    stmts
}

fn shape_size(shape: &[Dim]) -> Dim {
    Dim::product(shape)
}

/// Check if the reshape between two shapes is purely inserting/removing size-1 dims.
/// Returns true if removing all 1s from both shapes yields the same sequence.
fn is_pure_1dim_change(a: &[Dim], b: &[Dim]) -> bool {
    let a_no1: Vec<&Dim> = a.iter().filter(|d| !d.is_one()).collect();
    let b_no1: Vec<&Dim> = b.iter().filter(|d| !d.is_one()).collect();
    a_no1 == b_no1
}

// ---------------------------------------------------------------------------
// Lowering context: builds the flat instruction list for a single loop
// ---------------------------------------------------------------------------

struct LowerCtx<'a> {
    graph: &'a Graph,
    materialized: &'a [bool],
    body: Vec<Inst>,
    reduce: Option<ReduceDesc>,
}

/// Describes how the current loop's dimension variables map to a buffer's
/// index space. We thread this through movement ops.
#[derive(Clone, Debug)]
struct IndexMap {
    /// The shape of the iteration space these dims refer to.
    iter_shape: Vec<Dim>,
    /// For each dimension of the target buffer, how to compute it:
    /// Some((loop_dim, stride)) or None (broadcast / collapsed).
    entries: Vec<IndexEntry>,
    /// Constant offset added to the computed index (used by shrink).
    offset: Dim,
}

#[derive(Clone, Debug)]
enum IndexEntry {
    /// This buffer dimension maps to loop dimension `dim` with given stride.
    Dim { dim: usize, stride: Dim },
    /// Broadcast: this dimension is size 1, contributes 0 to the index.
    Broadcast,
}

impl IndexMap {
    /// Create an identity mapping: buffer dims match loop dims.
    fn identity(shape: &[Dim]) -> Self {
        let strides_vec = Dim::strides(shape);
        let entries = (0..shape.len())
            .map(|d| IndexEntry::Dim {
                dim: d,
                stride: strides_vec[d].clone(),
            })
            .collect();
        IndexMap {
            iter_shape: shape.to_vec(),
            entries,
            offset: Dim::Lit(0),
        }
    }

    /// Convert to an Index for codegen.
    fn to_index(&self) -> Index {
        let parts: Vec<(usize, Dim)> = self
            .entries
            .iter()
            .filter_map(|e| match e {
                IndexEntry::Dim { dim, stride } => Some((*dim, stride.clone())),
                IndexEntry::Broadcast => None,
            })
            .collect();
        if parts.is_empty() {
            Index::Strided {
                parts: vec![(0, Dim::Lit(0))],
                offset: self.offset.clone(),
            }
        } else {
            Index::Strided {
                parts,
                offset: self.offset.clone(),
            }
        }
    }

    /// Adjust for an expand: where the source shape has 1s, mark as broadcast.
    /// Handles different ndims by right-aligning (broadcasting pads on the left).
    fn through_expand(&self, expanded_shape: &[Dim], source_shape: &[Dim]) -> IndexMap {
        let source_strides = Dim::strides(source_shape);
        let exp_ndim = expanded_shape.len();
        let src_ndim = source_shape.len();
        let dim_offset = exp_ndim - src_ndim;

        let entries = (0..src_ndim)
            .map(|d| {
                let exp_d = d + dim_offset; // align right
                if source_shape[d].is_one() && !expanded_shape[exp_d].is_one() {
                    IndexEntry::Broadcast
                } else {
                    // Map to the expanded shape's loop dim, but use source stride
                    match &self.entries[exp_d] {
                        IndexEntry::Dim { dim, .. } => IndexEntry::Dim {
                            dim: *dim,
                            stride: source_strides[d].clone(),
                        },
                        IndexEntry::Broadcast => IndexEntry::Broadcast,
                    }
                }
            })
            .collect();
        IndexMap {
            iter_shape: self.iter_shape.clone(),
            entries,
            offset: self.offset.clone(),
        }
    }

    /// Adjust for a reshape that only inserts/removes size-1 dims.
    /// Preserves original strides (entries are copied as-is, not re-strided).
    /// Requires: is_pure_1dim_change(old_shape, new_shape) && entries.len() == old_shape.len()
    fn through_reshape_ndim(&self, old_shape: &[Dim], new_shape: &[Dim]) -> IndexMap {
        let mut new_entries = Vec::new();
        let mut oi = 0; // index into old_shape / self.entries
        let mut ni = 0; // index into new_shape

        while ni < new_shape.len() {
            if oi < old_shape.len() && old_shape[oi] == new_shape[ni] {
                // Sizes match — transfer entry as-is (preserve stride)
                new_entries.push(self.entries[oi].clone());
                oi += 1;
                ni += 1;
            } else if oi < old_shape.len() && old_shape[oi].is_one() {
                // Old has a 1-dim not in new — skip it
                oi += 1;
            } else if new_shape[ni].is_one() {
                // New has a 1-dim not in old — insert Broadcast
                new_entries.push(IndexEntry::Broadcast);
                ni += 1;
            } else {
                unreachable!(
                    "through_reshape_ndim called on non-pure-1dim reshape: old={:?} new={:?}",
                    old_shape, new_shape
                );
            }
        }

        IndexMap {
            iter_shape: self.iter_shape.clone(),
            entries: new_entries,
            offset: self.offset.clone(),
        }
    }

    /// Adjust for a permute: map output loop vars to input buffer dims.
    /// permute(x, order) means output dim d comes from input dim order[d].
    /// To index the input buffer: for each input dim j, find which output dim d
    /// maps to it (inv_order[j] = d), then use that loop variable with input strides.
    fn through_permute(&self, order: &[usize], input_shape: &[Dim]) -> IndexMap {
        let input_strides = Dim::strides(input_shape);
        // Build inverse permutation: inv[order[d]] = d
        let mut inv = vec![0; order.len()];
        for (d, &o) in order.iter().enumerate() {
            inv[o] = d;
        }
        // For each input dimension j, use the loop variable from output dim inv[j]
        let entries = (0..input_shape.len())
            .map(|j| {
                let out_d = inv[j];
                match &self.entries[out_d] {
                    IndexEntry::Dim { dim, .. } => IndexEntry::Dim {
                        dim: *dim,
                        stride: input_strides[j].clone(),
                    },
                    IndexEntry::Broadcast => IndexEntry::Broadcast,
                }
            })
            .collect();
        IndexMap {
            iter_shape: self.iter_shape.clone(),
            entries,
            offset: self.offset.clone(),
        }
    }

    /// Adjust for a shrink: use source strides and add lo offsets.
    fn through_shrink(&self, bounds: &[(Dim, Dim)], source_shape: &[Dim]) -> IndexMap {
        let source_strides = Dim::strides(source_shape);
        // Compute constant offset: sum of lo_d * source_stride_d
        let mut lo_offset = self.offset.clone();
        for (d, (lo, _)) in bounds.iter().enumerate() {
            if !lo.is_zero() {
                let term = Dim::Mul(Box::new(lo.clone()), Box::new(source_strides[d].clone())).simplify();
                lo_offset = Dim::Add(Box::new(lo_offset), Box::new(term)).simplify();
            }
        }
        // Replace strides with source strides
        let entries = bounds
            .iter()
            .enumerate()
            .map(|(d, (_lo, _hi))| match &self.entries[d] {
                IndexEntry::Dim { dim, .. } => IndexEntry::Dim {
                    dim: *dim,
                    stride: source_strides[d].clone(),
                },
                IndexEntry::Broadcast => IndexEntry::Broadcast,
            })
            .collect();
        IndexMap {
            iter_shape: self.iter_shape.clone(),
            entries,
            offset: lo_offset,
        }
    }
}

impl<'a> LowerCtx<'a> {
    fn push(&mut self, inst: Inst) -> InstRef {
        let idx = self.body.len();
        self.body.push(inst);
        idx
    }

    /// Emit instructions for node `i`, returning the InstRef for its result.
    /// `iter_shape` is the iteration space of the enclosing loop.
    fn emit_node(&mut self, i: usize, iter_shape: &[Dim]) -> InstRef {
        let idx_map = IndexMap::identity(iter_shape);
        self.emit_node_with_index(i, &idx_map)
    }

    /// Emit instructions for node `i` with a given index mapping.
    fn emit_node_with_index(&mut self, i: usize, idx_map: &IndexMap) -> InstRef {
        let node = &self.graph.nodes[i];

        // If this node is materialized and is not the node we're currently
        // building a loop for (i.e., it's an input to our loop), emit a Load.
        // We detect "input to our loop" by checking if we're being called
        // recursively (body is non-empty or we're in a recursive call for a
        // non-materialized node).
        // Actually, the simpler check: if it's materialized AND it's a source
        // or we've already decided to materialize it, emit Load.
        // The top-level call is the materialization point itself; we need to
        // recurse into its op, not load from it.

        match &node.op {
            // Sources are always loaded from
            Op::Input { .. } | Op::Constant(_) | Op::Arange { .. } => {
                let index = self.compute_load_index(i, idx_map);
                self.push(Inst::Load { buf: i, index })
            }

            // Reduce: set up the reduce descriptor and recurse into producer
            Op::ReduceSum { axis } | Op::ReduceMax { axis } => {
                let axis = *axis;
                let input_id = node.inputs[0].0;
                let input_shape = &self.graph.nodes[input_id].shape;
                let reduce_size = input_shape[axis].clone();

                let op = match &node.op {
                    Op::ReduceSum { .. } => ReduceOp::Sum,
                    Op::ReduceMax { .. } => ReduceOp::Max,
                    _ => unreachable!(),
                };

                // The reduce output shape has the reduced axis = 1.
                // The iteration space for the inner body is the input shape
                // (including the axis being reduced).
                // We need a special "reduce dim" that the loop will iterate over.
                let reduce_dim_index = input_shape.len(); // virtual dim after all real dims

                self.reduce = Some(ReduceDesc {
                    axis,
                    size: reduce_size,
                    op,
                });

                // Build index map for the input to the reduce.
                // The output loop iterates over the reduce's output shape (axis dim = 1).
                // We need to add the reduce axis dimension.
                let input_strides = Dim::strides(input_shape);
                let entries: Vec<IndexEntry> = (0..input_shape.len())
                    .map(|d| {
                        if d == axis {
                            // This dim is iterated by the reduce inner loop
                            IndexEntry::Dim {
                                dim: reduce_dim_index,
                                stride: input_strides[d].clone(),
                            }
                        } else {
                            // Map to the corresponding output dim.
                            // We need to figure out which output dim this is.
                            // The output shape is the same as input but with axis dim = 1.
                            // So dims before axis keep their index, dims after axis too.
                            IndexEntry::Dim {
                                dim: d,
                                stride: input_strides[d].clone(),
                            }
                        }
                    })
                    .collect();

                let inner_map = IndexMap {
                    iter_shape: idx_map.iter_shape.clone(),
                    entries,
                    offset: Dim::Lit(0),
                };

                self.emit_recursive(input_id, &inner_map)
            }

            // Elementwise unary
            Op::Neg => {
                let a = self.emit_recursive(node.inputs[0].0, idx_map);
                self.push(Inst::Neg(a))
            }
            Op::Recip => {
                let a = self.emit_recursive(node.inputs[0].0, idx_map);
                self.push(Inst::Recip(a))
            }
            Op::Exp2 => {
                let a = self.emit_recursive(node.inputs[0].0, idx_map);
                self.push(Inst::Exp2(a))
            }
            Op::Log2 => {
                let a = self.emit_recursive(node.inputs[0].0, idx_map);
                self.push(Inst::Log2(a))
            }
            Op::Sqrt => {
                let a = self.emit_recursive(node.inputs[0].0, idx_map);
                self.push(Inst::Sqrt(a))
            }

            // Elementwise binary
            Op::Add => {
                let a_id = node.inputs[0].0;
                let b_id = node.inputs[1].0;
                let out_shape = &node.shape;
                let a_shape = &self.graph.nodes[a_id].shape;
                let b_shape = &self.graph.nodes[b_id].shape;
                let a_map = self.broadcast_map(idx_map, out_shape, a_shape);
                let b_map = self.broadcast_map(idx_map, out_shape, b_shape);
                let a = self.emit_recursive(a_id, &a_map);
                let b = self.emit_recursive(b_id, &b_map);
                self.push(Inst::Add(a, b))
            }
            Op::Mul => {
                let a_id = node.inputs[0].0;
                let b_id = node.inputs[1].0;
                let out_shape = &node.shape;
                let a_shape = &self.graph.nodes[a_id].shape;
                let b_shape = &self.graph.nodes[b_id].shape;
                let a_map = self.broadcast_map(idx_map, out_shape, a_shape);
                let b_map = self.broadcast_map(idx_map, out_shape, b_shape);
                let a = self.emit_recursive(a_id, &a_map);
                let b = self.emit_recursive(b_id, &b_map);
                self.push(Inst::Mul(a, b))
            }
            Op::Max => {
                let a_id = node.inputs[0].0;
                let b_id = node.inputs[1].0;
                let out_shape = &node.shape;
                let a_shape = &self.graph.nodes[a_id].shape;
                let b_shape = &self.graph.nodes[b_id].shape;
                let a_map = self.broadcast_map(idx_map, out_shape, a_shape);
                let b_map = self.broadcast_map(idx_map, out_shape, b_shape);
                let a = self.emit_recursive(a_id, &a_map);
                let b = self.emit_recursive(b_id, &b_map);
                self.push(Inst::Max(a, b))
            }
            Op::CmpLt => {
                let a_id = node.inputs[0].0;
                let b_id = node.inputs[1].0;
                let out_shape = &node.shape;
                let a_shape = &self.graph.nodes[a_id].shape;
                let b_shape = &self.graph.nodes[b_id].shape;
                let a_map = self.broadcast_map(idx_map, out_shape, a_shape);
                let b_map = self.broadcast_map(idx_map, out_shape, b_shape);
                let a = self.emit_recursive(a_id, &a_map);
                let b = self.emit_recursive(b_id, &b_map);
                self.push(Inst::CmpLt(a, b))
            }

            // Movement ops: adjust index mapping and recurse
            Op::Reshape { .. } => {
                let input_id = node.inputs[0].0;
                let input_shape = &self.graph.nodes[input_id].shape;
                let output_shape = &node.shape;

                if input_shape.len() != output_shape.len()
                    && is_pure_1dim_change(output_shape, input_shape)
                    && idx_map.entries.len() == output_shape.len()
                {
                    // Reshape only inserts/removes size-1 dims — adjust entries
                    // so subsequent ops (permute, expand) see correct dim count.
                    let new_map = idx_map.through_reshape_ndim(output_shape, input_shape);
                    self.emit_recursive(input_id, &new_map)
                } else {
                    // Same ndim or dim merge/split — pass through.
                    // The flat index is preserved, so entries still compute
                    // the correct flat index for eventual buffer loads.
                    self.emit_recursive(input_id, idx_map)
                }
            }
            Op::Expand { shape } => {
                let input_id = node.inputs[0].0;
                let input_shape = &self.graph.nodes[input_id].shape;
                let new_map = idx_map.through_expand(shape, input_shape);
                self.emit_recursive(input_id, &new_map)
            }
            Op::Permute { order } => {
                let input_id = node.inputs[0].0;
                let input_shape = &self.graph.nodes[input_id].shape;
                let new_map = idx_map.through_permute(order, input_shape);
                self.emit_recursive(input_id, &new_map)
            }
            Op::Shrink { bounds } => {
                let input_id = node.inputs[0].0;
                let input_shape = &self.graph.nodes[input_id].shape;
                let new_map = idx_map.through_shrink(bounds, input_shape);
                self.emit_recursive(input_id, &new_map)
            }
            Op::Pad { .. } => {
                // Pad is materialized, so this should only be reached as a Load
                let index = self.compute_load_index(i, idx_map);
                self.push(Inst::Load { buf: i, index })
            }
        }
    }

    /// Either inline a node (if not materialized) or load from its buffer.
    fn emit_recursive(&mut self, i: usize, idx_map: &IndexMap) -> InstRef {
        if self.materialized[i] {
            // This node has its own buffer — emit a load
            let index = self.compute_load_index(i, idx_map);
            self.push(Inst::Load { buf: i, index })
        } else {
            // Inline this node's computation
            self.emit_node_with_index(i, idx_map)
        }
    }

    /// Compute the load index for a materialized buffer.
    fn compute_load_index(&self, buf_id: usize, idx_map: &IndexMap) -> Index {
        let buf_shape = &self.graph.nodes[buf_id].shape;
        if buf_shape.is_empty() {
            // Scalar
            return Index::Strided {
                parts: vec![],
                offset: idx_map.offset.clone(),
            };
        }

        // Check if the idx_map's iteration shape matches the buffer shape
        // If so, we can use Flat (only if no offset)
        if idx_map.offset.is_zero()
            && idx_map.iter_shape == *buf_shape
            && self.reduce.is_none()
        {
            let all_identity = idx_map.entries.iter().enumerate().all(|(d, e)| {
                matches!(e, IndexEntry::Dim { dim, .. } if *dim == d)
            });
            if all_identity {
                return Index::Flat;
            }
        }

        idx_map.to_index()
    }

    /// Create a broadcast-adjusted index map for a binary op's input.
    fn broadcast_map(
        &self,
        parent_map: &IndexMap,
        output_shape: &[Dim],
        input_shape: &[Dim],
    ) -> IndexMap {
        if input_shape.is_empty() {
            // Scalar input — always index 0
            return IndexMap {
                iter_shape: parent_map.iter_shape.clone(),
                entries: vec![],
                offset: Dim::Lit(0),
            };
        }
        if output_shape == input_shape {
            return parent_map.clone();
        }

        // Handle broadcasting: input may have fewer dims (padded with 1s on left)
        // or have 1s where output has larger dims
        let out_ndim = output_shape.len();
        let in_ndim = input_shape.len();
        let offset = out_ndim - in_ndim;
        let input_strides = Dim::strides(input_shape);

        let entries = (0..in_ndim)
            .map(|d| {
                let out_d = d + offset;
                if input_shape[d].is_one() && !output_shape[out_d].is_one() {
                    IndexEntry::Broadcast
                } else {
                    match &parent_map.entries[out_d] {
                        IndexEntry::Dim { dim, .. } => IndexEntry::Dim {
                            dim: *dim,
                            stride: input_strides[d].clone(),
                        },
                        IndexEntry::Broadcast => IndexEntry::Broadcast,
                    }
                }
            })
            .collect();

        IndexMap {
            iter_shape: parent_map.iter_shape.clone(),
            entries,
            offset: parent_map.offset.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tiling transform
// ---------------------------------------------------------------------------

/// Apply tiling to reduce loops whose dimensions are large enough to benefit.
/// This transforms the loop metadata only; codegen emits the tiled nest.
pub fn tile_reduce_loops(stmts: &mut Vec<Stmt>) {
    const TILE_M: usize = 8;
    const TILE_N: usize = 32;
    const TILE_K: usize = 32;
    // Minimum dimension size to bother tiling
    const MIN_DIM: usize = 16;

    for stmt in stmts.iter_mut() {
        if let Stmt::Loop {
            shape,
            reduce: Some(reduce),
            tile,
            ..
        } = stmt
        {
            // We tile 2D output + reduce (covers matmul pattern).
            // Need at least 2 output dims where the last two are large enough.
            if shape.len() < 2 {
                continue;
            }
            let ndim = shape.len();
            let m = &shape[ndim - 2];
            let n = &shape[ndim - 1];
            let k = &reduce.size;

            // Use concrete values for threshold checks; skip if symbolic.
            // Tiling requires concrete tile sizes and changes float
            // accumulation order, so we only tile when we can prove it
            // helps (all dims concrete and at least one large enough).
            let m_val = m.as_usize();
            let n_val = n.as_usize();
            let k_val = k.as_usize();

            // All three must be concrete for tiling to apply
            let (Some(mv), Some(nv), Some(kv)) = (m_val, n_val, k_val) else {
                continue;
            };

            // Skip if all dims are below threshold
            if mv < MIN_DIM && nv < MIN_DIM && kv < MIN_DIM {
                continue;
            }

            // Build tile sizes: leading dims get tile_size = dim_size (flat outer loops),
            // last two dims get tiled.
            let mut tiles: Vec<Dim> = Vec::with_capacity(ndim + 1);
            for d in 0..ndim - 2 {
                tiles.push(shape[d].clone()); // batch dims: tile = full size
            }
            tiles.push(Dim::Lit(TILE_M.min(mv)));
            tiles.push(Dim::Lit(TILE_N.min(nv)));
            tiles.push(Dim::Lit(TILE_K.min(kv))); // reduce dim tile

            *tile = Some(TileConfig { tiles });
        }
    }
}

/// Unfuse matmul bodies: when a tiled reduce loop has a fused body like
/// layernorm + matmul, split it into a separate elementwise loop for the
/// pre-processing and a clean Load-Load-Mul matmul body.
///
/// This enables the fast MR=8 micro-kernel which requires body_len==3.
pub fn unfuse_matmul_bodies(stmts: &mut Vec<Stmt>) {
    // Find the highest buf ID in use so new buffers don't conflict.
    let mut max_buf: usize = 0;
    for stmt in stmts.iter() {
        match stmt {
            Stmt::Alloc { buf, .. } | Stmt::Fill { buf, .. } | Stmt::FillArange { buf, .. } => {
                max_buf = max_buf.max(*buf);
            }
            Stmt::Loop { buf, .. } => {
                max_buf = max_buf.max(*buf);
            }
            Stmt::Pad { buf, input_buf, .. } => {
                max_buf = max_buf.max(*buf);
                max_buf = max_buf.max(*input_buf);
            }
        }
    }
    let mut next_buf = max_buf + 1;

    let mut insertions: Vec<(usize, Vec<Stmt>)> = Vec::new();

    for (si, stmt) in stmts.iter_mut().enumerate() {
        let Stmt::Loop { buf, shape, reduce: Some(reduce), body, result, tile: _ } = stmt else {
            continue;
        };
        if body.len() <= 3 { continue; }
        if reduce.op != ReduceOp::Sum { continue; }

        let result_idx = *result;
        let (pre_idx, weight_idx) = match &body[result_idx] {
            Inst::Mul(a, b) => (*a, *b),
            _ => continue,
        };

        let ndim = shape.len();
        if ndim < 2 { continue; }
        let n_dim = ndim - 1;
        let reduce_dim = ndim;

        // Check weight_load depends on both reduce_dim and n_dim
        let weight_ok = match &body[weight_idx] {
            Inst::Load { index: Index::Strided { parts, .. }, .. } => {
                parts.iter().any(|(d, _)| *d == n_dim) &&
                parts.iter().any(|(d, _)| *d == reduce_dim)
            }
            _ => false,
        };
        if !weight_ok { continue; }

        // pre chain must NOT depend on n_dim
        let dep_n = compute_dim_dep(body, n_dim);
        if dep_n[pre_idx] { continue; }

        // pre chain SHOULD depend on reduce_dim (otherwise hoisting handles it)
        let dep_k = compute_dim_dep(body, reduce_dim);
        if !dep_k[pre_idx] { continue; }

        // Build dim remapping: remove n_dim, reduce_dim → n_dim position
        // Original dims: 0..n_dim-1 (batch+m), n_dim (N), reduce_dim (K)
        // Pre dims: 0..n_dim-1 (batch+m), n_dim-1+1 = n_dim → K
        // So: d < n_dim → d, d == reduce_dim → n_dim, d == n_dim → skip

        // Compute transitive deps of pre_idx
        let pre_deps = compute_transitive_deps(body, pre_idx);

        // Build pre-processing body
        let mut pre_body: Vec<Inst> = Vec::new();
        let mut old_to_new: Vec<Option<usize>> = vec![None; body.len()];
        for j in 0..=pre_idx {
            if pre_deps[j] || j == pre_idx {
                let new_idx = pre_body.len();
                old_to_new[j] = Some(new_idx);
                let mut inst = remap_inst_refs(&body[j], &old_to_new);
                // Remap dim indices in Load instructions:
                // reduce_dim → n_dim (K is now the last dim of pre shape)
                // n_dim → skip (shouldn't appear, we checked dep_n)
                if let Inst::Load { index: Index::Strided { parts, .. }, .. } = &mut inst {
                    for (d, _) in parts.iter_mut() {
                        if *d == reduce_dim {
                            *d = n_dim; // K moves to position n_dim in pre shape
                        }
                    }
                }
                pre_body.push(inst);
            }
        }
        let pre_result = old_to_new[pre_idx].unwrap();

        // Pre shape: output dims (without last) + reduce dim as last
        // The pre-processing produces values for each (batch..., m, k) position
        let mut pre_shape: Vec<Dim> = shape[..n_dim].to_vec();
        pre_shape.push(reduce.size.clone());

        let pre_buf = next_buf;
        next_buf += 1;

        // Build the strided index for loading from pre_buf in the matmul.
        // pre_buf has shape (batch..., m_dim_out, K) stored contiguously.
        // In the matmul, we need to load pre_buf[batch_dims, m_dim, ki].
        // The strides are: last dim (K) has stride 1, m_dim has stride K,
        // batch dims have product strides.
        let pre_strides = Dim::strides(&pre_shape);
        let mut pre_load_parts: Vec<(usize, Dim)> = Vec::new();
        for (d, stride) in pre_strides.iter().enumerate() {
            if d < pre_shape.len() - 1 {
                // batch or m dim — maps to output dims 0..n_dim-1
                pre_load_parts.push((d, stride.clone()));
            }
        }
        // reduce_dim maps to the last dim of pre_shape (stride 1)
        pre_load_parts.push((reduce_dim, Dim::Lit(1)));

        let pre_load = Inst::Load {
            buf: pre_buf,
            index: Index::Strided { parts: pre_load_parts, offset: Dim::Lit(0) },
        };

        let new_body = vec![
            pre_load,
            body[weight_idx].clone(),
            Inst::Mul(0, 1),
        ];
        *body = new_body;
        *result = 2;

        insertions.push((si, vec![
            Stmt::Alloc { buf: pre_buf, size: Dim::product(&pre_shape) },
            Stmt::Loop {
                buf: pre_buf,
                shape: pre_shape,
                reduce: None,
                body: pre_body,
                result: pre_result,
                tile: None,
            },
        ]));
    }

    // Insert (reverse order for stable indices)
    for (si, new_stmts) in insertions.into_iter().rev() {
        for (i, s) in new_stmts.into_iter().enumerate() {
            stmts.insert(si + i, s);
        }
    }
}

fn compute_dim_dep(body: &[Inst], dim: usize) -> Vec<bool> {
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

fn compute_transitive_deps(body: &[Inst], target: usize) -> Vec<bool> {
    let mut needed = vec![false; body.len()];
    needed[target] = true;
    for j in (0..=target).rev() {
        if !needed[j] { continue; }
        match &body[j] {
            Inst::Neg(a) | Inst::Recip(a) | Inst::Exp2(a) | Inst::Log2(a) | Inst::Sqrt(a) => {
                needed[*a] = true;
            }
            Inst::Add(a, b) | Inst::Mul(a, b) | Inst::Max(a, b) | Inst::CmpLt(a, b) => {
                needed[*a] = true;
                needed[*b] = true;
            }
            _ => {}
        }
    }
    needed
}

fn remap_inst_refs(inst: &Inst, mapping: &[Option<usize>]) -> Inst {
    let r = |idx: usize| mapping[idx].unwrap_or(idx);
    match inst {
        Inst::Neg(a) => Inst::Neg(r(*a)),
        Inst::Recip(a) => Inst::Recip(r(*a)),
        Inst::Exp2(a) => Inst::Exp2(r(*a)),
        Inst::Log2(a) => Inst::Log2(r(*a)),
        Inst::Sqrt(a) => Inst::Sqrt(r(*a)),
        Inst::Add(a, b) => Inst::Add(r(*a), r(*b)),
        Inst::Mul(a, b) => Inst::Mul(r(*a), r(*b)),
        Inst::Max(a, b) => Inst::Max(r(*a), r(*b)),
        Inst::CmpLt(a, b) => Inst::CmpLt(r(*a), r(*b)),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_lang_graph::compile;

    #[test]
    fn test_lower_simple_add() {
        let graph = compile("let x = load([10]) let y = load([10]) let z = add(x, y)");
        let stmts = lower(&graph);
        // Should have: alloc x, alloc y, alloc z, loop z
        let allocs: Vec<_> = stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Alloc { .. }))
            .collect();
        let loops: Vec<_> = stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Loop { .. }))
            .collect();
        assert_eq!(allocs.len(), 3); // x, y, z
        assert_eq!(loops.len(), 1); // one loop for z
    }

    #[test]
    fn test_lower_fuses_chain() {
        // neg(add(x, y)) — the neg should be inlined into the add's consumer loop
        let graph = compile("let x = load([10]) let y = load([10]) let z = neg(add(x, y))");
        let stmts = lower(&graph);
        let loops: Vec<_> = stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Loop { .. }))
            .collect();
        // Only one loop (the final output), not two (add + neg)
        assert_eq!(loops.len(), 1);

        // The loop body should have: load x, load y, add, neg
        if let Stmt::Loop { body, result, .. } = &loops[0] {
            assert_eq!(body.len(), 4);
            assert!(matches!(body[0], Inst::Load { .. }));
            assert!(matches!(body[1], Inst::Load { .. }));
            assert!(matches!(body[2], Inst::Add(0, 1)));
            assert!(matches!(body[3], Inst::Neg(2)));
            assert_eq!(*result, 3);
        }
    }

    #[test]
    fn test_lower_matmul_fuses() {
        // A 2D matmul: [2,3] @ [3,4] -> [2,4]
        // Should produce: alloc A, alloc B, alloc output, one loop (the reduce)
        let graph = compile(
            "let a = load([2, 3]) let b = load([3, 4]) let c = matmul(a, b)",
        );
        let stmts = lower(&graph);

        let loops: Vec<_> = stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Loop { .. }))
            .collect();

        // The matmul decomposes into reshape+expand+mul+reducesum+reshape.
        // The reducesum is the only materialization point (besides inputs and output).
        // Wait — the final reshape is the output, and the reducesum is also materialized.
        // Let's count.
        eprintln!("Graph nodes: {}", graph.nodes.len());
        for (i, node) in graph.nodes.iter().enumerate() {
            eprintln!("  {}: {:?} shape={:?}", i, node.op, node.shape);
        }
        eprintln!("Stmts:");
        for s in &stmts {
            eprintln!("  {:?}", s);
        }

        // The reduce and the final reshape output are both materialized.
        // But the expand, mul, and inner reshapes should be fused.
        // Key check: no loop allocates the huge expanded shape.
        for s in &stmts {
            if let Stmt::Alloc { size, .. } = s {
                // The expanded shape would be 2*3*4=24, but the matmul intermediates
                // should NOT be allocated. Only: A(6), B(12), reduce output(8), final output(8).
                let sz = size.as_usize().expect("expected concrete size in test");
                assert!(
                    sz <= 12,
                    "Unexpected large allocation: size={size:?}"
                );
            }
        }

        // Should have a reduce loop
        let reduce_loops: Vec<_> = loops
            .iter()
            .filter(|s| {
                if let Stmt::Loop { reduce, .. } = s {
                    reduce.is_some()
                } else {
                    false
                }
            })
            .collect();
        assert!(!reduce_loops.is_empty(), "Expected at least one reduce loop");
    }

    #[test]
    fn test_lower_multi_consumer_materializes() {
        // x is used by both add and mul — it must be materialized
        let graph = compile(
            "let x = load([10]) let y = load([10]) let a = add(x, y) let b = mul(x, a)",
        );
        let stmts = lower(&graph);

        // add(x,y) has 1 consumer (mul's second input), but wait —
        // the result of add is used by b=mul(x, a), and b is the final output.
        // x has 2 consumers (add and mul), so x is materialized.
        // add has 1 consumer, so it should be inlined into mul's loop... but
        // add is also materialized because its output feeds mul which is the last node.
        // Actually: add's only consumer is mul. So add is NOT materialized (single consumer).
        // The final node is mul(x, add(x,y)). Its loop should inline add.

        let loops: Vec<_> = stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Loop { .. }))
            .collect();

        // Only one loop: for the final mul, which inlines the add
        assert_eq!(loops.len(), 1);
    }
}
