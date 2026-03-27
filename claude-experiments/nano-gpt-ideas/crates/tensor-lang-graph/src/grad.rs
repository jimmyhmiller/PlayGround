use std::collections::HashMap;

use crate::{Dim, Graph, Node, NodeId, Op};

impl Graph {
    /// Reverse-mode automatic differentiation.
    ///
    /// Given a scalar `loss` node and a set of parameter nodes (`wrt`),
    /// appends gradient computation nodes to the graph and returns
    /// the gradient NodeId for each parameter (in the same order as `wrt`).
    pub fn grad(&mut self, loss: NodeId, wrt: &[NodeId]) -> Vec<NodeId> {
        assert!(
            self.nodes[loss.0].shape.is_empty()
                || self.nodes[loss.0].shape.iter().all(|d| d.is_one()),
            "loss must be scalar, got shape {:?}",
            self.nodes[loss.0].shape
        );

        // grad_map: accumulated gradient for each forward node
        let mut grad_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Seed: d(loss)/d(loss) = 1, shaped like loss
        let one = self.add_node(Op::Constant(1.0), vec![]);
        let seed = if self.nodes[loss.0].shape.is_empty() {
            one
        } else {
            let shape = self.nodes[loss.0].shape.clone();
            let expanded = self.add_node(Op::Expand { shape }, vec![one]);
            expanded
        };
        grad_map.insert(loss, seed);

        // Walk in reverse topological order (nodes are already topo-sorted)
        for i in (0..=loss.0).rev() {
            let node_id = NodeId(i);
            let g_out = match grad_map.get(&node_id) {
                Some(&g) => g,
                None => continue, // node doesn't contribute to loss
            };

            // Clone what we need from the node before mutably borrowing self
            let node = self.nodes[i].clone();
            self.backward_op(node_id, &node, g_out, &mut grad_map);
        }

        // Collect gradients for requested parameters
        wrt.iter()
            .map(|&param| {
                *grad_map
                    .get(&param)
                    .unwrap_or_else(|| panic!("no gradient for node {:?}", param))
            })
            .collect()
    }

    fn backward_op(
        &mut self,
        _node_id: NodeId,
        node: &Node,
        g_out: NodeId,
        grad_map: &mut HashMap<NodeId, NodeId>,
    ) {
        match &node.op {
            // Data nodes: no inputs to propagate to
            Op::Input { .. } | Op::Constant(_) | Op::Arange { .. } => {}

            // --- Unary ops ---
            Op::Neg => {
                // d/da neg(a) = -1
                let g = self.add_node(Op::Neg, vec![g_out]);
                self.accumulate_grad(grad_map, node.inputs[0], g);
            }

            Op::Recip => {
                // d/da (1/a) = -1/a^2 = -recip(a)^2
                // grad_a = g_out * (-1) * recip(a)^2
                let a = node.inputs[0];
                let ra = self.add_node(Op::Recip, vec![a]);
                let ra2 = self.add_node(Op::Mul, vec![ra, ra]);
                let neg_ra2 = self.add_node(Op::Neg, vec![ra2]);
                let g = self.add_node(Op::Mul, vec![g_out, neg_ra2]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Exp2 => {
                // d/da exp2(a) = exp2(a) * ln(2)
                let a = node.inputs[0];
                let exp2_a = self.add_node(Op::Exp2, vec![a]);
                let ln2 = self.add_node(Op::Constant(std::f64::consts::LN_2), vec![]);
                let local = self.add_node(Op::Mul, vec![exp2_a, ln2]);
                let g = self.add_node(Op::Mul, vec![g_out, local]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Log2 => {
                // d/da log2(a) = 1 / (a * ln(2))
                let a = node.inputs[0];
                let ln2 = self.add_node(Op::Constant(std::f64::consts::LN_2), vec![]);
                let a_ln2 = self.add_node(Op::Mul, vec![a, ln2]);
                let inv = self.add_node(Op::Recip, vec![a_ln2]);
                let g = self.add_node(Op::Mul, vec![g_out, inv]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Sqrt => {
                // d/da sqrt(a) = 1 / (2 * sqrt(a))
                let a = node.inputs[0];
                let two = self.add_node(Op::Constant(2.0), vec![]);
                let sqrt_a = self.add_node(Op::Sqrt, vec![a]);
                let denom = self.add_node(Op::Mul, vec![two, sqrt_a]);
                let inv = self.add_node(Op::Recip, vec![denom]);
                let g = self.add_node(Op::Mul, vec![g_out, inv]);
                self.accumulate_grad(grad_map, a, g);
            }

            // --- Binary ops ---
            Op::Add => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let a_shape = self.nodes[a.0].shape.clone();
                let b_shape = self.nodes[b.0].shape.clone();

                let ga = self.unbroadcast(g_out, &a_shape);
                let gb = self.unbroadcast(g_out, &b_shape);
                self.accumulate_grad(grad_map, a, ga);
                self.accumulate_grad(grad_map, b, gb);
            }

            Op::Mul => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let a_shape = self.nodes[a.0].shape.clone();
                let b_shape = self.nodes[b.0].shape.clone();

                // grad_a = g_out * b, unbroadcast to a's shape
                let ga_full = self.add_node(Op::Mul, vec![g_out, b]);
                let ga = self.unbroadcast(ga_full, &a_shape);
                // grad_b = g_out * a, unbroadcast to b's shape
                let gb_full = self.add_node(Op::Mul, vec![g_out, a]);
                let gb = self.unbroadcast(gb_full, &b_shape);
                self.accumulate_grad(grad_map, a, ga);
                self.accumulate_grad(grad_map, b, gb);
            }

            Op::Max => {
                // Subgradient: grad goes to whichever input is larger
                let a = node.inputs[0];
                let b = node.inputs[1];
                let a_shape = self.nodes[a.0].shape.clone();
                let b_shape = self.nodes[b.0].shape.clone();

                // mask_b = (a < b) -> 1 where b wins, 0 where a wins
                let mask_b = self.add_node(Op::CmpLt, vec![a, b]);
                // mask_a = 1 - mask_b
                let one = self.add_node(Op::Constant(1.0), vec![]);
                let neg_mask_b = self.add_node(Op::Neg, vec![mask_b]);
                let mask_a = self.add_node(Op::Add, vec![one, neg_mask_b]);

                let ga_full = self.add_node(Op::Mul, vec![g_out, mask_a]);
                let ga = self.unbroadcast(ga_full, &a_shape);
                let gb_full = self.add_node(Op::Mul, vec![g_out, mask_b]);
                let gb = self.unbroadcast(gb_full, &b_shape);
                self.accumulate_grad(grad_map, a, ga);
                self.accumulate_grad(grad_map, b, gb);
            }

            Op::CmpLt => {
                // Not differentiable; gradient is 0 (no contribution)
            }

            // --- Reduce ops ---
            Op::ReduceSum { axis: _ } => {
                // grad = expand g_out back to input shape
                let a = node.inputs[0];
                let a_shape = self.nodes[a.0].shape.clone();
                let g = self.add_node(Op::Expand { shape: a_shape }, vec![g_out]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::ReduceMax { axis } => {
                // Subgradient: grad goes only to max elements
                let a = node.inputs[0];
                let a_shape = self.nodes[a.0].shape.clone();
                let axis = *axis;

                // max_expanded = expand(reduce_max_result, input_shape)
                // We need the forward reduce_max output. Since _node_id IS the
                // reduce_max, its output is the max values with shape [..., 1, ...]
                let max_expanded =
                    self.add_node(Op::Expand { shape: a_shape.clone() }, vec![_node_id]);

                // mask = 1 - cmplt(a, max_expanded)  (1 where a == max)
                let lt = self.add_node(Op::CmpLt, vec![a, max_expanded]);
                let one = self.add_node(Op::Constant(1.0), vec![]);
                let neg_lt = self.add_node(Op::Neg, vec![lt]);
                let mask = self.add_node(Op::Add, vec![one, neg_lt]);

                // Normalize mask so gradient is split evenly among ties
                let mask_sum = self.add_node(Op::ReduceSum { axis }, vec![mask]);
                let mask_sum_expanded =
                    self.add_node(Op::Expand { shape: a_shape.clone() }, vec![mask_sum]);
                let inv_count = self.add_node(Op::Recip, vec![mask_sum_expanded]);
                let mask_normalized = self.add_node(Op::Mul, vec![mask, inv_count]);

                let g_expanded =
                    self.add_node(Op::Expand { shape: a_shape }, vec![g_out]);
                let g = self.add_node(Op::Mul, vec![g_expanded, mask_normalized]);
                self.accumulate_grad(grad_map, a, g);
            }

            // --- Movement ops ---
            Op::Reshape { shape: _ } => {
                let a = node.inputs[0];
                let a_shape = self.nodes[a.0].shape.clone();
                let g = self.add_node(Op::Reshape { shape: a_shape }, vec![g_out]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Permute { order } => {
                let a = node.inputs[0];
                // Inverse permutation
                let mut inv = vec![0usize; order.len()];
                for (i, &o) in order.iter().enumerate() {
                    inv[o] = i;
                }
                let g = self.add_node(Op::Permute { order: inv }, vec![g_out]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Expand { shape: _ } => {
                // Reverse of expand: sum along expanded axes
                let a = node.inputs[0];
                let a_shape = self.nodes[a.0].shape.clone();
                let g = self.unbroadcast(g_out, &a_shape);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Pad { padding } => {
                // Reverse of pad: shrink to remove the padding
                let a = node.inputs[0];
                let a_shape = self.nodes[a.0].shape.clone();
                let bounds: Vec<(Dim, Dim)> = a_shape
                    .iter()
                    .zip(padding.iter())
                    .map(|(dim, &(lo, _hi))| {
                        let start = Dim::Lit(lo);
                        let end = Dim::Add(Box::new(start.clone()), Box::new(dim.clone())).simplify();
                        (start, end)
                    })
                    .collect();
                let g = self.add_node(Op::Shrink { bounds }, vec![g_out]);
                self.accumulate_grad(grad_map, a, g);
            }

            Op::Shrink { bounds } => {
                // Reverse of shrink: pad to restore original size
                let a = node.inputs[0];
                let a_shape = self.nodes[a.0].shape.clone();
                let _out_shape = node.shape.clone();
                // For each dim: lo padding = bounds[i].0, hi padding = a_shape[i] - bounds[i].1
                // But pad requires concrete (usize, usize) padding.
                // Since shrink bounds can be symbolic, we need to handle this carefully.
                // For now, only support concrete bounds.
                let padding: Vec<(usize, usize)> = a_shape
                    .iter()
                    .zip(bounds.iter())
                    .map(|(orig_dim, (lo, hi))| {
                        let lo_val = lo
                            .as_usize()
                            .expect("shrink backward requires concrete bounds");
                        let orig_val = orig_dim
                            .as_usize()
                            .expect("shrink backward requires concrete input shape");
                        let hi_val = hi
                            .as_usize()
                            .expect("shrink backward requires concrete bounds");
                        (lo_val, orig_val - hi_val)
                    })
                    .collect();
                let g = self.add_node(Op::Pad { padding }, vec![g_out]);
                self.accumulate_grad(grad_map, a, g);
            }
        }
    }

    /// Sum along axes that were broadcast, and reshape to target shape.
    /// This reverses the effect of broadcasting from `target_shape` to `g`'s shape.
    fn unbroadcast(&mut self, g: NodeId, target_shape: &[Dim]) -> NodeId {
        let g_shape = self.nodes[g.0].shape.clone();

        // Scalar target: sum everything
        if target_shape.is_empty() {
            let mut result = g;
            // Sum all axes from left to right (each sum keeps dim as 1)
            for axis in 0..g_shape.len() {
                result = self.add_node(Op::ReduceSum { axis }, vec![result]);
            }
            // Reshape to scalar
            result = self.add_node(Op::Reshape { shape: vec![] }, vec![result]);
            return result;
        }

        // Same shape: no-op
        if g_shape == target_shape {
            return g;
        }

        let g_ndim = g_shape.len();
        let t_ndim = target_shape.len();

        // Left-pad target shape with 1s to match g_ndim
        let pad_len = g_ndim.saturating_sub(t_ndim);
        let mut padded_target: Vec<Dim> = vec![Dim::Lit(1); pad_len];
        padded_target.extend_from_slice(target_shape);

        // Sum along axes where padded_target is 1 but g is not
        let mut result = g;
        for axis in 0..g_ndim {
            if padded_target[axis].is_one() && !g_shape[axis].is_one() {
                result = self.add_node(Op::ReduceSum { axis }, vec![result]);
            }
        }

        // Reshape to original target shape (removes leading 1s from padding)
        let cur_shape = self.nodes[result.0].shape.clone();
        if cur_shape != target_shape {
            result = self.add_node(Op::Reshape { shape: target_shape.to_vec() }, vec![result]);
        }

        result
    }

    /// Accumulate gradient: if a gradient already exists for this node, add to it.
    fn accumulate_grad(
        &mut self,
        grad_map: &mut HashMap<NodeId, NodeId>,
        node: NodeId,
        grad: NodeId,
    ) {
        match grad_map.get(&node) {
            Some(&existing) => {
                let sum = self.add_node(Op::Add, vec![existing, grad]);
                grad_map.insert(node, sum);
            }
            None => {
                grad_map.insert(node, grad);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{dims, Graph, NodeId, Op};

    /// Build a graph that reduces a tensor to a scalar by summing all axes,
    /// then reshaping to []. Returns the scalar loss node.
    fn sum_to_scalar(g: &mut Graph, node: NodeId) -> NodeId {
        let shape = g.nodes[node.0].shape.clone();
        if shape.is_empty() {
            return node;
        }
        let mut cur = node;
        for axis in 0..shape.len() {
            cur = g.add_node(Op::ReduceSum { axis }, vec![cur]);
        }
        g.add_node(Op::Reshape { shape: vec![] }, vec![cur])
    }

    #[test]
    fn test_grad_constant_mul() {
        let mut g = Graph::new();
        let a = g.add_node(Op::Input { name: "a".into() }, vec![]);
        let three = g.add_node(Op::Constant(3.0), vec![]);
        let loss = g.add_node(Op::Mul, vec![a, three]);
        let grads = g.grad(loss, &[a]);
        assert_eq!(grads.len(), 1);
    }

    #[test]
    fn test_grad_add() {
        let mut g = Graph::new();
        let a = g.add_node(Op::Input { name: "a".into() }, vec![]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![]);
        let loss = g.add_node(Op::Add, vec![a, b]);
        let grads = g.grad(loss, &[a, b]);
        assert_eq!(grads.len(), 2);
    }

    #[test]
    fn test_grad_chain_shapes() {
        // a^2 with tensor input, check grad shape
        let mut g = Graph::new();
        let a = g.add_node(Op::Input { name: "a".into() }, vec![]);
        g.set_input_shape(a, dims(&[3, 4]));
        let sq = g.add_node(Op::Mul, vec![a, a]);
        let loss = sum_to_scalar(&mut g, sq);
        let grads = g.grad(loss, &[a]);
        assert_eq!(g.nodes[grads[0].0].shape, dims(&[3, 4]));
    }

    #[test]
    fn test_grad_reshape_shape() {
        let mut g = Graph::new();
        let a = g.add_node(Op::Input { name: "a".into() }, vec![]);
        g.set_input_shape(a, dims(&[2, 2]));
        let r = g.add_node(Op::Reshape { shape: dims(&[4]) }, vec![a]);
        let loss = sum_to_scalar(&mut g, r);
        let grads = g.grad(loss, &[a]);
        assert_eq!(g.nodes[grads[0].0].shape, dims(&[2, 2]));
    }

    #[test]
    fn test_grad_permute_shape() {
        let mut g = Graph::new();
        let a = g.add_node(Op::Input { name: "a".into() }, vec![]);
        g.set_input_shape(a, dims(&[2, 3]));
        let p = g.add_node(Op::Permute { order: vec![1, 0] }, vec![a]);
        let loss = sum_to_scalar(&mut g, p);
        let grads = g.grad(loss, &[a]);
        assert_eq!(g.nodes[grads[0].0].shape, dims(&[2, 3]));
    }

    #[test]
    fn test_grad_broadcast_add_shapes() {
        let mut g = Graph::new();
        let a = g.add_node(Op::Input { name: "a".into() }, vec![]);
        g.set_input_shape(a, dims(&[4, 1]));
        let b = g.add_node(Op::Input { name: "b".into() }, vec![]);
        g.set_input_shape(b, dims(&[1, 3]));
        let add = g.add_node(Op::Add, vec![a, b]);
        let loss = sum_to_scalar(&mut g, add);
        let grads = g.grad(loss, &[a, b]);
        assert_eq!(g.nodes[grads[0].0].shape, dims(&[4, 1]));
        assert_eq!(g.nodes[grads[1].0].shape, dims(&[1, 3]));
    }
}
