#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use rand::Rng;
    use std::collections::HashMap;
    use tensor_lang_graph::{dims, Dim, Graph, NodeId, Op};

    use crate::eval_with_inputs;

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

    fn check_grad_numerically(
        build: impl Fn() -> (Graph, Vec<NodeId>, NodeId),
        input_data: &[ArrayD<f32>],
        eps: f64,
        tol: f32,
    ) {
        let (mut g, param_ids, loss) = build();
        let grad_ids = g.grad(loss, &param_ids);

        let mut inputs: HashMap<String, ArrayD<f32>> = HashMap::new();
        for (i, data) in input_data.iter().enumerate() {
            inputs.insert(format!("input_{i}"), data.clone());
        }

        let all_values = eval_with_inputs(&g, &inputs);
        let analytic_grads: Vec<ArrayD<f32>> =
            grad_ids.iter().map(|&gid| all_values[gid.0].clone()).collect();

        for (p_idx, analytic) in analytic_grads.iter().enumerate() {
            let param_name = format!("input_{p_idx}");
            let param_data = &input_data[p_idx];
            let n_elems = param_data.len();

            for elem_idx in 0..n_elems {
                let mut data_plus = param_data.clone();
                data_plus.as_slice_mut().unwrap()[elem_idx] += eps as f32;
                let mut inputs_plus = inputs.clone();
                inputs_plus.insert(param_name.clone(), data_plus);
                let (g_plus, _, loss_plus) = build();
                let vals_plus = eval_with_inputs(&g_plus, &inputs_plus);
                let f_plus = vals_plus[loss_plus.0].iter().next().copied().unwrap() as f64;

                let mut data_minus = param_data.clone();
                data_minus.as_slice_mut().unwrap()[elem_idx] -= eps as f32;
                let mut inputs_minus = inputs.clone();
                inputs_minus.insert(param_name.clone(), data_minus);
                let (g_minus, _, loss_minus) = build();
                let vals_minus = eval_with_inputs(&g_minus, &inputs_minus);
                let f_minus = vals_minus[loss_minus.0].iter().next().copied().unwrap() as f64;

                let numerical = ((f_plus - f_minus) / (2.0 * eps)) as f32;
                let analytic_flat: Vec<f32> = analytic.iter().copied().collect();
                let analytic_val = analytic_flat[elem_idx];

                let diff = (numerical - analytic_val).abs();
                let scale = numerical.abs().max(analytic_val.abs()).max(1e-2);
                assert!(
                    diff / scale < tol,
                    "grad mismatch for param {p_idx} elem {elem_idx}: \
                     analytic={analytic_val}, numerical={numerical}, diff={diff}"
                );
            }
        }
    }

    fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    fn rand_array(shape: &[usize], seed: u64) -> ArrayD<f32> {
        let mut rng = seeded_rng(seed);
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen_range(-1.5..1.5))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
    }

    fn rand_positive_array(shape: &[usize], seed: u64) -> ArrayD<f32> {
        let mut rng = seeded_rng(seed);
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen_range(0.2..2.0))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
    }

    fn rand_small_array(shape: &[usize], seed: u64) -> ArrayD<f32> {
        let mut rng = seeded_rng(seed);
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|_| rng.gen_range(-0.8..0.8))
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
    }

    #[test]
    fn test_numgrad_mul_scalar() {
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                let b = g.add_node(Op::Input { name: "input_1".into() }, vec![]);
                let loss = g.add_node(Op::Mul, vec![a, b]);
                (g, vec![a, b], loss)
            },
            &[
                ArrayD::from_elem(IxDyn(&[]), 3.0f32),
                ArrayD::from_elem(IxDyn(&[]), 5.0f32),
            ],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_add_tensor() {
        let a_data = rand_array(&[3, 4], 42);
        let b_data = rand_array(&[3, 4], 43);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let b = g.add_node(Op::Input { name: "input_1".into() }, vec![]);
                g.set_input_shape(b, dims(&[3, 4]));
                let add = g.add_node(Op::Add, vec![a, b]);
                let loss = sum_to_scalar(&mut g, add);
                (g, vec![a, b], loss)
            },
            &[a_data, b_data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_mul_tensor() {
        let a_data = rand_array(&[3, 4], 42);
        let b_data = rand_array(&[3, 4], 43);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let b = g.add_node(Op::Input { name: "input_1".into() }, vec![]);
                g.set_input_shape(b, dims(&[3, 4]));
                let mul = g.add_node(Op::Mul, vec![a, b]);
                let loss = sum_to_scalar(&mut g, mul);
                (g, vec![a, b], loss)
            },
            &[a_data, b_data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_neg() {
        let data = rand_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let neg = g.add_node(Op::Neg, vec![a]);
                let loss = sum_to_scalar(&mut g, neg);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_recip() {
        let data = rand_positive_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let r = g.add_node(Op::Recip, vec![a]);
                let loss = sum_to_scalar(&mut g, r);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_exp2() {
        let data = rand_small_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let e = g.add_node(Op::Exp2, vec![a]);
                let loss = sum_to_scalar(&mut g, e);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_log2() {
        let data = rand_positive_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let l = g.add_node(Op::Log2, vec![a]);
                let loss = sum_to_scalar(&mut g, l);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_sqrt() {
        let data = rand_positive_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let s = g.add_node(Op::Sqrt, vec![a]);
                let loss = sum_to_scalar(&mut g, s);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_square() {
        let data = rand_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let sq = g.add_node(Op::Mul, vec![a, a]);
                let loss = sum_to_scalar(&mut g, sq);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_broadcast_mul() {
        let a_data = rand_array(&[4, 1], 42);
        let b_data = rand_array(&[1, 3], 43);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[4, 1]));
                let b = g.add_node(Op::Input { name: "input_1".into() }, vec![]);
                g.set_input_shape(b, dims(&[1, 3]));
                let mul = g.add_node(Op::Mul, vec![a, b]);
                let loss = sum_to_scalar(&mut g, mul);
                (g, vec![a, b], loss)
            },
            &[a_data, b_data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_reshape() {
        let data = rand_array(&[2, 6], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[2, 6]));
                let r = g.add_node(Op::Reshape { shape: dims(&[3, 4]) }, vec![a]);
                let sq = g.add_node(Op::Mul, vec![r, r]);
                let loss = sum_to_scalar(&mut g, sq);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_permute() {
        let data = rand_array(&[2, 3], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[2, 3]));
                let p = g.add_node(Op::Permute { order: vec![1, 0] }, vec![a]);
                let sq = g.add_node(Op::Mul, vec![p, p]);
                let loss = sum_to_scalar(&mut g, sq);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_reduce_sum() {
        let data = rand_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let sq = g.add_node(Op::Mul, vec![a, a]);
                let s = g.add_node(Op::ReduceSum { axis: 1 }, vec![sq]);
                let loss = sum_to_scalar(&mut g, s);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_expand() {
        let data = rand_array(&[3, 1], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 1]));
                let e = g.add_node(Op::Expand { shape: dims(&[3, 4]) }, vec![a]);
                let sq = g.add_node(Op::Mul, vec![e, e]);
                let loss = sum_to_scalar(&mut g, sq);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_pad() {
        let data = rand_array(&[2, 3], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[2, 3]));
                let p = g.add_node(Op::Pad { padding: vec![(1, 1), (0, 2)] }, vec![a]);
                let sq = g.add_node(Op::Mul, vec![p, p]);
                let loss = sum_to_scalar(&mut g, sq);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_shrink() {
        let data = rand_array(&[4, 6], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[4, 6]));
                let s = g.add_node(
                    Op::Shrink {
                        bounds: vec![
                            (Dim::Lit(1), Dim::Lit(3)),
                            (Dim::Lit(0), Dim::Lit(4)),
                        ],
                    },
                    vec![a],
                );
                let sq = g.add_node(Op::Mul, vec![s, s]);
                let loss = sum_to_scalar(&mut g, sq);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_chain_exp_mul() {
        let data = rand_small_array(&[3, 4], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[3, 4]));
                let e = g.add_node(Op::Exp2, vec![a]);
                let prod = g.add_node(Op::Mul, vec![e, a]);
                let loss = sum_to_scalar(&mut g, prod);
                (g, vec![a], loss)
            },
            &[data],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_numgrad_softmax() {
        let data = rand_small_array(&[2, 5], 42);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let x = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(x, dims(&[2, 5]));

                let mx = g.add_node(Op::ReduceMax { axis: 1 }, vec![x]);
                let neg_mx = g.add_node(Op::Neg, vec![mx]);
                let shifted = g.add_node(Op::Add, vec![x, neg_mx]);
                let log2e = g.add_node(Op::Constant(std::f64::consts::LOG2_E), vec![]);
                let scaled = g.add_node(Op::Mul, vec![shifted, log2e]);
                let ex = g.add_node(Op::Exp2, vec![scaled]);
                let sum_ex = g.add_node(Op::ReduceSum { axis: 1 }, vec![ex]);
                let inv_sum = g.add_node(Op::Recip, vec![sum_ex]);
                let softmax = g.add_node(Op::Mul, vec![ex, inv_sum]);

                let prod = g.add_node(Op::Mul, vec![softmax, x]);
                let loss = sum_to_scalar(&mut g, prod);
                (g, vec![x], loss)
            },
            &[data],
            1e-3,
            5e-2,
        );
    }

    #[test]
    fn test_numgrad_matmul() {
        let a_data = rand_array(&[2, 3], 42);
        let b_data = rand_array(&[3, 4], 43);
        check_grad_numerically(
            || {
                let mut g = Graph::new();
                let a = g.add_node(Op::Input { name: "input_0".into() }, vec![]);
                g.set_input_shape(a, dims(&[2, 3]));
                let b = g.add_node(Op::Input { name: "input_1".into() }, vec![]);
                g.set_input_shape(b, dims(&[3, 4]));

                let a_r = g.add_node(Op::Reshape { shape: dims(&[2, 3, 1]) }, vec![a]);
                let b_r = g.add_node(Op::Reshape { shape: dims(&[1, 3, 4]) }, vec![b]);
                let a_e = g.add_node(Op::Expand { shape: dims(&[2, 3, 4]) }, vec![a_r]);
                let b_e = g.add_node(Op::Expand { shape: dims(&[2, 3, 4]) }, vec![b_r]);
                let prod = g.add_node(Op::Mul, vec![a_e, b_e]);
                let summed = g.add_node(Op::ReduceSum { axis: 1 }, vec![prod]);
                let result = g.add_node(Op::Reshape { shape: dims(&[2, 4]) }, vec![summed]);

                let loss = sum_to_scalar(&mut g, result);
                (g, vec![a, b], loss)
            },
            &[a_data, b_data],
            1e-3,
            1e-2,
        );
    }
}
