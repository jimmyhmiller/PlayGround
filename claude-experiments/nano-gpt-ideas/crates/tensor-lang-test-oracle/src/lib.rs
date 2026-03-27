mod grad_check;

use ndarray::{ArrayD, IxDyn};
use tensor_lang_graph::{Dim, Graph, Op};
use std::collections::HashMap;

fn dim_shape_to_usize(shape: &[Dim]) -> Vec<usize> {
    shape.iter().map(|d| d.as_usize().expect("symbolic dim not yet supported in oracle")).collect()
}

/// Execute a graph with specific input values using ndarray as the reference.
/// Returns the value of every node in the graph.
pub fn eval_with_inputs(graph: &Graph, inputs: &HashMap<String, ArrayD<f32>>) -> Vec<ArrayD<f32>> {
    let mut buffers: Vec<Option<ArrayD<f32>>> = vec![None; graph.nodes.len()];

    for (i, node) in graph.nodes.iter().enumerate() {
        let result = match &node.op {
            Op::Input { name } => {
                inputs.get(name)
                    .unwrap_or_else(|| panic!("missing input: {name}"))
                    .clone()
            }
            _ => eval_node(node, &buffers),
        };
        buffers[i] = Some(result);
    }

    buffers.into_iter().map(|b| b.unwrap()).collect()
}

fn eval_node(node: &tensor_lang_graph::Node, buffers: &[Option<ArrayD<f32>>]) -> ArrayD<f32> {
    match &node.op {
        Op::Input { name } => panic!("input {name} not provided"),

        Op::Constant(v) => ArrayD::from_elem(IxDyn(&[]), *v as f32),

        Op::Arange { size } => {
            let sz = size.as_usize().expect("symbolic dim not yet supported in oracle");
            ArrayD::from_shape_vec(IxDyn(&[sz]), (0..sz).map(|i| i as f32).collect()).unwrap()
        }

        Op::Neg => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            -a
        }
        Op::Recip => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.mapv(|x| 1.0 / x)
        }
        Op::Exp2 => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.mapv(|x| (2.0_f32).powf(x))
        }
        Op::Log2 => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.mapv(|x| x.log2())
        }
        Op::Sqrt => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.mapv(|x| x.sqrt())
        }

        Op::Add => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let b = buffers[node.inputs[1].0].as_ref().unwrap();
            broadcast_binop(a, b, |x, y| x + y, &dim_shape_to_usize(&node.shape))
        }
        Op::Mul => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let b = buffers[node.inputs[1].0].as_ref().unwrap();
            broadcast_binop(a, b, |x, y| x * y, &dim_shape_to_usize(&node.shape))
        }
        Op::Max => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let b = buffers[node.inputs[1].0].as_ref().unwrap();
            broadcast_binop(a, b, |x, y| x.max(y), &dim_shape_to_usize(&node.shape))
        }
        Op::CmpLt => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let b = buffers[node.inputs[1].0].as_ref().unwrap();
            broadcast_binop(a, b, |x, y| if x < y { 1.0 } else { 0.0 }, &dim_shape_to_usize(&node.shape))
        }

        Op::ReduceSum { axis } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.sum_axis(ndarray::Axis(*axis)).insert_axis(ndarray::Axis(*axis))
        }
        Op::ReduceMax { axis } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.map_axis(ndarray::Axis(*axis), |lane| {
                lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            }).insert_axis(ndarray::Axis(*axis))
        }

        Op::Reshape { shape } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let usize_shape = dim_shape_to_usize(shape);
            // Ensure contiguous layout before reshape (permute can make it non-contiguous)
            let contiguous = a.as_standard_layout().into_owned();
            contiguous.into_shape_with_order(IxDyn(&usize_shape)).unwrap()
        }
        Op::Permute { order } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            a.clone().permuted_axes(IxDyn(order))
        }
        Op::Expand { shape } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let usize_shape = dim_shape_to_usize(shape);
            a.broadcast(IxDyn(&usize_shape)).unwrap().to_owned()
        }

        Op::Pad { padding } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let out_shape: Vec<usize> = a.shape().iter()
                .zip(padding.iter())
                .map(|(&d, &(lo, hi))| d + lo + hi)
                .collect();
            let mut out = ArrayD::zeros(IxDyn(&out_shape));
            let mut view = out.slice_each_axis_mut(|ax| {
                let (lo, _) = padding[ax.axis.index()];
                let dim = a.shape()[ax.axis.index()];
                ndarray::Slice::from(lo..(lo + dim))
            });
            view.assign(a);
            out
        }
        Op::Shrink { bounds } => {
            let a = buffers[node.inputs[0].0].as_ref().unwrap();
            let usize_bounds: Vec<(usize, usize)> = bounds.iter().map(|(lo, hi)| {
                (lo.as_usize().expect("symbolic dim not yet supported in oracle"),
                 hi.as_usize().expect("symbolic dim not yet supported in oracle"))
            }).collect();
            a.slice_each_axis(|ax| {
                let (lo, hi) = usize_bounds[ax.axis.index()];
                ndarray::Slice::from(lo..hi)
            }).to_owned()
        }
    }
}

/// Apply a binary operation with broadcasting.
fn broadcast_binop(
    a: &ArrayD<f32>,
    b: &ArrayD<f32>,
    f: impl Fn(f32, f32) -> f32,
    out_shape: &[usize],
) -> ArrayD<f32> {
    let out_dim = IxDyn(out_shape);

    let a_b = if a.shape() == out_shape {
        a.view()
    } else {
        a.broadcast(out_dim.clone()).unwrap()
    };
    let b_b = if b.shape() == out_shape {
        b.view()
    } else {
        b.broadcast(out_dim).unwrap()
    };

    ndarray::Zip::from(&a_b).and(&b_b).map_collect(|&x, &y| f(x, y))
}

/// Compare a flat f32 buffer (from a backend) against the oracle output for a specific node.
/// Returns Ok(()) if within tolerance, Err with details if not.
pub fn compare(
    oracle: &ArrayD<f32>,
    backend_output: &[f32],
    atol: f32,
) -> Result<(), String> {
    let oracle_flat: Vec<f32> = oracle.iter().copied().collect();

    if oracle_flat.len() != backend_output.len() {
        return Err(format!(
            "length mismatch: oracle={}, backend={}",
            oracle_flat.len(), backend_output.len()
        ));
    }

    for (i, (o, b)) in oracle_flat.iter().zip(backend_output.iter()).enumerate() {
        if (o - b).abs() > atol {
            return Err(format!(
                "mismatch at index {i}: oracle={o}, backend={b}, diff={}",
                (o - b).abs()
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tensor_lang_graph::compile;

    #[test]
    fn test_oracle_add() {
        let g = compile("let x = load([2, 3]) let y = load([2, 3]) let z = add(x, y)");
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        inputs.insert("input_1".into(), array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn());
        let results = eval_with_inputs(&g, &inputs);
        let expected = array![[11.0f32, 22.0, 33.0], [44.0, 55.0, 66.0]].into_dyn();
        assert_eq!(results.last().unwrap(), &expected);
    }

    #[test]
    fn test_oracle_broadcast_add() {
        let g = compile("let x = load([2, 3]) let y = add(x, 10.0)");
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let results = eval_with_inputs(&g, &inputs);
        let expected = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].into_dyn();
        assert_eq!(results.last().unwrap(), &expected);
    }

    #[test]
    fn test_oracle_reduce_sum() {
        let g = compile("let x = load([2, 3]) let s = sum(x, axis: 1)");
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let results = eval_with_inputs(&g, &inputs);
        let expected = array![[6.0f32], [15.0]].into_dyn();
        assert_eq!(results.last().unwrap(), &expected);
    }

    #[test]
    fn test_oracle_reduce_max() {
        let g = compile("let x = load([2, 3]) let m = max(x, axis: 1)");
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [6.0, 5.0, 4.0]].into_dyn());
        let results = eval_with_inputs(&g, &inputs);
        let expected = array![[3.0f32], [6.0]].into_dyn();
        assert_eq!(results.last().unwrap(), &expected);
    }

    #[test]
    fn test_oracle_softmax() {
        let input = r#"
            fn softmax(x) {
                let m = max(x, axis: 1)
                let e = exp(sub(x, m))
                let s = sum(e, axis: 1)
                mul(recip(s), e)
            }
            let x = load([2, 3])
            let y = softmax(x)
        "#;
        let g = compile(input);
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn());
        let results = eval_with_inputs(&g, &inputs);
        let out = results.last().unwrap();

        // Softmax rows should sum to 1
        for row in 0..2 {
            let row_sum: f32 = (0..3).map(|c| out[[row, c]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {row} sums to {row_sum}");
        }

        // Uniform input [1,1,1] should give uniform output [1/3, 1/3, 1/3]
        for c in 0..3 {
            assert!((out[[1, c]] - 1.0 / 3.0).abs() < 1e-5);
        }

        // [1,2,3] softmax: last element should be largest
        assert!(out[[0, 2]] > out[[0, 1]]);
        assert!(out[[0, 1]] > out[[0, 0]]);
    }

    #[test]
    fn test_oracle_permute() {
        let g = compile("let x = load([2, 3]) let y = permute(x, [1, 0])");
        let mut inputs = HashMap::new();
        inputs.insert("input_0".into(), array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let results = eval_with_inputs(&g, &inputs);
        let expected = array![[1.0f32, 4.0], [2.0, 5.0], [3.0, 6.0]].into_dyn();
        assert_eq!(results.last().unwrap(), &expected);
    }

    #[test]
    fn test_compare_helper() {
        let oracle = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
        assert!(compare(&oracle, &[1.0, 2.0, 3.0, 4.0], 1e-6).is_ok());
        assert!(compare(&oracle, &[1.0, 2.0, 3.0, 4.1], 1e-6).is_err());
        assert!(compare(&oracle, &[1.0, 2.0, 3.0, 4.1], 0.2).is_ok());
    }
}
