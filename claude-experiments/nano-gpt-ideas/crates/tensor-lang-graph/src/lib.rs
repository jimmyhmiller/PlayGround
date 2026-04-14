pub mod dim;
pub mod grad;
pub mod nanogpt;
pub mod runtime_trait;

pub use runtime_trait::TensorRuntime;

use std::collections::HashMap;
use tensor_lang_parser::{self as parser, Arg, BinOpKind, Expr, Item, Stmt};

pub use dim::Dim;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // Data
    Input { name: String },
    Constant(f64),
    Arange { size: Dim },

    // Unary (elementwise)
    Neg,
    Recip,
    Exp2,
    Log2,
    Sqrt,

    // Binary (elementwise)
    Add,
    Mul,
    Max,
    CmpLt,

    // Reduce
    ReduceSum { axis: usize },
    ReduceMax { axis: usize },

    // Movement
    Reshape { shape: Vec<Dim> },
    Permute { order: Vec<usize> },
    Expand { shape: Vec<Dim> },
    Pad { padding: Vec<(usize, usize)> },
    Shrink { bounds: Vec<(Dim, Dim)> },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<Dim>,
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, op: Op, inputs: Vec<NodeId>) -> NodeId {
        let shape = self.infer_shape(&op, &inputs);
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, inputs, shape });
        id
    }

    fn input_shape(&self, id: NodeId) -> &[Dim] {
        &self.nodes[id.0].shape
    }

    fn infer_shape(&self, op: &Op, inputs: &[NodeId]) -> Vec<Dim> {
        match op {
            Op::Input { .. } => {
                // Shape set after creation via set_input_shape
                vec![]
            }
            Op::Constant(_) => {
                // Scalar
                vec![]
            }
            Op::Arange { size } => {
                vec![size.clone()]
            }

            // Unary elementwise — same shape as input
            Op::Neg | Op::Recip | Op::Exp2 | Op::Log2 | Op::Sqrt => {
                self.input_shape(inputs[0]).to_vec()
            }

            // Binary elementwise — broadcast
            Op::Add | Op::Mul | Op::Max | Op::CmpLt => {
                let a = self.input_shape(inputs[0]);
                let b = self.input_shape(inputs[1]);
                broadcast_shapes(a, b)
            }

            // Reduce — axis becomes size 1 (keepdim style, makes broadcasting work)
            Op::ReduceSum { axis } | Op::ReduceMax { axis } => {
                let a = self.input_shape(inputs[0]);
                let mut out = a.to_vec();
                if *axis < out.len() {
                    out[*axis] = Dim::Lit(1);
                }
                out
            }

            // Movement
            Op::Reshape { shape } => shape.clone(),
            Op::Permute { order } => {
                let a = self.input_shape(inputs[0]);
                order.iter().map(|&i| a[i].clone()).collect()
            }
            Op::Expand { shape } => shape.clone(),
            Op::Pad { padding } => {
                let a = self.input_shape(inputs[0]);
                a.iter().zip(padding.iter())
                    .map(|(dim, &(lo, hi))| {
                        Dim::Add(Box::new(dim.clone()), Box::new(Dim::Lit(lo + hi))).simplify()
                    })
                    .collect()
            }
            Op::Shrink { bounds } => {
                bounds.iter().map(|(lo, hi)| {
                    Dim::Sub(Box::new(hi.clone()), Box::new(lo.clone())).simplify()
                }).collect()
            }
        }
    }

    pub fn set_input_shape(&mut self, id: NodeId, shape: Vec<Dim>) {
        self.nodes[id.0].shape = shape;
    }


    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n    rankdir=TB;\n    node [shape=record, fontname=\"monospace\"];\n");

        for (i, node) in self.nodes.iter().enumerate() {
            let label = match &node.op {
                Op::Input { name } => format!("input\\n{name}"),
                Op::Constant(v) => format!("{v}"),
                Op::Arange { size } => format!("arange\\n{size:?}"),
                Op::Neg => "neg".into(),
                Op::Recip => "recip".into(),
                Op::Exp2 => "exp2".into(),
                Op::Log2 => "log2".into(),
                Op::Sqrt => "sqrt".into(),
                Op::Add => "+".into(),
                Op::Mul => "*".into(),
                Op::Max => "max".into(),
                Op::CmpLt => "<".into(),
                Op::ReduceSum { axis } => format!("sum\\naxis={axis}"),
                Op::ReduceMax { axis } => format!("max\\naxis={axis}"),
                Op::Reshape { shape } => format!("reshape\\n{shape:?}"),
                Op::Permute { order } => format!("permute\\n{order:?}"),
                Op::Expand { shape } => format!("expand\\n{shape:?}"),
                Op::Pad { padding } => format!("pad\\n{padding:?}"),
                Op::Shrink { bounds } => format!("shrink\\n{bounds:?}"),
            };

            let color = match &node.op {
                Op::Input { .. } | Op::Constant(_) | Op::Arange { .. } => "lightblue",
                Op::Neg | Op::Recip | Op::Exp2 | Op::Log2 | Op::Sqrt => "lightyellow",
                Op::Add | Op::Mul | Op::Max | Op::CmpLt => "lightgreen",
                Op::ReduceSum { .. } | Op::ReduceMax { .. } => "lightsalmon",
                _ => "lightgray",
            };

            out.push_str(&format!(
                "    n{i} [label=\"{label}\", style=filled, fillcolor={color}];\n"
            ));

            for input in &node.inputs {
                out.push_str(&format!("    n{} -> n{i};\n", input.0));
            }
        }

        out.push_str("}\n");
        out
    }
}

/// Broadcast two shapes following NumPy rules:
/// - Align from the right
/// - Dimensions must match, or one must be 1
/// - Missing dimensions are treated as 1
pub fn broadcast_shapes(a: &[Dim], b: &[Dim]) -> Vec<Dim> {
    // Scalar (empty shape) broadcasts to anything
    if a.is_empty() { return b.to_vec(); }
    if b.is_empty() { return a.to_vec(); }

    let len = a.len().max(b.len());
    let mut result = vec![Dim::Lit(0); len];
    for i in 0..len {
        let da = if i < a.len() { &a[a.len() - 1 - i] } else { &Dim::Lit(1) };
        let db = if i < b.len() { &b[b.len() - 1 - i] } else { &Dim::Lit(1) };
        if da == db {
            result[len - 1 - i] = da.clone();
        } else if da.is_one() {
            result[len - 1 - i] = db.clone();
        } else if db.is_one() {
            result[len - 1 - i] = da.clone();
        } else {
            panic!("cannot broadcast shapes {a:?} and {b:?}: dimension {da:?} vs {db:?}");
        }
    }
    result
}

// Compiler: walks AST, builds graph

struct Compiler {
    graph: Graph,
    env: HashMap<String, NodeId>,
    // Function definitions, stored for when they're called
    fns: HashMap<String, parser::FnDef>,
    input_count: usize,
    /// Declared dimension parameters (symbolic dims like "T" for seq_len)
    dim_params: std::collections::HashSet<String>,
    /// Concrete values for dimension parameters. If a dim is in this map,
    /// it resolves to Dim::Lit instead of Dim::Param.
    dim_values: HashMap<String, usize>,
    /// Named float constants injected from the environment.
    /// Referenced as identifiers in expressions, compiled to Constant nodes.
    constants: HashMap<String, f64>,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            graph: Graph::new(),
            env: HashMap::new(),
            fns: HashMap::new(),
            input_count: 0,
            dim_params: std::collections::HashSet::new(),
            dim_values: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    fn compile_program(mut self, items: &[Item]) -> Graph {
        // First pass: collect function definitions and dim declarations
        // (recursing into for-loop bodies for fns/dims as well).
        self.collect_decls(items);

        // Second pass: compile top-level let bindings, unrolling for-loops.
        self.compile_items(items);

        self.graph
    }

    fn collect_decls(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::FnDef(f) => { self.fns.insert(f.name.clone(), f.clone()); }
                Item::DimDecl(name) => { self.dim_params.insert(name.clone()); }
                Item::ForLoop { body, .. } => self.collect_decls(body),
                _ => {}
            }
        }
    }

    fn compile_items(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::Let(binding) => {
                    let id = self.compile_expr(&binding.value);
                    self.env.insert(binding.name.clone(), id);
                }
                Item::ForLoop { var, start, end, body } => {
                    let start_val = self.eval_concrete_int(start);
                    let end_val = self.eval_concrete_int(end);
                    let had_prev = self.dim_values.contains_key(var);
                    let prev = self.dim_values.get(var).copied();
                    for i in start_val..end_val {
                        // Expose loop var as both a dim value (for shapes) and
                        // a constant (for use in runtime expressions).
                        self.dim_values.insert(var.clone(), i);
                        self.constants.insert(var.clone(), i as f64);
                        self.compile_items(body);
                    }
                    // Restore prior binding (if any).
                    if had_prev {
                        self.dim_values.insert(var.clone(), prev.unwrap());
                        self.constants.insert(var.clone(), prev.unwrap() as f64);
                    } else {
                        self.dim_values.remove(var);
                        self.constants.remove(var);
                    }
                }
                // FnDef and DimDecl already handled in collect_decls.
                Item::FnDef(_) | Item::DimDecl(_) => {}
            }
        }
    }

    /// Evaluate a for-loop bound. Must resolve to a concrete integer:
    /// either a number literal, a declared dim with a known value, or an
    /// arithmetic expression over those.
    fn eval_concrete_int(&self, expr: &Expr) -> usize {
        let dim = self.compile_dim(expr);
        dim.as_usize().unwrap_or_else(|| {
            panic!("for-loop bound must be a concrete integer, got {dim:?}")
        })
    }

    fn compile_expr(&mut self, expr: &Expr) -> NodeId {
        match expr {
            Expr::Number(n) => self.graph.add_node(Op::Constant(*n), vec![]),

            Expr::Ident(name) => {
                if let Some(&id) = self.env.get(name) {
                    id
                } else if let Some(&val) = self.constants.get(name) {
                    self.graph.add_node(Op::Constant(val), vec![])
                } else {
                    panic!("undefined variable: {name}")
                }
            }

            Expr::Array(_) => {
                // Arrays are only used as arguments to ops (like shapes),
                // not as standalone expressions in the graph
                panic!("array literals can't appear as standalone expressions")
            }

            Expr::BinOp { op, lhs, rhs } => {
                let l = self.compile_expr(lhs);
                let r = self.compile_expr(rhs);
                match op {
                    BinOpKind::Add => self.graph.add_node(Op::Add, vec![l, r]),
                    BinOpKind::Mul => self.graph.add_node(Op::Mul, vec![l, r]),
                    BinOpKind::Sub => {
                        let neg = self.graph.add_node(Op::Neg, vec![r]);
                        self.graph.add_node(Op::Add, vec![l, neg])
                    }
                    BinOpKind::Div => {
                        let rec = self.graph.add_node(Op::Recip, vec![r]);
                        self.graph.add_node(Op::Mul, vec![l, rec])
                    }
                }
            }

            Expr::Call { name, args } => self.compile_call(name, args),
        }
    }

    fn compile_call(&mut self, name: &str, args: &[Arg]) -> NodeId {
        match name {
            // Unary primitives
            "neg" => {
                let a = self.compile_positional(args, 0);
                self.graph.add_node(Op::Neg, vec![a])
            }
            "recip" => {
                let a = self.compile_positional(args, 0);
                self.graph.add_node(Op::Recip, vec![a])
            }
            "exp" => {
                // exp(x) = exp2(x * LOG2_E)
                let a = self.compile_positional(args, 0);
                let log2e = self.graph.add_node(Op::Constant(std::f64::consts::LOG2_E), vec![]);
                let scaled = self.graph.add_node(Op::Mul, vec![a, log2e]);
                self.graph.add_node(Op::Exp2, vec![scaled])
            }
            "log" => {
                // log(x) = log2(x) * LN_2
                let a = self.compile_positional(args, 0);
                let l = self.graph.add_node(Op::Log2, vec![a]);
                let ln2 = self.graph.add_node(Op::Constant(std::f64::consts::LN_2), vec![]);
                self.graph.add_node(Op::Mul, vec![l, ln2])
            }
            "exp2" => {
                let a = self.compile_positional(args, 0);
                self.graph.add_node(Op::Exp2, vec![a])
            }
            "log2" => {
                let a = self.compile_positional(args, 0);
                self.graph.add_node(Op::Log2, vec![a])
            }
            "sqrt" => {
                let a = self.compile_positional(args, 0);
                self.graph.add_node(Op::Sqrt, vec![a])
            }

            // Binary primitives
            "add" => {
                let a = self.compile_positional(args, 0);
                let b = self.compile_positional(args, 1);
                self.graph.add_node(Op::Add, vec![a, b])
            }
            "mul" => {
                let a = self.compile_positional(args, 0);
                let b = self.compile_positional(args, 1);
                self.graph.add_node(Op::Mul, vec![a, b])
            }
            "sub" => {
                let a = self.compile_positional(args, 0);
                let b = self.compile_positional(args, 1);
                let neg = self.graph.add_node(Op::Neg, vec![b]);
                self.graph.add_node(Op::Add, vec![a, neg])
            }
            "div" => {
                let a = self.compile_positional(args, 0);
                let b = self.compile_positional(args, 1);
                let rec = self.graph.add_node(Op::Recip, vec![b]);
                self.graph.add_node(Op::Mul, vec![a, rec])
            }
            "max" => {
                let a = self.compile_positional(args, 0);
                // max(x, axis: n) is reduce, max(x, y) is binary
                if let Some(axis) = self.get_named_arg(args, "axis") {
                    let axis = self.const_usize(axis);
                    self.graph.add_node(Op::ReduceMax { axis }, vec![a])
                } else {
                    let b = self.compile_positional(args, 1);
                    self.graph.add_node(Op::Max, vec![a, b])
                }
            }
            "cmplt" => {
                let a = self.compile_positional(args, 0);
                let b = self.compile_positional(args, 1);
                self.graph.add_node(Op::CmpLt, vec![a, b])
            }

            // Reduce
            "sum" => {
                let a = self.compile_positional(args, 0);
                let axis = self.get_named_arg(args, "axis")
                    .expect("sum requires axis: argument");
                let axis = self.const_usize(axis);
                self.graph.add_node(Op::ReduceSum { axis }, vec![a])
            }

            // Movement
            "reshape" => {
                let a = self.compile_positional(args, 0);
                let shape = self.compile_shape(args, 1);
                self.graph.add_node(Op::Reshape { shape }, vec![a])
            }
            "permute" => {
                let a = self.compile_positional(args, 0);
                // Permute order is always concrete indices
                let order: Vec<usize> = self.compile_shape(args, 1)
                    .iter().map(|d| d.as_usize().expect("permute order must be concrete")).collect();
                self.graph.add_node(Op::Permute { order }, vec![a])
            }
            "expand" => {
                let a = self.compile_positional(args, 0);
                let shape = self.compile_shape(args, 1);
                self.graph.add_node(Op::Expand { shape }, vec![a])
            }
            "pad" => {
                let a = self.compile_positional(args, 0);
                // Pad amounts are always concrete
                let pairs = self.compile_pairs(args, 1);
                let padding: Vec<(usize, usize)> = pairs.iter().map(|(lo, hi)| {
                    (lo.as_usize().expect("pad lo must be concrete"),
                     hi.as_usize().expect("pad hi must be concrete"))
                }).collect();
                self.graph.add_node(Op::Pad { padding }, vec![a])
            }
            "shrink" => {
                let a = self.compile_positional(args, 0);
                let bounds = self.compile_pairs(args, 1);
                self.graph.add_node(Op::Shrink { bounds }, vec![a])
            }

            // matmul(A, B): decompose into reshape + expand + mul + sum
            // A[..., M, K] @ B[..., K, N] -> [..., M, N]
            "matmul" => {
                let a = self.compile_positional(args, 0);
                let b = self.compile_positional(args, 1);
                let a_shape = self.graph.nodes[a.0].shape.clone();
                let b_shape = self.graph.nodes[b.0].shape.clone();

                assert!(a_shape.len() >= 2, "matmul requires at least 2D tensors, got {:?}", a_shape);
                assert!(b_shape.len() >= 2, "matmul requires at least 2D tensors, got {:?}", b_shape);

                let ndim_a = a_shape.len();
                let ndim_b = b_shape.len();
                let m = a_shape[ndim_a - 2].clone();
                let k = a_shape[ndim_a - 1].clone();
                let k2 = b_shape[ndim_b - 2].clone();
                let n = b_shape[ndim_b - 1].clone();
                assert_eq!(k, k2, "matmul inner dimensions must match: {:?} vs {:?}", k, k2);

                // Get batch dims (everything except last 2)
                let batch_a = &a_shape[..ndim_a - 2];
                let batch_b = &b_shape[..ndim_b - 2];
                let batch = broadcast_shapes(batch_a, batch_b);

                // A: [batch_a..., M, K] -> [batch_a..., M, K, 1]
                let mut a_reshaped: Vec<Dim> = batch_a.to_vec();
                a_reshaped.push(m.clone());
                a_reshaped.push(k.clone());
                a_reshaped.push(Dim::Lit(1));

                // A: [batch_a..., M, K] -> [batch_a..., M, K, 1]
                let mut a_reshaped: Vec<Dim> = batch_a.to_vec();
                a_reshaped.push(m.clone());
                a_reshaped.push(k.clone());
                a_reshaped.push(Dim::Lit(1));

                // B: [batch_b..., K, N] -> [batch_b..., 1, K, N]
                let mut b_reshaped: Vec<Dim> = batch_b.to_vec();
                b_reshaped.push(Dim::Lit(1));
                b_reshaped.push(k.clone());
                b_reshaped.push(n.clone());

                // Expand target: [batch..., M, K, N]
                let mut expanded_shape: Vec<Dim> = batch.clone();
                expanded_shape.push(m.clone());
                expanded_shape.push(k);
                expanded_shape.push(n.clone());

                let a_r = self.graph.add_node(Op::Reshape { shape: a_reshaped }, vec![a]);
                let b_r = self.graph.add_node(Op::Reshape { shape: b_reshaped }, vec![b]);
                let a_e = self.graph.add_node(Op::Expand { shape: expanded_shape.clone() }, vec![a_r]);
                let b_e = self.graph.add_node(Op::Expand { shape: expanded_shape.clone() }, vec![b_r]);
                let prod = self.graph.add_node(Op::Mul, vec![a_e, b_e]);
                let k_axis = batch.len() + 1;
                let summed = self.graph.add_node(Op::ReduceSum { axis: k_axis }, vec![prod]);
                // Reshape to remove K=1 dim: [..., M, 1, N] -> [..., M, N]
                let mut out_shape: Vec<Dim> = batch;
                out_shape.push(m);
                out_shape.push(n);
                self.graph.add_node(Op::Reshape { shape: out_shape }, vec![summed])
            }

            // arange(n) produces [0, 1, 2, ..., n-1] as a 1D tensor
            "arange" => {
                let size = self.compile_dim(self.get_positional_expr(args, 0));
                self.graph.add_node(Op::Arange { size }, vec![])
            }

            // load creates an input node with a known shape
            "load" => {
                let shape = self.compile_shape(args, 0);
                let name = format!("input_{}", self.input_count);
                self.input_count += 1;
                let id = self.graph.add_node(Op::Input { name }, vec![]);
                self.graph.set_input_shape(id, shape);
                id
            }

            // User-defined function call
            _ => {
                let f = self.fns.get(name)
                    .unwrap_or_else(|| panic!("undefined function: {name}"))
                    .clone();

                // Save outer env, bind params
                let outer_env = self.env.clone();
                for (i, param) in f.params.iter().enumerate() {
                    let arg_id = self.compile_positional(args, i);
                    self.env.insert(param.clone(), arg_id);
                }

                // Compile body
                let mut result = None;
                for stmt in &f.body {
                    match stmt {
                        Stmt::Let(binding) => {
                            let id = self.compile_expr(&binding.value);
                            self.env.insert(binding.name.clone(), id);
                        }
                        Stmt::Expr(expr) => {
                            result = Some(self.compile_expr(expr));
                        }
                    }
                }

                // Restore outer env
                self.env = outer_env;
                result.expect("function body must end with an expression")
            }
        }
    }

    fn get_positional_expr<'a>(&self, args: &'a [Arg], index: usize) -> &'a Expr {
        let mut pos = 0;
        for arg in args {
            if let Arg::Positional(expr) = arg {
                if pos == index {
                    return expr;
                }
                pos += 1;
            }
        }
        panic!("missing positional argument at index {index}")
    }

    fn compile_positional(&mut self, args: &[Arg], index: usize) -> NodeId {
        let mut pos = 0;
        for arg in args {
            if let Arg::Positional(expr) = arg {
                if pos == index {
                    return self.compile_expr(expr);
                }
                pos += 1;
            }
        }
        panic!("missing positional argument at index {index}")
    }

    fn get_named_arg<'a>(&self, args: &'a [Arg], name: &str) -> Option<&'a Expr> {
        for arg in args {
            if let Arg::Named { name: n, value } = arg {
                if n == name {
                    return Some(value);
                }
            }
        }
        None
    }

    fn const_usize(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Number(n) => *n as usize,
            _ => panic!("expected constant number, got {expr:?}"),
        }
    }

    fn compile_dim(&self, expr: &Expr) -> Dim {
        match expr {
            Expr::Number(n) => Dim::Lit(*n as usize),
            Expr::Ident(name) => {
                if let Some(&val) = self.dim_values.get(name) {
                    Dim::Lit(val)
                } else if self.dim_params.contains(name) {
                    Dim::Param(name.clone())
                } else {
                    panic!("unknown dimension parameter: {name}")
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                let l = self.compile_dim(lhs);
                let r = self.compile_dim(rhs);
                let dim = match op {
                    BinOpKind::Mul => Dim::Mul(Box::new(l), Box::new(r)),
                    BinOpKind::Div => Dim::Div(Box::new(l), Box::new(r)),
                    BinOpKind::Add => Dim::Add(Box::new(l), Box::new(r)),
                    BinOpKind::Sub => Dim::Sub(Box::new(l), Box::new(r)),
                };
                dim.simplify()
            }
            _ => panic!("expected dimension expression, got {expr:?}"),
        }
    }

    fn compile_shape(&mut self, args: &[Arg], index: usize) -> Vec<Dim> {
        let mut pos = 0;
        for arg in args {
            if let Arg::Positional(expr) = arg {
                if pos == index {
                    return match expr {
                        Expr::Array(elems) => {
                            elems.iter().map(|e| self.compile_dim(e)).collect()
                        }
                        _ => panic!("expected array literal for shape"),
                    };
                }
                pos += 1;
            }
        }
        panic!("missing positional argument at index {index}")
    }

    /// Parse a nested array like [[1, 2], [3, 4]] into Vec<(Dim, Dim)>
    fn compile_pairs(&mut self, args: &[Arg], index: usize) -> Vec<(Dim, Dim)> {
        let mut pos = 0;
        for arg in args {
            if let Arg::Positional(expr) = arg {
                if pos == index {
                    return match expr {
                        Expr::Array(elems) => {
                            elems.iter().map(|e| match e {
                                Expr::Array(pair) => {
                                    assert_eq!(pair.len(), 2, "expected pair [a, b]");
                                    (self.compile_dim(&pair[0]), self.compile_dim(&pair[1]))
                                }
                                _ => panic!("expected array pair"),
                            }).collect()
                        }
                        _ => panic!("expected array of pairs"),
                    };
                }
                pos += 1;
            }
        }
        panic!("missing positional argument at index {index}")
    }
}

pub fn compile(input: &str) -> Graph {
    let items = parser::parse(input);
    Compiler::new().compile_program(&items)
}

/// Compile with concrete values for dimension parameters and named constants.
/// Dims declared with `dim T` in the source will resolve to `Dim::Lit(value)`
/// instead of `Dim::Param("T")` if present in `dims`.
/// Constants are available as named identifiers in expressions.
pub fn compile_with_env(
    input: &str,
    dims: &HashMap<String, usize>,
    constants: &HashMap<String, f64>,
) -> Graph {
    let items = parser::parse(input);
    let mut compiler = Compiler::new();
    compiler.dim_values = dims.clone();
    compiler.constants = constants.clone();
    compiler.compile_program(&items)
}

/// Helper to create a Vec<Dim> from concrete values.
pub fn dims(vals: &[usize]) -> Vec<Dim> {
    vals.iter().map(|&v| Dim::Lit(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let g = compile("let x = 42");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].op, Op::Constant(42.0));
    }

    #[test]
    fn test_add() {
        let g = compile("let x = 1 let y = 2 let z = add(x, y)");
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.nodes[2].op, Op::Add);
        assert_eq!(g.nodes[2].inputs, vec![NodeId(0), NodeId(1)]);
    }

    #[test]
    fn test_sub_lowers_to_neg_add() {
        let g = compile("let x = 1 let y = 2 let z = sub(x, y)");
        // sub(x, y) → add(x, neg(y))
        assert_eq!(g.nodes.len(), 4);
        assert_eq!(g.nodes[2].op, Op::Neg);
        assert_eq!(g.nodes[2].inputs, vec![NodeId(1)]);
        assert_eq!(g.nodes[3].op, Op::Add);
        assert_eq!(g.nodes[3].inputs, vec![NodeId(0), NodeId(2)]);
    }

    #[test]
    fn test_binop_sub_lowers() {
        let g = compile("let x = 1 let y = 2 let z = x - y");
        // same lowering: add(x, neg(y))
        assert_eq!(g.nodes[2].op, Op::Neg);
        assert_eq!(g.nodes[3].op, Op::Add);
    }

    #[test]
    fn test_reduce_sum() {
        let g = compile("let x = load([10, 32]) let s = sum(x, axis: 1)");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[1].op, Op::ReduceSum { axis: 1 });
        assert_eq!(g.nodes[1].inputs, vec![NodeId(0)]);
    }

    #[test]
    fn test_exp_lowers_to_exp2() {
        let g = compile("let x = 1 let e = exp(x)");
        // exp(x) = exp2(x * LOG2_E)
        assert_eq!(g.nodes.len(), 4);
        assert_eq!(g.nodes[1].op, Op::Constant(std::f64::consts::LOG2_E));
        assert_eq!(g.nodes[2].op, Op::Mul);
        assert_eq!(g.nodes[3].op, Op::Exp2);
    }

    #[test]
    fn test_function_call_inlines() {
        let input = r#"
            fn double(x) {
                add(x, x)
            }
            let a = 5
            let b = double(a)
        "#;
        let g = compile(input);
        // a=Const(5), then add(a, a)
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[0].op, Op::Constant(5.0));
        assert_eq!(g.nodes[1].op, Op::Add);
        assert_eq!(g.nodes[1].inputs, vec![NodeId(0), NodeId(0)]);
    }

    #[test]
    fn test_shared_node() {
        // When a variable is used twice, both uses point to the same NodeId
        let g = compile("let x = 1 let y = mul(x, x)");
        assert_eq!(g.nodes[1].inputs, vec![NodeId(0), NodeId(0)]);
    }

    #[test]
    fn test_softmax() {
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

        // Check the graph has the right structure:
        // input, reduce_max, neg, add, const(LOG2E), mul, exp2, reduce_sum, recip, mul
        let ops: Vec<_> = g.nodes.iter().map(|n| &n.op).collect();
        assert!(matches!(ops[0], Op::Input { .. }));
        assert!(matches!(ops[1], Op::ReduceMax { axis: 1 }));
        assert!(matches!(ops[2], Op::Neg));
        assert!(matches!(ops[3], Op::Add));
        // exp lowering: const LOG2E, mul, exp2
        assert!(matches!(ops[4], Op::Constant(_)));
        assert!(matches!(ops[5], Op::Mul));
        assert!(matches!(ops[6], Op::Exp2));
        assert!(matches!(ops[7], Op::ReduceSum { axis: 1 }));
        assert!(matches!(ops[8], Op::Recip));
        assert!(matches!(ops[9], Op::Mul));
    }

    #[test]
    fn test_permute() {
        let g = compile("let x = load([2, 3]) let y = permute(x, [1, 0])");
        assert_eq!(g.nodes[1].op, Op::Permute { order: vec![1, 0] });
    }

    #[test]
    fn test_shape_input() {
        let g = compile("let x = load([4, 10])");
        assert_eq!(g.nodes[0].shape, dims(&[4, 10]));
    }

    #[test]
    fn test_shape_elementwise() {
        let g = compile("let x = load([4, 10]) let y = neg(x)");
        assert_eq!(g.nodes[1].shape, dims(&[4, 10]));
    }

    #[test]
    fn test_shape_reduce() {
        let g = compile("let x = load([4, 10]) let s = sum(x, axis: 1)");
        assert_eq!(g.nodes[1].shape, dims(&[4, 1]));
    }

    #[test]
    fn test_shape_reduce_axis0() {
        let g = compile("let x = load([4, 10]) let s = sum(x, axis: 0)");
        assert_eq!(g.nodes[1].shape, dims(&[1, 10]));
    }

    #[test]
    fn test_shape_broadcast_scalar() {
        // mul(tensor, scalar_constant) should keep tensor shape
        let g = compile("let x = load([4, 10]) let y = mul(x, 2.0)");
        assert_eq!(g.nodes[2].shape, dims(&[4, 10]));
    }

    #[test]
    fn test_shape_permute() {
        let g = compile("let x = load([4, 10]) let y = permute(x, [1, 0])");
        assert_eq!(g.nodes[1].shape, dims(&[10, 4]));
    }

    #[test]
    fn test_shape_softmax() {
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
        // input [4,10], reduce_max [4,1], neg [4,1], add broadcasts to [4,10]
        assert_eq!(g.nodes[0].shape, dims(&[4, 10]));   // input
        assert_eq!(g.nodes[1].shape, dims(&[4, 1]));     // reduce_max
        assert_eq!(g.nodes[2].shape, dims(&[4, 1]));     // neg
        assert_eq!(g.nodes[3].shape, dims(&[4, 10]));    // add (broadcast [4,10] + [4,1])
        // exp chain
        assert_eq!(g.nodes[6].shape, dims(&[4, 10]));    // exp2
        assert_eq!(g.nodes[7].shape, dims(&[4, 1]));     // reduce_sum
        assert_eq!(g.nodes[8].shape, dims(&[4, 1]));     // recip
        assert_eq!(g.nodes[9].shape, dims(&[4, 10]));    // final mul (broadcast [4,1] * [4,10])
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&dims(&[4, 10]), &dims(&[10])), dims(&[4, 10]));
        assert_eq!(broadcast_shapes(&dims(&[4, 10]), &dims(&[1])), dims(&[4, 10]));
        assert_eq!(broadcast_shapes(&dims(&[3, 1, 5]), &dims(&[1, 4, 1])), dims(&[3, 4, 5]));
        assert_eq!(broadcast_shapes(&dims(&[]), &dims(&[4, 10])), dims(&[4, 10]));
        assert_eq!(broadcast_shapes(&dims(&[4, 10]), &dims(&[])), dims(&[4, 10]));
    }

    #[test]
    #[should_panic(expected = "cannot broadcast")]
    fn test_broadcast_shapes_error() {
        broadcast_shapes(&dims(&[3, 4]), &dims(&[5, 4]));
    }

    #[test]
    fn test_matmul_2d() {
        // matmul([2,3], [3,4]) -> [2,4]
        let g = compile("let a = load([2, 3]) let b = load([3, 4]) let c = matmul(a, b)");
        // Last node should have shape [2, 4]
        let last = g.nodes.last().unwrap();
        assert_eq!(last.shape, dims(&[2, 4]));
    }

    #[test]
    fn test_matmul_batched() {
        // matmul([2,3,4], [2,4,5]) -> [2,3,5]
        let g = compile("let a = load([2, 3, 4]) let b = load([2, 4, 5]) let c = matmul(a, b)");
        let last = g.nodes.last().unwrap();
        assert_eq!(last.shape, dims(&[2, 3, 5]));
    }

    #[test]
    fn test_matmul_decomposition() {
        // Verify matmul decomposes to reshape, expand, mul, sum, reshape
        let g = compile("let a = load([2, 3]) let b = load([3, 4]) let c = matmul(a, b)");
        let ops: Vec<_> = g.nodes.iter().map(|n| &n.op).collect();
        // nodes 0,1 are inputs
        // then: reshape A, reshape B, expand A, expand B, mul, sum, reshape out
        assert!(matches!(ops[2], Op::Reshape { .. }));  // A -> [2,3,1]
        assert!(matches!(ops[3], Op::Reshape { .. }));  // B -> [1,3,4]
        assert!(matches!(ops[4], Op::Expand { .. }));   // A -> [2,3,4]
        assert!(matches!(ops[5], Op::Expand { .. }));   // B -> [2,3,4]
        assert!(matches!(ops[6], Op::Mul));
        assert!(matches!(ops[7], Op::ReduceSum { axis: 1 })); // sum over K
        assert!(matches!(ops[8], Op::Reshape { .. }));  // [2,1,4] -> [2,4]
    }

    #[test]
    fn test_arange() {
        let g = compile("let x = arange(5)");
        assert_eq!(g.nodes[0].op, Op::Arange { size: Dim::Lit(5) });
        assert_eq!(g.nodes[0].shape, dims(&[5]));
    }

    #[test]
    fn test_nanogpt_compiles() {
        // Tiny config: B=1, T=4, vocab=8, d=16, heads=2, layers=2
        let g = crate::nanogpt::compile_gpt2(1, Some(4), 8, 16, 2, 2);
        // Last node should be the logits with shape [1, 4, 8] (B, T, vocab)
        let last = g.nodes.last().unwrap();
        assert_eq!(last.shape, dims(&[1, 4, 8]));
        // Count inputs
        let n_inputs = g.nodes.iter().filter(|n| matches!(&n.op, Op::Input { .. })).count();
        assert_eq!(n_inputs, crate::nanogpt::nanogpt_input_count(2));
    }

    #[test]
    fn print_softmax_dot() {
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
        println!("{}", g.to_dot());
    }

    #[test]
    fn test_symbolic_dim_basic() {
        let input = "dim T\nlet x = load([1, T, 768])";
        let g = compile(input);
        assert_eq!(g.nodes[0].shape.len(), 3);
        assert_eq!(g.nodes[0].shape[0], Dim::Lit(1));
        assert_eq!(g.nodes[0].shape[1], Dim::Param("T".into()));
        assert_eq!(g.nodes[0].shape[2], Dim::Lit(768));
    }

    #[test]
    fn test_symbolic_dim_propagates() {
        // Symbolic T should propagate through elementwise ops
        let input = "dim T\nlet x = load([1, T, 768])\nlet y = neg(x)";
        let g = compile(input);
        assert_eq!(g.nodes[1].shape[1], Dim::Param("T".into()));
    }

    #[test]
    fn test_symbolic_dim_reduce() {
        // Reduce on a non-symbolic axis keeps T
        let input = "dim T\nlet x = load([1, T, 768])\nlet y = sum(x, axis: 2)";
        let g = compile(input);
        assert_eq!(g.nodes[1].shape[0], Dim::Lit(1));
        assert_eq!(g.nodes[1].shape[1], Dim::Param("T".into()));
        assert_eq!(g.nodes[1].shape[2], Dim::Lit(1));
    }

    #[test]
    fn test_symbolic_dim_reshape() {
        let input = "dim T\nlet x = load([1, T, 768])\nlet y = reshape(x, [T, 768])";
        let g = compile(input);
        assert_eq!(g.nodes[1].shape.len(), 2);
        assert_eq!(g.nodes[1].shape[0], Dim::Param("T".into()));
        assert_eq!(g.nodes[1].shape[1], Dim::Lit(768));
    }

    #[test]
    fn test_static_gpt2_symbolic_t() {
        // Leave T symbolic; all other dims concrete. Final shape should
        // preserve T as a parameter.
        let g = crate::nanogpt::compile_gpt2(1, None, 8, 16, 2, 2);
        let last = g.nodes.last().unwrap();
        assert_eq!(last.shape.len(), 3);
        assert_eq!(last.shape[0], Dim::Lit(1));
        assert_eq!(last.shape[1], Dim::Param("T".into()));
        assert_eq!(last.shape[2], Dim::Lit(8));
    }

    #[test]
    fn test_symbolic_dim_matmul() {
        // matmul with symbolic batch dim: [1, T, 768] @ [768, 2304]
        let input = "dim T\nlet a = load([1, T, 768])\nlet b = load([768, 2304])\nlet c = matmul(a, b)";
        let g = compile(input);
        let last = g.nodes.last().unwrap();
        // Output should be [1, T, 2304]
        assert_eq!(last.shape.len(), 3);
        assert_eq!(last.shape[0], Dim::Lit(1));
        assert_eq!(last.shape[1], Dim::Param("T".into()));
        assert_eq!(last.shape[2], Dim::Lit(2304));
    }
}
