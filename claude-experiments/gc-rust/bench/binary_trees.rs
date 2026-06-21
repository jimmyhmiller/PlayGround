// Rust equivalent of examples/binary_trees.gcr — the allocation/tracing
// benchmark. This is the head-to-head that matters: gc-rust's GC allocation +
// tracing vs Rust's Box (malloc/free, freed by Drop at end of each iteration).
// Same shape: enum tree, depth 16, 40 trees, node-count checksum.
enum Tree {
    Leaf,
    Node(Box<Tree>, Box<Tree>),
}
fn make(depth: i64) -> Tree {
    if depth == 0 {
        Tree::Leaf
    } else {
        Tree::Node(Box::new(make(depth - 1)), Box::new(make(depth - 1)))
    }
}
fn check(t: &Tree) -> i64 {
    match t {
        Tree::Leaf => 1,
        Tree::Node(l, r) => 1 + check(l) + check(r),
    }
}
fn main() {
    let max_depth = 16;
    let mut total = 0i64;
    let mut iter = 0;
    while iter < 40 {
        let t = make(max_depth);
        total = total + check(&t);
        iter = iter + 1;
    }
    println!("{}", total);
}
