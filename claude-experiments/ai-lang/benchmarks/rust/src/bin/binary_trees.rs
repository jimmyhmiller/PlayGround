// Binary trees, allocation stress — mirrors benchmarks/ail/binary_trees.ail.
// Plain Box allocation per node (the idiomatic Rust equivalent of
// ai-lang's per-node GC allocation).

enum Tree {
    Branch(Box<Tree>, Box<Tree>),
    Leaf,
}

fn make(d: i64) -> Tree {
    if d == 0 {
        Tree::Leaf
    } else {
        Tree::Branch(Box::new(make(d - 1)), Box::new(make(d - 1)))
    }
}

fn check(t: &Tree) -> i64 {
    match t {
        Tree::Leaf => 1,
        Tree::Branch(l, r) => 1 + check(l) + check(r),
    }
}

fn main() {
    let (depth, iters) = (16i64, 40i64);
    let t0 = std::time::Instant::now();
    let mut acc: i64 = 0;
    for _ in 0..iters {
        acc += check(&make(depth));
    }
    let ms = t0.elapsed().as_millis();
    println!("RESULT binary_trees {} ms checksum={}", ms, acc);
}
