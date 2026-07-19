//! Verifies discriminant-aware enum walking against a live heap. Retains boxes
//! of an enum-bearing `Node`; each node's active `Shape` variant has a known
//! number of heap pointers, and `next` is a niche `Option<Box<Node>>`.
//!
//! Expected (sound + complete): edges out of each Node go to EXACTLY its active
//! variant's boxes — Rect=2, Circle=1, Point=0 — plus its `Some(next)` if any.
//! No false edges from inactive variants.

use memscope::{MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

#[derive(Debug)]
#[allow(dead_code)]
enum Shape {
    Circle { radius: Box<f64> },       // 1 pointer
    Rect { w: Box<u64>, h: Box<u64> }, // 2 pointers
    Point,                             // 0 pointers
}

#[derive(Debug)]
#[allow(dead_code)]
struct Node {
    tag: u64,
    shape: Shape,
    next: Option<Box<Node>>,
}

fn main() {
    memscope::set_mode(Mode::Full);
    let sock = memscope::start_agent().expect("agent");
    println!("enumserve: attach with");
    println!("    cargo run -p memscope-cli --release -- graph --sock {sock}");

    // 100 of each shape, no chaining (next = None) to isolate the variant test.
    let mut nodes: Vec<Box<Node>> = Vec::new();
    for i in 0..100u64 {
        nodes.push(Box::new(Node {
            tag: i,
            shape: Shape::Rect {
                w: Box::new(i),
                h: Box::new(i + 1),
            },
            next: None,
        }));
        nodes.push(Box::new(Node {
            tag: i,
            shape: Shape::Circle {
                radius: Box::new(i as f64),
            },
            next: None,
        }));
        nodes.push(Box::new(Node {
            tag: i,
            shape: Shape::Point,
            next: None,
        }));
    }

    // One short chain to exercise Some(next).
    let chain = Box::new(Node {
        tag: 999,
        shape: Shape::Point,
        next: Some(Box::new(Node {
            tag: 1000,
            shape: Shape::Point,
            next: None,
        })),
    });
    nodes.push(chain);

    let expected_shape_edges = 100 * 2 + 100 * 1 + 100 * 0; // Rect=2, Circle=1, Point=0
    println!("enumserve: built {} nodes; expected shape edges = {expected_shape_edges}, plus 1 Some(next) edge", nodes.len());

    std::hint::black_box(&nodes);
    loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
