//! Forces a few enum shapes into the binary so we can inspect their DWARF
//! variant_part representation (direct-tag, niche/Option, fieldless).

#[derive(Debug)]
#[allow(dead_code)]
enum Shape {
    Circle { radius: Box<f64> },        // data variant with a pointer
    Rect { w: Box<u64>, h: Box<u64> },  // data variant with two pointers
    Point,                              // fieldless
}

#[derive(Debug)]
#[allow(dead_code)]
struct Node {
    tag: u8,
    shape: Shape,
    next: Option<Box<Node>>, // niche-optimized nullable pointer
}

fn main() {
    let n = Node {
        tag: 1,
        shape: Shape::Rect {
            w: Box::new(3),
            h: Box::new(4),
        },
        next: Some(Box::new(Node {
            tag: 2,
            shape: Shape::Circle {
                radius: Box::new(1.5),
            },
            next: None,
        })),
    };
    std::hint::black_box(&n);
    println!("{:?}", n.tag);
}
