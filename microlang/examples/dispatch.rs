//! The dispatch axis: one polymorphic program, three swappable strategies.
//!
//! `total` sums `(area shape)` over a list of alternating record types. The one
//! `(area ...)` call site is hit four times, seeing Circle, Square, Circle,
//! Square. Every strategy computes the same answer; they differ only in the
//! per-site inline cache, which you swap with one `set_dispatch` call and
//! nothing else changes.

use microlang::{Dispatch, LowBitModel, Megamorphic, MonomorphicIc, PolymorphicIc, Runtime, TreeWalk};

const SHAPES: &str = r#"
    (defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
    (defmethod area Square (fn (s) (+ (field s 0) (field s 0))))
    (def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))
    (total (list (record 'Circle 3) (record 'Square 4) (record 'Circle 5) (record 'Square 6)))
"#;

fn run(label: &str, d: Box<dyn Dispatch>) {
    let mut rt = Runtime::<LowBitModel>::new();
    rt.set_dispatch(d);
    let cs = TreeWalk;
    let r = rt.eval_str(&cs, SHAPES);
    let s = rt.dispatch_stats();
    println!(
        "  [{label:14}] => {}    (hits {}, misses {})",
        rt.print(r),
        s.hits,
        s.misses
    );
}

fn main() {
    println!("area over [Circle Square Circle Square], three dispatch strategies:");
    run("Megamorphic", Box::new(Megamorphic::new()));
    run("MonomorphicIc", Box::new(MonomorphicIc::new()));
    run("PolymorphicIc", Box::new(PolymorphicIc::new(4)));
    println!(
        "\nSame answer, same call site. The megamorphic strategy looks up every call;\n\
         the monomorphic cache thrashes on alternating types (all misses); the\n\
         polymorphic cache holds both and hits after warmup. Swapping the strategy\n\
         was one `set_dispatch` call — dispatch is a free axis, like value and GC."
    );
}
