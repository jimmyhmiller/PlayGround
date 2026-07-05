//! A tiny calculator language, instantiated on TWO value models with the SAME
//! generic engine. The only thing that changes is `Repr::is_immediate`, and
//! that alone decides whether integer-heavy or float-heavy code allocates.
//!
//! This is the concrete form of the Clojure numeric argument: a fixnum-first
//! language wants `LowBit` (integers immediate); a float-first scripting
//! language wants `NanBox`. Picking the wrong one taxes the common case on
//! every arithmetic op. The original toolkit only shipped the NanBox fast path
//! in its authoring layer, so an integer-primary language got the slow column
//! below no matter what.

use microlang::{LowBitModel, NanBoxModel, Runtime, TreeWalk, Val, ValueModel};

fn run<M: ValueModel>(label: &str, src: &str) {
    let mut rt = Runtime::<M>::new();
    let cs = TreeWalk;
    let forms = rt.read_all(src);
    let before = rt.allocs;
    let mut res = rt.encode(Val::Nil);
    for f in forms {
        res = rt.eval_top(&cs, f);
    }
    println!(
        "  [{label:6}] {src:<28} => {:<6}  ({} heap allocs during eval)",
        rt.print(res),
        rt.allocs - before
    );
}

fn main() {
    println!("integer-heavy expression  (fixnum-first language):");
    run::<LowBitModel>("LowBit", "(+ (* 2 3) (* 4 5))");
    run::<NanBoxModel>("NanBox", "(+ (* 2 3) (* 4 5))");

    println!("\nfloat-heavy expression  (scripting-style language):");
    run::<LowBitModel>("LowBit", "(+ (* 2.0 3.0) (* 4.0 5.0))");
    run::<NanBoxModel>("NanBox", "(+ (* 2.0 3.0) (* 4.0 5.0))");

    println!(
        "\nSame engine, same source. The immediate category (one `is_immediate`\n\
         line per model) decides which column pays for boxing. That is the\n\
         entire 'value layout is a free choice' axis, made real and measured."
    );
}
