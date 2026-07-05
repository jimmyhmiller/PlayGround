//! The execution axis, made tangible: one language, two strategies.
//!
//! `TreeWalk` re-dispatches on the `Ir` every time a node runs. `ClosureComp`
//! compiles each `Ir` subtree into a Rust closure once and caches function
//! bodies. Same source, same `Ir`, same `CodeSpace` contract — you swap the
//! value you pass to the runtime. Deopt/tiering in a real toolkit is exactly
//! this: a hotter backend behind the same seam, with the interpreter tier
//! always available as the fallback.

use microlang::{ClosureComp, CodeSpace, LowBitModel, Runtime};

fn run(label: &str, cs: &dyn CodeSpace<LowBitModel>, src: &str) -> String {
    let mut rt = Runtime::<LowBitModel>::new();
    let out = rt.eval_str(cs, src);
    let s = rt.print(out);
    println!("  [{label:11}] => {s}");
    s
}

fn main() {
    let prog = "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 6)";

    println!("same program, two execution tiers:");
    let a = run("TreeWalk", &microlang::TreeWalk, prog);
    let b = run("ClosureComp", &ClosureComp::<LowBitModel>::new(), prog);
    assert_eq!(a, b);

    // compile-once: one body compiled no matter how deep the recursion
    println!("\ncompile-once (ClosureComp):");
    let cc = ClosureComp::<LowBitModel>::new();
    let mut rt = Runtime::<LowBitModel>::new();
    rt.eval_str(&cc, prog);
    println!(
        "  fact recursed 6 deep; {} function body compiled + cached (not per call).",
        cc.compiled_bodies()
    );

    // late binding: a compiled fn calls one defined later
    println!("\nlate binding / forward reference (ClosureComp):");
    let cc2 = ClosureComp::<LowBitModel>::new();
    let mut rt2 = Runtime::<LowBitModel>::new();
    let r = rt2.eval_str(
        &cc2,
        r#"
        (def even? (fn (n) (if (= n 0) true  (odd?  (- n 1)))))
        (def odd?  (fn (n) (if (= n 0) false (even? (- n 1)))))
        (even? 10)
        "#,
    );
    println!(
        "  (even? 10) => {}   ; even? was compiled before odd? existed",
        rt2.print(r)
    );
}
