//! Proof of orthogonality: EVERY combination of the axes runs, and all agree.
//!
//! Two programs cover the full cross product:
//!   * arithmetic + recursion, across {3 value representations} × {3 execution
//!     tiers, including the bytecode emit tier};
//!   * records + methods + dispatch + a mid-program MOVING GC, across
//!     {3 representations} × {2 general tiers} × {6 dispatch strategies}.
//! A moving collector runs underneath the whole thing. 45 combinations; one
//! answer each.

use microlang::{
    AlwaysMonomorphic, BlacklistAfter, BytecodeVm, ClosureComp, CodeSpace, Dispatch, HighBitModel,
    LowBitModel, Megamorphic, ModelEmit, MonomorphicIc, NanBoxModel, NeverSpeculate, PolymorphicIc,
    Runtime, Speculative, TreeWalk,
};

const ARITH: &str = "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)";
const FEATURE: &str = concat!(
    "(defmethod area Circle (fn (s) (* (field s 0) (field s 0))))",
    "(defmethod area Square (fn (s) (+ (field s 0) (field s 0))))",
    "(def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))",
    "(def shapes (list (record 'Circle 3) (record 'Square 4) (record 'Circle 5) (record 'Square 6)))",
    "(gc)",
    "(total shapes)"
);

const TIERS: [&str; 3] = ["treewalk", "closure ", "bytecode"];
const DISPATCH: [&str; 6] = [
    "megamorphic",
    "mono-ic",
    "poly-ic(4)",
    "spec/never",
    "spec/always",
    "spec/blacklist2",
];

fn tier<M: ModelEmit>(i: usize) -> Box<dyn CodeSpace<M>> {
    match i {
        0 => Box::new(TreeWalk),
        1 => Box::new(ClosureComp::<M>::new()),
        _ => Box::new(BytecodeVm::<M>::new()),
    }
}
fn dispatch(i: usize) -> Box<dyn Dispatch> {
    match i {
        0 => Box::new(Megamorphic::new()),
        1 => Box::new(MonomorphicIc::new()),
        2 => Box::new(PolymorphicIc::new(4)),
        3 => Box::new(Speculative::new(Megamorphic::new(), NeverSpeculate)),
        4 => Box::new(Speculative::new(Megamorphic::new(), AlwaysMonomorphic)),
        _ => Box::new(Speculative::new(Megamorphic::new(), BlacklistAfter(2))),
    }
}

fn model<M: ModelEmit>(name: &str, ok: &mut usize) {
    // arithmetic across all three tiers
    for t in 0..3 {
        let mut rt = Runtime::<M>::new();
        let r = rt.eval_str(tier::<M>(t).as_ref(), ARITH);
        let s = rt.print(r);
        let pass = s == "720";
        *ok += pass as usize;
        println!("  {name:7} {}          => {s:<4} {}", TIERS[t], if pass { "ok" } else { "FAIL" });
    }
    // feature program across the two general tiers × six dispatch strategies
    for t in 0..2 {
        for d in 0..6 {
            let mut rt = Runtime::<M>::new();
            rt.set_dispatch(dispatch(d));
            let r = rt.eval_str(tier::<M>(t).as_ref(), FEATURE);
            let s = rt.print(r);
            let pass = s == "54";
            *ok += pass as usize;
            println!(
                "  {name:7} {} {:<16} => {s:<4} {}",
                TIERS[t],
                DISPATCH[d],
                if pass { "ok" } else { "FAIL" }
            );
        }
    }
}

fn main() {
    let mut ok = 0;
    println!("value-representation × execution-tier × dispatch-strategy, moving GC underneath:\n");
    model::<LowBitModel>("LowBit", &mut ok);
    model::<NanBoxModel>("NanBox", &mut ok);
    model::<HighBitModel>("HighBit", &mut ok);
    println!("\n{ok}/45 combinations agree.");
    println!(
        "\nThree value layouts, three execution tiers (interpreter, closure-compiler,\n\
         bytecode VM), four dispatch strategies (two with three speculation policies),\n\
         a moving collector under all of it. The axes are orthogonal: any combination\n\
         is a valid program, and they all compute the same answer."
    );
}
