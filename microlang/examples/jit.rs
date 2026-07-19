//! The native emit tier, made tangible.
//!
//! `JitCranelift` is a fourth `CodeSpace`, alongside the tree-walker, the
//! closure-compiler, and the bytecode VM — but it compiles the `Ir` to real host
//! machine code via Cranelift. Same source, same `Ir`, same `CodeSpace` contract:
//! you swap the value you hand the runtime.
//!
//! What this example shows:
//!   1. The JIT computes the same answers as the interpreter tier.
//!   2. Compile-once: one native body per function, cached across recursion.
//!   3. The emitted code differs BY VALUE MODEL from one source — the same
//!      property `bytecode.rs` demonstrated for its op stream, now in Cranelift
//!      IR (LowBit shifts to untag before `imul`; HighBit does a bare `imul`;
//!      NanBox boxes integers, so it emits a runtime call instead).

use microlang::ir::Ir;
use microlang::{
    CodeSpace, HighBitModel, JitCranelift, LowBitModel, ModelArithJit, NanBoxModel, Runtime,
    TreeWalk,
};

const FACT: &str = "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 6)";

fn eval<M: ModelArithJit>(cs: &dyn CodeSpace<M>, src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let r = microlang::sexpr::eval_str(&mut rt, cs, src);
    rt.print(r)
}

fn main() {
    println!("== 1. same program, interpreter vs. native JIT ==");
    let interp = {
        let mut rt = Runtime::<LowBitModel>::new();
        let r = microlang::sexpr::eval_str(&mut rt, &TreeWalk, FACT);
        rt.print(r)
    };
    let jit = eval(&JitCranelift::<LowBitModel>::new(), FACT);
    println!("  (fact 6)  tree-walk => {interp}");
    println!("  (fact 6)  cranelift => {jit}");
    assert_eq!(interp, jit);

    println!("\n== 2. compile-once (native bodies reused across 6-deep recursion) ==");
    let cs = JitCranelift::<LowBitModel>::new();
    let mut rt = Runtime::<LowBitModel>::new();
    microlang::sexpr::eval_str(&mut rt, &cs, FACT);
    println!(
        "  fact recursed 6 deep; {} function bodies compiled to machine code (not per call).",
        cs.compiled_bodies()
    );

    println!("\n== 3. the value model decides the native arithmetic strategy ==");
    println!("  fast-path for (fn (n m) (* n m)), by value model:");
    for (name, ir) in [
        ("LowBit ", model_mul_ir::<LowBitModel>()),
        ("HighBit", model_mul_ir::<HighBitModel>()),
        ("NanBox ", model_mul_ir::<NanBoxModel>()),
    ] {
        // The untag shape is the value-axis signature: LowBit shifts the low tag
        // off (`sshr`), HighBit sign-extends the low 61 bits (`ishl`+`sshr`), and
        // NanBox boxes integers so it has NO immediate-int fast path (its guard is
        // a constant-false `iconst 0`, so every integer multiply is a runtime call).
        let has = |mn: &str| ir.lines().any(|l| l.trim().starts_with(&format!("v")) && l.contains(mn));
        let boxed = ir.contains("iconst.i64 0") && !ir.lines().any(|l| l.contains("sshr"));
        let shape = if boxed {
            "no fast path — integers are boxed; multiply is a runtime call".to_string()
        } else {
            let mut ops = Vec::new();
            for mn in ["ishl", "sshr", "imul", "ireduce"] {
                if has(mn) {
                    ops.push(mn);
                }
            }
            format!("fast path uses {}", ops.join(" + "))
        };
        println!("  {name}: {shape}");
    }

    println!("\n== 4. arithmetic agrees across all three value models ==");
    let arith = "(+ (* 2 3) (* 4 5))";
    let a = eval(&JitCranelift::<LowBitModel>::new(), arith);
    let b = eval(&JitCranelift::<HighBitModel>::new(), arith);
    let c = eval(&JitCranelift::<NanBoxModel>::new(), arith);
    println!("  LowBit={a}  HighBit={b}  NanBox={c}");
    assert!(a == "26" && b == "26" && c == "26");

    println!("\n== 5. the full numeric tower: overflow promotes to BigInt on the JIT ==");
    // The guarded fast path range-checks and falls back to the promoting runtime,
    // so the JIT matches the tree-walker — unlike the wrapping bytecode tier.
    let big = "(* 100000000000 100000000000)";
    let jit_big = eval(&JitCranelift::<LowBitModel>::new(), big);
    let tw_big = {
        let mut rt = Runtime::<LowBitModel>::new();
        let v = microlang::sexpr::eval_str(&mut rt, &TreeWalk, big);
        rt.print(v)
    };
    println!("  (* 10^11 10^11)  tree-walk => {tw_big}");
    println!("  (* 10^11 10^11)  cranelift => {jit_big}");
    assert_eq!(jit_big, tw_big);
    assert_eq!(jit_big, "10000000000000000000000");

    println!("\nall checks passed.");
}

/// Compile the body of `(fn (n m) (* n m))` and return the emitted Cranelift IR.
fn model_mul_ir<M: ModelArithJit>() -> String {
    let mut rt = Runtime::<M>::new();
    let forms = microlang::sexpr::read_all(&mut rt, "(fn (n m) (* n m))");
    let cs = JitCranelift::<M>::new();
    let ir = microlang::sexpr::analyze(&mut rt, &cs, forms[0]);
    match &ir {
        Ir::Lambda { body, .. } => cs.dump_ir(body),
        _ => cs.dump_ir(&ir),
    }
}
