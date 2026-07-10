//! The capstone: a bytecode tier where the VALUE MODEL emits the arithmetic.
//! Same source, three representations, three different instruction streams — and
//! the same answer. This is the "emit half" made concrete: swapping the
//! representation changes the generated code, and nothing in the compiler does.

use microlang::{BytecodeVm, HighBitModel, LowBitModel, ModelEmit, NanBoxModel, Runtime};

fn show<M: ModelEmit>(name: &str, src: &str) {
    // disassemble the emitted bytecode
    let mut rt = Runtime::<M>::new();
    let vm = BytecodeVm::<M>::new();
    let forms = microlang::sexpr::read_all(&mut rt, src);
    let ir = microlang::sexpr::analyze(&mut rt, &vm, forms[0]);
    let ops = BytecodeVm::<M>::disassemble(&ir);
    // and run it
    let mut rt2 = Runtime::<M>::new();
    let vm2 = BytecodeVm::<M>::new();
    let r = microlang::sexpr::eval_str(&mut rt2, &vm2, src);
    println!("  {name:8} => {:<4}  [{}]", rt2.print(r), ops.join("  "));
}

fn main() {
    let src = "(+ (* 2 3) (* 4 5))";
    println!("{src}  — one source, three value representations, three emissions:\n");
    show::<LowBitModel>("LowBit", src);
    show::<HighBitModel>("HighBit", src);
    show::<NanBoxModel>("NanBox", src);
    println!(
        "\nSame answer (26). Different bytecode: LowBit shifts to untag before each\n\
         multiply; HighBit needs no shift (value sits unshifted under a high tag);\n\
         NanBox boxes integers, so arithmetic is a slow runtime call. The compiler\n\
         is generic over the model's `emit_*` interface — swap the representation\n\
         and the generated code changes; the compiler does not. That is the emit\n\
         half of the value axis, and the same shape a machine-code tier would use\n\
         for GC barriers, dispatch guards, and deopt."
    );

    println!("\nrecursion under the bytecode VM (fact 6):");
    let mut rt = Runtime::<LowBitModel>::new();
    let vm = BytecodeVm::<LowBitModel>::new();
    let r = microlang::sexpr::eval_str(&mut rt, 
        &vm,
        "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)",
    );
    println!("  => {}", rt.print(r));
}
