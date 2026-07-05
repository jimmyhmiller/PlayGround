//! One standard IR, executed two ways. A frontend lowers source to `Ir` ONCE
//! (axis-neutral: `Prim(Add)`, `Call`, `Global` — not `AddRawLowBit`), then the
//! SAME `Ir` value is handed to an interpreter and to the bytecode JIT. Both
//! consume it; both agree. This is the "standard IR for interpretation and JIT"
//! the toolkit is built around — it is what `CodeSpace` means.

use microlang::{BytecodeVm, CodeSpace, LowBitModel, Runtime, TreeWalk};
use microlang::value::Locals;

fn main() {
    let mut rt = Runtime::<LowBitModel>::new();

    // A helper function lives in the global environment.
    rt.eval_str(&TreeWalk, "(def sq (fn (n) (* n n)))");

    // Lower ONE expression to the standard IR, once. `ir` is axis-neutral: it
    // says (+ (sq 6) (sq 4)) in terms of Call/Global/Prim, committing to no
    // value layout, no dispatch, no execution strategy.
    let form = rt.read_all("(+ (sq 6) (sq 4))")[0];
    let ir = rt.analyze(&TreeWalk, form);

    let top: Locals = None;

    // Execute that SAME `ir` with an interpreter (tree-walk).
    let interp = TreeWalk;
    let r_interp = interp.eval_ir(&interp, &mut rt, &ir, &top);

    // Execute the SAME `ir` with the bytecode JIT (lowers to bytecode + runs).
    let jit = BytecodeVm::<LowBitModel>::new();
    let r_jit = jit.eval_ir(&jit, &mut rt, &ir, &top);

    println!("source     : (+ (sq 6) (sq 4))");
    println!("standard IR : one axis-neutral `Ir` value, produced once");
    println!("  interpreter (tree-walk)  => {}", rt.print(r_interp));
    println!("  JIT (bytecode compiler)  => {}", rt.print(r_jit));
    assert_eq!(rt.print(r_interp), rt.print(r_jit));

    println!(
        "\nThe SAME `Ir` value fed both executors. Even the `sq` it calls is\n\
         tree-walked by one and bytecode-compiled by the other — same answer.\n\
         The IR is the shared, axis-neutral contract; the executor (interpret or\n\
         JIT) and the value/dispatch/GC strategies are applied on top of it. That\n\
         separation is exactly why one IR serves every combination."
    );
}
