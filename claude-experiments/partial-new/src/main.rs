//! A principled, generic partial evaluator, demonstrated by "compiling"
//! Brainfuck via the first Futamura projection.
//!
//! `engine.rs` is the entire generic specializer. `bf.rs` is one client. The
//! engine never mentions Brainfuck; a second client would reuse it verbatim.

mod bf;
mod engine;
mod residual;

fn demo(name: &str, src: &str, input: &[u8], show_residual: bool) {
    println!("================================================================");
    println!("  {name}");
    println!("================================================================");

    let bf = bf::Bf::new(src);
    let prog = engine::specialize(&bf, bf::State::start());

    println!(
        "source commands : {}\nresidual blocks : {}\nresidual ops    : {}",
        bf.prog_len(),
        prog.blocks.len(),
        prog.op_count(),
    );

    if show_residual {
        println!("\n--- residual program (no pc, no dispatch; pure tape ops) ---");
        print!("{prog}");
    }

    // Prove the partial evaluation preserved semantics.
    let reference = bf::run_reference(src, input);
    let residual = bf::run_residual(&prog, input);
    assert_eq!(
        reference, residual,
        "residual output diverged from the reference interpreter!"
    );

    println!("\noutput          : {:?}", String::from_utf8_lossy(&residual));
    println!("matches oracle  : yes ({} bytes)\n", residual.len());
}

fn main() {
    // 1. A run of `+` coalesces into one residual op (partially-static cell).
    demo("constant: print '0'", &("+".repeat(48) + "."), &[], true);

    // 2. A loop: 3 * 3 = 9 (tab). The dispatch melts; the BF loop survives as a
    //    residual loop whose back-edge was tied by memoization.
    demo("loop: 3*3 via [>+++<-]", "+++[>+++<-]>.", &[], true);

    // 3. Pointer motion coalesces across the run (partially-static pointer).
    demo("pointer coalescing", ">>>+++<<<++.>>>.", &[], true);

    // 4. The classic: Hello World. Many nested loops, all tied off; output is
    //    verified byte-for-byte against the reference interpreter.
    let hello = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]\
                 >>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    demo("Hello World", hello, &[], false);

    // 5. echo: read until 0, write back. Demonstrates I/O residualization and a
    //    data-dependent loop over dynamic input.
    demo("cat (echo input)", ",[.,]", b"partial!\0", false);

    println!("All demos matched the reference interpreter.");
}
