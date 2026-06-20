//! Every `examples/*.coil` must compile (front-end + codegen) — a CI guard so a
//! core change can't silently break an example. (The Never regression that broke
//! examples/defer.coil slipped through precisely because no test compiled the
//! examples.) Uses emit_ir = read→expand→resolve→check→mono→codegen (no link), so
//! it catches both type errors and LLVM-verification/codegen bugs.

#[test]
fn all_examples_compile() {
    // Compile on a thread with a large stack: the compiler recurses over the AST,
    // and a big example (calc) exceeds the 2 MiB default test-thread stack (it's
    // fine on the 8 MiB main thread — `coil run` works). 64 MiB matches release use.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run_all)
        .expect("spawn")
        .join()
        .expect("examples compile thread");
}

fn run_all() {
    let mut compiled = 0;
    for entry in std::fs::read_dir("examples").expect("read examples/") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("coil") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read example");
        if let Err(e) = coil::emit_ir(&src) {
            panic!("example {} failed to compile:\n{e}", path.display());
        }
        compiled += 1;
    }
    assert!(compiled > 0, "no examples found to compile");
}
