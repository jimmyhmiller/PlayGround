//! Single-expression runner on the native JIT tier (mirror of `one.rs`), for
//! shell-driven stress testing. Reads the program from argv[1].
#![cfg(feature = "jit")]

use microlang::jit_cranelift::JitCranelift;
use microlang::{LowBitModel, Runtime};

fn run(src: &str) -> String {
    let mut rt = Runtime::<LowBitModel>::new();
    let backend = JitCranelift::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &backend, src);
    clojure_stub::clj_str(&rt, r)
}

fn main() {
    let arg = std::env::args().nth(1).unwrap();
    // A readable file path -> run its contents (keeps shell/lldb args clean);
    // otherwise treat the arg itself as the program.
    let src = std::fs::read_to_string(&arg).unwrap_or(arg);
    println!("{}", run(&src));
}
