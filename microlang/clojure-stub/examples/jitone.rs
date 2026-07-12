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
    println!("{}", run(&std::env::args().nth(1).unwrap()));
}
