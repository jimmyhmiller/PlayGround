use microlang::{LowBitModel, Runtime, TreeWalk};
fn run(src: &str) -> String {
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    clojure_stub::clj_str(&rt, r)
}
fn main() { println!("{}", run(&std::env::args().nth(1).unwrap())); }
