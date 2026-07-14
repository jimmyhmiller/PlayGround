use microlang::{LowBitModel, Runtime, TreeWalk};
fn main() {
    let src = std::env::args().nth(1).unwrap();
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, &src);
    println!("=> {}", clojure_stub::clj_str(&rt, r));
}
