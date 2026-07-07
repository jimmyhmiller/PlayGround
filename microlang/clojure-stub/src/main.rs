use microlang::{LowBitModel, Runtime, TreeWalk};

fn main() {
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, "(defn sq [n] (* n n)) (println (sq 7)) (sq 7)");
    println!("clojure-stub (second consumer) => {}", rt.print(r));
}
