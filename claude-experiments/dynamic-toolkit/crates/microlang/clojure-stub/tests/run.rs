use microlang::{LowBitModel, Runtime, TreeWalk};

#[test]
fn second_consumer_runs() {
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, "(defn sq [n] (* n n)) (sq 7)");
    assert_eq!(rt.print(r), "49");
}
