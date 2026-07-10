use microlang::{LowBitModel, Runtime, TreeWalk};

fn main() {
    let program = r#"
        (defprotocol Shape
          (area [this])
          (name-of [this]))

        (deftype Circle [r])
        (deftype Rect [w h])

        (extend-type Circle Shape
          (area [this] (let [r (field this 0)] (* 3 (* r r))))
          (name-of [this] :circle))
        (extend-type Rect Shape
          (area [this] (* (field this 0) (field this 1)))
          (name-of [this] :rect))

        (defn summarize [shapes]
          (map (fn [s] {:kind (name-of s) :area (area s)}) shapes))

        (summarize [(->Circle 10) (->Rect 3 4) (->Circle 1)])
    "#;
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, program);
    println!("{}", clojure_stub::clj_str(&rt, r));

    for src in [
        "(->> (range 20) (filter even?) (map (fn [x] (* x x))) (reduce + 0))",
        "(get-in {:user {:name :alice :roles [:admin]}} [:user :roles])",
        "(loop [n 5 acc []] (if (= n 0) acc (recur (dec n) (conj acc n))))",
        "(defmacro unless [c a b] `(if ~c ~b ~a)) (unless false :yes :no)",
    ] {
        let mut rt = Runtime::<LowBitModel>::new();
        let r = clojure_stub::run(&mut rt, &TreeWalk, src);
        println!("{src}\n  => {}", clojure_stub::clj_str(&rt, r));
    }
}
