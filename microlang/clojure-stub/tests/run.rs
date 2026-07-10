//! The mini-Clojure frontend end to end: reader (`[] {} #{} :kw`), its own macro
//! expander + `clojure.core` (seqs, collections, HOFs). Output via the frontend's
//! Clojure formatter.

use microlang::{LowBitModel, Runtime, TreeWalk};

fn run(src: &str) -> String {
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    clojure_stub::clj_str(&rt, r)
}

#[test]
fn macros_and_fns() {
    assert_eq!(run("(defn fact [n] (if (< n 2) 1 (* n (fact (- n 1))))) (fact 5)"), "120");
    assert_eq!(run("(when true 1 2 3)"), "3");
    assert_eq!(run("(cond (< 3 2) :a (< 5 4) :b :else :c)"), ":c");
    assert_eq!(run("(and 1 2 3)"), "3");
    assert_eq!(run("(or false nil 7)"), "7");
}

#[test]
fn vectors_and_seqs() {
    assert_eq!(run("[1 2 3]"), "[1 2 3]");
    assert_eq!(run("(map inc [1 2 3])"), "(2 3 4)");
    assert_eq!(run("(filter even? (range 10))"), "(0 2 4 6 8)");
    assert_eq!(run("(reduce + 0 (range 5))"), "10");
    assert_eq!(run("(count [10 20 30])"), "3");
    assert_eq!(run("(nth [10 20 30] 1)"), "20");
    assert_eq!(run("(conj [1 2] 3)"), "[1 2 3]");
    assert_eq!(run("(first [1 2 3])"), "1");
    assert_eq!(run("(rest [1 2 3])"), "(2 3)");
    assert_eq!(run("(reverse [1 2 3])"), "(3 2 1)");
    assert_eq!(run("(into [] (map inc [1 2 3]))"), "[2 3 4]");
}

#[test]
fn keywords_and_maps() {
    assert_eq!(run(":hello"), ":hello");
    assert_eq!(run("(= :a :a)"), "true");
    assert_eq!(run("(= :a :b)"), "false");
    assert_eq!(run("(get {:a 1 :b 2} :b)"), "2");
    assert_eq!(run("(get {:a 1} :missing)"), "nil");
    assert_eq!(run("(contains? {:a 1} :a)"), "true");
    assert_eq!(run("(get (assoc {:a 1} :b 2) :b)"), "2");
    assert_eq!(run("(keys {:a 1 :b 2})"), "(:a :b)");
    assert_eq!(run("(count {:a 1 :b 2})"), "2");
}

#[test]
fn equality_is_structural() {
    assert_eq!(run("(= [1 2 3] [1 2 3])"), "true");
    assert_eq!(run("(= [1 2] [1 2 3])"), "false");
    assert_eq!(run(r#"(= "ab" "ab")"#), "true");
}

#[test]
fn syntax_quote_macros() {
    // ` ~ with a plain template
    assert_eq!(run("(defmacro unless [c a b] `(if ~c ~b ~a)) (unless false 1 2)"), "1");
    // ~@ splicing
    assert_eq!(run("(defmacro lst [& xs] `(list ~@xs)) (lst 1 2 3)"), "(1 2 3)");
    // a macro that builds a vector template
    assert_eq!(run("(defmacro pair [a b] `[~a ~b]) (pair 1 2)"), "[1 2]");
    // auto-gensym: the `x#` is one fresh symbol, so this let is hygienic-ish
    assert_eq!(
        run("(defmacro twice [e] `(let [x# ~e] (+ x# x#))) (twice 21)"),
        "42"
    );
    // syntax-quote nests unquoted computation
    assert_eq!(run("(defmacro add [a b] `(+ ~a ~b)) (add 20 22)"), "42");
}

#[test]
fn destructuring() {
    // sequential
    assert_eq!(run("(let [[a b c] [1 2 3]] (+ a (+ b c)))"), "6");
    // nested
    assert_eq!(run("(let [[a [b c]] [1 [2 3]]] (+ a (* b c)))"), "7");
    // rest
    assert_eq!(run("(let [[x & more] [1 2 3 4]] more)"), "(2 3 4)");
    // map :keys
    assert_eq!(run("(let [{:keys [a b]} {:a 10 :b 20}] (+ a b))"), "30");
    // fn param destructuring
    assert_eq!(run("(defn sum-pair [[a b]] (+ a b)) (sum-pair [3 4])"), "7");
    assert_eq!(run("(defn f [{:keys [x y]}] (* x y)) (f {:x 5 :y 6})"), "30");
    // ignore with _
    assert_eq!(run("(let [[_ b] [1 2]] b)"), "2");
}

#[test]
fn loop_recur() {
    assert_eq!(run("(loop [i 0 acc 0] (if (< i 5) (recur (inc i) (+ acc i)) acc))"), "10");
    // O(1) stack: a big loop must not overflow (trampolined tail recur)
    assert_eq!(run("(loop [n 100000 acc 0] (if (= n 0) acc (recur (dec n) (inc acc))))"), "100000");
    assert_eq!(run("(defn count-up [n] (loop [i 0 out []] (if (< i n) (recur (inc i) (conj out i)) out))) (count-up 4)"), "[0 1 2 3]");
}

/// The LITERAL clojure/core.clj bootstrap forms run on our interop shim: `fn*`,
/// `(. clojure.lang.RT …)` interop, `Class/method`, `^{…}` metadata, value-less
/// `def`, and multi-arity `conj`. This is the actual Clojure source, unedited.
#[test]
fn real_core_clj_bootstrap() {
    const BOOTSTRAP: &str = r#"
        (def unquote)
        (def
         ^{:arglists '([x seq]) :doc "cons" :static true}
         cons (fn* ^:static cons [x seq] (. clojure.lang.RT (cons x seq))))
        (def
         ^{:arglists '([coll]) :static true}
         first (fn ^:static first [coll] (. clojure.lang.RT (first coll))))
        (def
         ^{:arglists '([coll]) :static true}
         next (fn ^:static next [x] (. clojure.lang.RT (next x))))
        (def
         ^{:arglists '([coll]) :static true}
         rest (fn ^:static rest [x] (. clojure.lang.RT (more x))))
        (def
         ^{:doc "Same as (first (next x))" :static true}
         second (fn ^:static second [x] (first (next x))))
        (def
         ^{:arglists '([coll]) :static true}
         seq (fn ^:static seq [coll] (. clojure.lang.RT (seq coll))))
        (def
         ^{:arglists '([] [coll] [coll x] [coll x & xs]) :static true}
         conj (fn ^:static conj
                ([] [])
                ([coll] coll)
                ([coll x] (clojure.lang.RT/conj coll x))
                ([coll x & xs]
                 (if xs
                   (recur (clojure.lang.RT/conj coll x) (first xs) (next xs))
                   (clojure.lang.RT/conj coll x)))))
    "#;
    let run = |expr: &str| -> String {
        let mut rt = Runtime::<LowBitModel>::new();
        let src = format!("{BOOTSTRAP}\n{expr}");
        let r = clojure_stub::run(&mut rt, &TreeWalk, &src);
        clojure_stub::clj_str(&rt, r)
    };
    // cons / first / next / rest / second (all via RT interop)
    assert_eq!(run("(second (cons 1 (cons 2 nil)))"), "2");
    assert_eq!(run("(first (cons 10 nil))"), "10");
    // seq over a vector (RT/seq -> our host runtime)
    assert_eq!(run("(seq [1 2 3])"), "(1 2 3)");
    assert_eq!(run("(first [7 8])"), "7");
    // multi-arity conj (0/1/2 args) + the VARIADIC arity (recur-in-fn)
    assert_eq!(run("(conj)"), "[]");
    assert_eq!(run("(conj [9])"), "[9]");
    assert_eq!(run("(conj [1 2] 3)"), "[1 2 3]");
    assert_eq!(run("(conj (cons 1 nil) 0)"), "(0 1)");
    assert_eq!(run("(conj [1] 2 3 4)"), "[1 2 3 4]"); // variadic: recur re-enters the arity
}

/// A macro defined the REAL clojure.core way: `(def ^{:macro true} name (fn*
/// [&form &env & args] …))`. Exercises `:macro` metadata + the `&form`/`&env`
/// calling convention on literal source.
#[test]
fn real_core_clj_macro_def() {
    let src = r#"
        (def
         ^{:macro true}
         my-when (fn* my-when [&form &env test & body]
                   (cons 'if (cons test (cons (cons 'do body) nil)))))
        (my-when true 1 2 42)
    "#;
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    assert_eq!(clojure_stub::clj_str(&rt, r), "42");
}

/// The LITERAL real `assoc` def (multi-arity + `clojure.lang.RT/assoc` + recur).
/// The functional layer of core.clj loads unedited via the shim.
#[test]
fn real_core_clj_assoc() {
    let src = r#"
        (def
         ^{:arglists '([map key val] [map key val & kvs]) :static true}
         assoc
         (fn ^:static assoc
           ([map key val] (clojure.lang.RT/assoc map key val))
           ([map key val & kvs]
            (let [ret (clojure.lang.RT/assoc map key val)]
              (if kvs
                (if (next kvs)
                  (recur ret (first kvs) (second kvs) (nnext kvs))
                  (throw "assoc expects even args"))
                ret)))))
        (get (assoc (assoc {} :a 1) :b 2) :b)
    "#;
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    assert_eq!(clojure_stub::clj_str(&rt, r), "2");
}

/// Metadata as values: `with-meta`/`meta` work AND stay transparent — a
/// metadata'd vector is still a vector, its contents read unchanged.
#[test]
fn metadata() {
    assert_eq!(run("(meta (with-meta [1 2] {:a 1}))"), "{:a 1}");
    assert_eq!(run("(meta [1 2])"), "nil");
    assert_eq!(run("(vector? (with-meta [1 2] {:a 1}))"), "true"); // transparent
    assert_eq!(run("(first (with-meta [7 8 9] {:x 1}))"), "7"); // ops see through meta
    assert_eq!(run("(count (with-meta [7 8 9] {:x 1}))"), "3");
    assert_eq!(run("(meta (quote sym))"), "nil"); // non-collection: no meta
    // keyword-as-function in head position: (:k m) -> (get m :k)
    assert_eq!(run("(:a {:a 1 :b 2})"), "1");
    assert_eq!(run("(:missing {:a 1})"), "nil");
}

#[test]
fn protocols() {
    // a user type + a protocol implemented for it
    let shape = r#"
        (defprotocol Shape (area [this]) (perimeter [this]))
        (deftype Rect [w h])
        (extend-type Rect Shape
          (area [this] (* (field this 0) (field this 1)))
          (perimeter [this] (* 2 (+ (field this 0) (field this 1)))))
    "#;
    assert_eq!(run(&format!("{shape} (area (->Rect 3 4))")), "12");
    assert_eq!(run(&format!("{shape} (perimeter (->Rect 3 4))")), "14");

    // polymorphism: two types, same protocol, dispatch by type
    let poly = r#"
        (defprotocol Describe (describe [this]))
        (deftype Dog [name])
        (deftype Cat [name])
        (extend-type Dog Describe (describe [this] :woof))
        (extend-type Cat Describe (describe [this] :meow))
    "#;
    assert_eq!(run(&format!("{poly} (describe (->Dog :rex))")), ":woof");
    assert_eq!(run(&format!("{poly} (describe (->Cat :tom))")), ":meow");

    // extend a BUILT-IN type (vectors are records tagged 'Vector)
    let ext = r#"
        (defprotocol Sized (size [this]))
        (extend-type Vector Sized (size [this] (count this)))
    "#;
    assert_eq!(run(&format!("{ext} (size [10 20 30])")), "3");
}

#[test]
fn core_breadth() {
    assert_eq!(run("(get-in {:a {:b {:c 42}}} [:a :b :c])"), "42");
    assert_eq!(run("(assoc-in {:a {:b 1}} [:a :b] 99)"), "{:a {:b 99}}");
    assert_eq!(run("(update {:n 5} :n inc)"), "{:n 6}");
    assert_eq!(run("(some even? [1 3 5 6])"), "6");
    assert_eq!(run("(every? even? [2 4 6])"), "true");
    assert_eq!(run("(mapv inc [1 2 3])"), "[2 3 4]");
    assert_eq!(run("((comp inc inc) 5)"), "7");
    assert_eq!(run("(map (partial + 10) [1 2 3])"), "(11 12 13)");
    assert_eq!(run("(if-let [x (get {:a 1} :a)] x :none)"), "1");
    assert_eq!(run("(if-let [x (get {:a 1} :missing)] x :none)"), ":none");
    assert_eq!(run("(map-indexed (fn [i x] [i x]) [:a :b])"), "([0 :a] [1 :b])");
}

#[test]
fn threading_and_composition() {
    assert_eq!(run("(-> {:a 1} (assoc :b 2) (get :b))"), "2");
    assert_eq!(run("(-> [1 2 3] (conj 4) count)"), "4");
    assert_eq!(run("(->> (range 5) (filter even?) (map inc))"), "(1 3 5)");
}

#[test]
fn callable_objects() {
    // Keywords are functions of a map.
    assert_eq!(run("(:n {:n 7})"), "7");
    assert_eq!(run("(map :n [{:n 1} {:n 2} {:n 3}])"), "(1 2 3)");
    // Maps and vectors are functions of a key/index.
    assert_eq!(run("({:a 1 :b 2} :b)"), "2");
    assert_eq!(run("([:x :y :z] 1)"), ":y");
}

#[test]
fn instance_predicate() {
    assert_eq!(run("(instance? clojure.lang.Keyword :a)"), "true");
    assert_eq!(run("(instance? clojure.lang.Keyword 3)"), "false");
    assert_eq!(run("(instance? String \"hi\")"), "true");
    assert_eq!(run("(instance? Long 42)"), "true");
    assert_eq!(run("(instance? clojure.lang.IPersistentVector [1 2])"), "true");
    assert_eq!(run("(instance? clojure.lang.IPersistentMap {:a 1})"), "true");
}

#[test]
#[should_panic(expected = "throw")]
fn throw_aborts() {
    run("(throw \"boom\")");
}

#[test]
fn real_core_clj_setmacro() {
    // Real core.clj registers a macro via `(. (var name) (setMacro))` after a
    // plain `(def name (fn* name [&form &env ...] ...))`, rather than metadata.
    let src = r#"
        (def my-when
          (fn* my-when [&form &env test & body]
            (cons 'if (cons test (cons (cons 'do body) nil)))))
        (. (var my-when) (setMacro))
        (my-when true 1 2 99)
    "#;
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    assert_eq!(clojure_stub::clj_str(&rt, r), "99");
}

#[test]
fn real_style_defn_macro() {
    // A `defn` defined the way core.clj does: a macro that destructures fdecl
    // (optional docstring), then emits `(def name (fn name params body...))`.
    // Proves the var+setMacro path plus seq/first/next/cons/syntax-quote suffice
    // to host a real-shaped defn without any special toolkit support.
    let src = r#"
        (def my-defn
          (fn* my-defn [&form &env name & fdecl]
            (let [fdecl (if (instance? String (first fdecl)) (next fdecl) fdecl)
                  params (first fdecl)
                  body (next fdecl)]
              (cons 'def
                (cons name
                  (list (cons `fn (cons name (cons params body)))))))))
        (. (var my-defn) (setMacro))
        (my-defn square "doc" [x] (* x x))
        (square 9)
    "#;
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    assert_eq!(clojure_stub::clj_str(&rt, r), "81");
}

#[test]
fn literal_core_defn() {
    // A near-verbatim clojure.core `defn`: docstring + attr-map handling, sigs
    // for :arglists, with-meta on the name symbol, and `(cons `fn fdecl)`. Loaded
    // via the real var+setMacro registration. Then used to define + call fns.
    let src = r#"
        (def defn
          (fn* defn [&form &env name & fdecl]
            (let [m (if (instance? String (first fdecl)) {:doc (first fdecl)} {})
                  fdecl (if (instance? String (first fdecl)) (next fdecl) fdecl)
                  m (if (map? (first fdecl)) (conj m (first fdecl)) m)
                  fdecl (if (map? (first fdecl)) (next fdecl) fdecl)
                  fdecl (if (vector? (first fdecl)) (list fdecl) fdecl)
                  m (conj {:arglists (list 'quote (sigs fdecl))} m)
                  m (conj (if (meta name) (meta name) {}) m)]
              (list 'def (with-meta name m)
                    (cons `fn fdecl)))))
        (. (var defn) (setMacro))
        (defn cube "the cube" [x] (* x x x))
        (defn add2 ([x] (add2 x 1)) ([x y] (+ x y)))
        (list (cube 3) (add2 10) (add2 10 5))
    "#;
    let mut rt = Runtime::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
    assert_eq!(clojure_stub::clj_str(&rt, r), "(27 11 15)");
}

#[test]
fn variadic_arithmetic() {
    assert_eq!(run("(+ 1 2 3 4)"), "10");
    assert_eq!(run("(+)"), "0");
    assert_eq!(run("(* 2 3 4)"), "24");
    assert_eq!(run("(*)"), "1");
    assert_eq!(run("(- 10 1 2 3)"), "4");
    assert_eq!(run("(- 5)"), "-5");
    assert_eq!(run("(< 1 2 3)"), "true");
    assert_eq!(run("(< 1 2 2)"), "false");
    assert_eq!(run("(<= 1 2 2)"), "true");
    assert_eq!(run("(>= 3 3 2)"), "true");
    assert_eq!(run("(= 1 1 1)"), "true");
    assert_eq!(run("(not= 1 1 2)"), "true");
    assert_eq!(run("(reduce + 0 [1 2 3 4 5])"), "15");
}

#[test]
fn destructuring_as_or() {
    assert_eq!(run("(let [[a b :as v] [1 2]] [a b v])"), "[1 2 [1 2]]");
    assert_eq!(run("(let [{:keys [x y] :as m} {:x 1 :y 2}] [x y m])"), "[1 2 {:x 1, :y 2}]");
    assert_eq!(run("(let [{:keys [x y] :or {y 9}} {:x 1}] [x y])"), "[1 9]");
    assert_eq!(run("(let [{a :aa :or {a 0}} {}] a)"), "0");
    assert_eq!(run("((fn [[a & r]] [a r]) [1 2 3])"), "[1 (2 3)]");
}

#[test]
fn control_macros() {
    assert_eq!(run("(if-not false :yes :no)"), ":yes");
    assert_eq!(run("(if-not true :yes)"), "nil");
    assert_eq!(run("(let [a (atom 0)] (dotimes [i 5] (swap! a + i)) @a)"), "10");
    assert_eq!(run("(let [a (atom 0)] (doseq [x [1 2 3]] (swap! a + x)) @a)"), "6");
    assert_eq!(run("(case 2 1 :one 2 :two :other)"), ":two");
    assert_eq!(run("(case 9 1 :one 2 :two :other)"), ":other");
    assert_eq!(run("(when-first [x [10 20]] x)"), "10");
    assert_eq!(run("(let [a (atom 0) i (atom 0)] (while (< @i 3) (swap! a + 1) (swap! i + 1)) @a)"), "3");
}
