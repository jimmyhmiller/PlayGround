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
#[should_panic]
fn throw_aborts() {
    // An uncaught throw unwinds the whole computation.
    run("(throw \"boom\")");
}

#[test]
fn try_catch_finally() {
    // catch binds the thrown value; the try value is the handler's value.
    assert_eq!(run("(try (throw \"oops\") (catch Exception e e))"), "\"oops\"");
    // no throw -> body value; catch is skipped.
    assert_eq!(run("(try 42 (catch Exception e :never))"), "42");
    // finally runs for effect on the normal path but doesn't change the value.
    assert_eq!(run("(let [a (atom 0)] (try 7 (finally (reset! a 1))) @a)"), "1");
    assert_eq!(run("(try 7 (finally 999))"), "7");
    // finally runs on the throw path too, then the catch value is returned.
    assert_eq!(
        run("(let [a (atom :x)] [(try (throw :bang) (catch Exception e :caught) (finally (reset! a :ran))) @a])"),
        "[:caught :ran]"
    );
    // a thrown record value round-trips through catch.
    assert_eq!(run("(try (throw {:err 1}) (catch Exception e (:err e)))"), "1");
    // nested: inner catch handles, outer body continues.
    assert_eq!(run("(try (+ 1 (try (throw :z) (catch Exception e 9))) (catch Exception e :outer))"), "10");
    // re-throw from a catch propagates to the outer handler.
    assert_eq!(run("(try (try (throw :a) (catch Exception e (throw :b))) (catch Exception e e))"), ":b");
}

#[test]
fn typed_catch() {
    // A specific class matches only the corresponding runtime tag; the first
    // matching clause wins (ClojureScript's instanceof-chain model).
    assert_eq!(
        run("(try (throw \"s\") (catch clojure.lang.Keyword e :kw) (catch String e :str))"),
        ":str"
    );
    assert_eq!(
        run("(try (throw :k) (catch clojure.lang.Keyword e :kw) (catch String e :str))"),
        ":kw"
    );
    // :default is the explicit catch-all when no typed clause matches.
    assert_eq!(
        run("(try (throw 42) (catch String e :str) (catch :default e [:default e]))"),
        "[:default 42]"
    );
    // A constructor exception carries its class name as a tag, so a typed catch
    // on that class matches it — and a different class does not.
    assert_eq!(
        run("(try (throw (RuntimeException. \"boom\")) (catch RuntimeException e :caught))"),
        ":caught"
    );
    assert_eq!(
        run("(try (throw (RuntimeException. \"boom\")) (catch clojure.lang.Keyword e :kw) (catch :default e :fell-through))"),
        ":fell-through"
    );
    // No clause matches and there is no catch-all -> the throw propagates out.
    assert_eq!(
        run("(try (try (throw :a) (catch String e :str)) (catch :default e :outer))"),
        ":outer"
    );
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

#[test]
fn lazy_sequences() {
    // Infinite range, consumed lazily.
    assert_eq!(run("(take 5 (range))"), "(0 1 2 3 4)");
    assert_eq!(run("(take 3 (map inc (range)))"), "(1 2 3)");
    assert_eq!(run("(take 4 (filter even? (range)))"), "(0 2 4 6)");
    // iterate / repeat / cycle / repeatedly.
    assert_eq!(run("(take 5 (iterate (fn [x] (* x 2)) 1))"), "(1 2 4 8 16)");
    assert_eq!(run("(take 3 (repeat :x))"), "(:x :x :x)");
    assert_eq!(run("(repeat 3 :y)"), "(:y :y :y)");
    assert_eq!(run("(take 7 (cycle [1 2 3]))"), "(1 2 3 1 2 3 1)");
    // take-while / drop-while on an infinite seq.
    assert_eq!(run("(take-while (fn [x] (< x 5)) (range))"), "(0 1 2 3 4)");
    assert_eq!(run("(take 3 (drop-while (fn [x] (< x 10)) (range)))"), "(10 11 12)");
    // lazy-seq is only realized on demand: a self-referential fib stream.
    assert_eq!(
        run("(defn nats [n] (lazy-seq (cons n (nats (inc n))))) (take 5 (nats 0))"),
        "(0 1 2 3 4)"
    );
    // range with start/end/step.
    assert_eq!(run("(range 2 8)"), "(2 3 4 5 6 7)");
    assert_eq!(run("(range 0 10 2)"), "(0 2 4 6 8)");
    // composition over an infinite source, forced by take.
    assert_eq!(run("(reduce + 0 (take 5 (map (fn [x] (* x x)) (range))))"), "30");
    // nth into an infinite seq.
    assert_eq!(run("(nth (map inc (range)) 100)"), "101");
    // keep drops nils.
    assert_eq!(run("(keep (fn [x] (if (even? x) x nil)) (range 6))"), "(0 2 4)");
}

#[test]
fn real_os_threads() {
    // A future runs its thunk on a real std::thread sharing the heap; deref joins.
    assert_eq!(run("(deref (future (+ 1 2)))"), "3");
    assert_eq!(run("@(future (* 6 7))"), "42");
    // The worker calls GLOBALS defined on the main thread (shared env).
    assert_eq!(
        run("(defn work [n] (reduce + 0 (range n))) (deref (future (work 100)))"),
        "4950"
    );
    // Two futures run concurrently over the shared heap, then combine.
    assert_eq!(
        run("(defn work [n] (reduce + 0 (range n))) (let [a (future (work 100)) b (future (work 200))] (+ @a @b))"),
        "24850"
    );
    // A worker allocates lots of heap (lists) concurrently with the main thread.
    assert_eq!(
        run("(let [f (future (count (map inc (range 500))))] (+ @f (count (range 300))))"),
        "800"
    );
    // Nested futures: a worker spawns its own worker.
    assert_eq!(run("(deref (future (deref (future (+ 10 20)))))"), "30");
}

#[test]
fn thread_stress() {
    // Spawn many workers that each allocate + compute over the shared heap in
    // parallel; the total must be exact every run (no lost/torn allocations).
    let src = r#"
        (defn work [n] (reduce + 0 (map (fn [x] (* x x)) (range n))))
        (defn spawn-n [k]
          (if (%num-eq k 0) nil (%cons (%spawn (fn [] (work 60))) (spawn-n (%sub k 1)))))
        (defn await-all [fs] (if (nil? fs) 0 (%add (%await (%first fs)) (await-all (%rest fs)))))
        (await-all (spawn-n 16))
    "#;
    // work(60) = sum of squares 0..59 = 70210; times 16 workers = 1123360.
    assert_eq!(run(src), "1123360");
}

#[test]
#[ignore = "concurrent explicit (gc) has a residual STW-rendezvous race; \
            single-threaded gc and parallel non-gc threads are solid. Needs a \
            race detector (TSan/Helgrind) on the Linux host to pinpoint."]
fn concurrent_gc() {
    // Workers allocate heavily AND force moving collections while sibling threads
    // run concurrently. The safepoint STW protocol must stop every mutator before
    // any object moves; results must stay exact and no thread may crash.
    let src = r#"
        (defn churn [n]
          (loop [i 0 acc 0]
            (if (%num-eq i n)
                acc
                (let [xs (map (fn [x] (* x x)) (range 40))]
                  (do (gc)
                      (recur (%add i 1) (%add acc (reduce + 0 xs))))))))
        (defn spawn-n [k]
          (if (%num-eq k 0) nil (%cons (%spawn (fn [] (churn 24))) (spawn-n (%sub k 1)))))
        (defn await-all [fs] (if (nil? fs) 0 (%add (%await (%first fs)) (await-all (%rest fs)))))
        (await-all (spawn-n 8))
    "#;
    // sum of squares 0..39 = 20540; churn(24) = 24*20540 = 492960; x8 = 3943680.
    assert_eq!(run(src), "3943680");
}

#[test]
fn atom_cas_concurrent() {
    // Many worker threads each swap! the SAME shared atom many times. With a real
    // compare-and-set retry loop, not one increment may be lost: the total is exact.
    let src = r#"
        (def counter (atom 0))
        (defn bump-n [k] (if (%num-eq k 0) nil (do (swap! counter (fn [x] (%add x 1))) (bump-n (%sub k 1)))))
        (defn spawn-n [k] (if (%num-eq k 0) nil (%cons (%spawn (fn [] (bump-n 250))) (spawn-n (%sub k 1)))))
        (defn await-all [fs] (if (nil? fs) nil (do (%await (%first fs)) (await-all (%rest fs)))))
        (await-all (spawn-n 8))
        (deref counter)
    "#;
    // 8 threads * 250 increments = 2000, exact iff no lost updates.
    assert_eq!(run(src), "2000");
}
