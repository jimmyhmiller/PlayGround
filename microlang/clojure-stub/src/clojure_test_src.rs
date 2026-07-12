//! `clojure.test` — a pragmatic subset (deftest / is / are / testing / thrown? /
//! run-tests), written in IDIOMATIC Clojure (syntax-quote, not hand-built lists).
//! This is a real test of the fork: the macros rely on syntax-quote auto-qualifying
//! their helper references to `clojure.test/…` so they resolve at the caller's site.
//! Enough to run real library test suites (e.g. clojure.math.combinatorics's own).

pub const CLOJURE_TEST: &str = r##"
(ns clojure.test)

;; A registry of every `deftest` as [name thunk] pairs, plus per-run counters.
(def -tests (atom []))
(def ^:dynamic *report-counters* nil)
(def ^:dynamic *testing-context* nil)

(defn -inc! [k]
  (when *report-counters*
    (swap! *report-counters* (fn [m] (assoc m k (inc (get m k 0)))))))

(defn -ctx [] (if *testing-context* (str " (" *testing-context* ")") ""))

(defn -report-fail [form val]
  (-inc! :fail)
  (println (str "FAIL" (-ctx)))
  (println "expected:" (pr-str form))
  (println "  actual:" (pr-str val)))

(defn -report-error [form e]
  (-inc! :error)
  (println (str "ERROR" (-ctx)) "in" (pr-str form) "->" (str e)))

;; Evaluate a thunked assertion: truthy = pass, falsey = fail (show form + value),
;; a thrown exception = error.
(defn -run-is [thunk form]
  (let [r (try {:v (thunk)} (catch Throwable e {:e e}))]
    (cond
      (contains? r :e) (-report-error form (:e r))
      (:v r) (-inc! :pass)
      :else (-report-fail form (:v r)))))

(defmacro is [form & _msg]
  `(-run-is (fn [] ~form) '~form))

;; `(are [x y] (= x y) a b c d …)` -> an `is` per row, binding argv to the row.
(defmacro are [argv expr & data]
  `(do ~@(map (fn [row] `(is (let [~@(interleave argv row)] ~expr)))
              (partition (count argv) data))))

(defmacro testing [label & body]
  `(binding [*testing-context* ~label] ~@body))

;; `deftest` defines a 0-arg test fn AND registers it so `run-tests` finds it.
(defmacro deftest [name & body]
  `(do (def ~name (fn [] ~@body))
       (swap! -tests conj [(quote ~name) ~name])))

;; `(thrown? Class body…)` — true iff evaluating body throws (the class is ignored;
;; this dialect has no Java class hierarchy). Used inside `is`.
(defmacro thrown? [_cls & body]
  `(try ~@body false (catch Throwable e# true)))

;; Run every registered test under fresh counters; print a summary; return it.
(defn run-tests [& _nss]
  (binding [*report-counters* (atom {:test 0 :pass 0 :fail 0 :error 0})]
    (doseq [t (deref -tests)]
      (-inc! :test)
      (try ((second t))
           (catch Throwable e (-report-error (first t) e))))
    (let [c (deref *report-counters*)]
      (println)
      (println "Ran" (:test c) "tests containing"
               (+ (:pass c) (:fail c) (:error c)) "assertions.")
      (println (:fail c) "failures," (:error c) "errors.")
      c)))

(defn successful? [c] (and (zero? (:fail c 0)) (zero? (:error c 0))))
"##;
