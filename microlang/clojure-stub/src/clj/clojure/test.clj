;; `clojure.test` — a pragmatic subset (deftest / is / are / testing / thrown? /
;; run-tests), written in IDIOMATIC Clojure (syntax-quote, not hand-built lists).
;; This is a real test of the fork: the macros rely on syntax-quote auto-qualifying
;; their helper references to `clojure.test/…` so they resolve at the caller's site.
;; Enough to run real library test suites (e.g. clojure.math.combinatorics's own).
;;
;; `is` goes through the REAL extension points — `assert-expr` (a multimethod
;; consulted at MACROEXPANSION time) and `report`/`do-report` (a multimethod on
;; :type) — because real libraries extend exactly those: test.check's
;; clojure-test integration adds an `assert-expr` method for its `check?` form
;; and `report` methods for its trial/shrink progress types.

(ns clojure.test)

;; A registry of every `deftest` as [name thunk] pairs, plus per-run counters.
(def -tests (atom []))
(def ^:dynamic *report-counters* nil)
(def ^:dynamic *testing-context* nil)
(def ^:dynamic *testing-vars* (list))

(defn -inc! [k]
  (when *report-counters*
    (swap! *report-counters* (fn [m] (assoc m k (inc (get m k 0)))))))

(defn -ctx [] (if *testing-context* (str " (" *testing-context* ")") ""))

(defn testing-vars-str [_m]
  (if (seq *testing-vars*) (str (first *testing-vars*)) ""))

;; Real clojure.test rebinds *test-out*; this printer has one output, so the
;; macro is just its body — the point is that library code CALLS it.
(defmacro with-test-out [& body] `(do ~@body))

;; ─────────────── reporting ───────────────
;; `report` is a MULTIMETHOD on :type, as in real clojure.test — libraries
;; check `(instance? clojure.lang.MultiFn report)` and add methods for their
;; own progress types (test.check's ::trial/::shrunk), so a plain fn here
;; disables them.
(defmulti report :type)
(defmethod report :default [_m] nil)
(defmethod report :pass [_m] (-inc! :pass))
(defmethod report :fail [m]
  (-inc! :fail)
  (println (str "FAIL" (-ctx) (when (:message m) (str " " (:message m)))))
  (println "expected:" (pr-str (:expected m)))
  (println "  actual:" (pr-str (:actual m))))
(defmethod report :error [m]
  (-inc! :error)
  (println (str "ERROR" (-ctx)) "in" (pr-str (:expected m)) "->" (str (:actual m))))

(defn do-report [m] (report m))

;; ─────────────── assertions ───────────────
;; `assert-expr` runs at MACROEXPANSION time: `is` asks it for the code of the
;; assertion. Dispatch is the form's head symbol; libraries add methods for
;; their own assertion forms (`(is (check? …))`, `(is (thrown? …))`).
(defmulti assert-expr (fn [_msg form] (if (seq? form) (first form) :default)))

(defmethod assert-expr :default [msg form]
  `(let [v# ~form]
     (if v#
       (do-report {:type :pass :message ~msg :expected '~form :actual v#})
       (do-report {:type :fail :message ~msg :expected '~form :actual v#}))
     v#))

;; `(is (thrown? Class body…))` passes iff evaluating body throws. The class is
;; ignored (this dialect has no Java class hierarchy).
(defmethod assert-expr 'thrown? [msg form]
  `(let [threw# (try ~@(rest (rest form)) false (catch Throwable e# true))]
     (if threw#
       (do-report {:type :pass :message ~msg :expected '~form :actual :threw})
       (do-report {:type :fail :message ~msg :expected '~form :actual :did-not-throw}))
     threw#))

(defmacro try-expr [msg form]
  `(try ~(assert-expr msg form)
        (catch Throwable e#
          (do-report {:type :error :message ~msg :expected '~form :actual e#}))))

(defmacro is
  ([form] `(is ~form nil))
  ([form msg] `(try-expr ~msg ~form)))

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
           (catch Throwable e
             (do-report {:type :error :expected (first t) :actual e}))))
    (let [c (deref *report-counters*)]
      (println)
      (println "Ran" (:test c) "tests containing"
               (+ (:pass c) (:fail c) (:error c)) "assertions.")
      (println (:fail c) "failures," (:error c) "errors.")
      c)))

(defn successful? [c] (and (zero? (:fail c 0)) (zero? (:error c 0))))
