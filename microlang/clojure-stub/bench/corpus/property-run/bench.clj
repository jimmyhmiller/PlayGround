;; property-run — REAL test.check: seeded quick-check runs (pass + fail +
;; shrink) and seeded generation of nested data. Generator rose-trees are
;; lazy-seq + protocol heavy — exactly the "unprofiled shape" a new library
;; brings. Seeded, so both runtimes traverse IDENTICAL rose trees (the
;; splittable RNG is bit-for-bit vs the JVM).
;;
;;   MICROLANG_PATH=clojure-stub/bench:clojure-stub/vendor/test.check \
;;     ./target/release/microclj --jit clojure-stub/bench/corpus/property-run/bench.clj
;;   cd clojure-stub/bench && clojure -M:test.check corpus/property-run/bench.clj

(ns bench-property-run
  (:require [harness :as h]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.random :as random]
            [clojure.test.check.rose-tree :as rose]))

;; ── seeded generation (no quick-check loop): forces generator + rose paths ──
(def gen-nested
  (gen/vector (gen/map gen/keyword (gen/one-of [gen/small-integer
                                                (gen/vector gen/nat 0 4)]))
              1 6))

(defn- sample-at
  "The root value of `g` grown at `size` from a SEEDED rng — deterministic on
  both runtimes (gen internals are API here; same pinned version both sides)."
  [g seed size]
  (rose/root (gen/call-gen g (random/make-java-util-splittable-random seed) size)))

(defn- fold-val
  "Content-forcing numeric fold over generated nested data."
  [v]
  (cond
    (map? v) (reduce-kv (fn [a k x] (+ a (count (name k)) (fold-val x))) 1 v)
    (vector? v) (reduce + 1 (map fold-val v))
    (number? v) (rem v 1009)
    :else 0))

;; ── properties ────────────────────────────────────────────────
(def prop-sort-idempotent
  (prop/for-all [v (gen/vector gen/small-integer)]
    (= (sort v) (sort (sort v)))))

(def prop-small-sum
  ;; FAILS and shrinks: the shrink search is the workload
  (prop/for-all [v (gen/vector gen/nat)]
    (< (reduce + 0 v) 120)))

;; ── workloads ─────────────────────────────────────────────────
(h/run-workloads
 [["pr-generate"
   (fn [] (loop [seed 0 acc 0]
            (if (< seed 30)
              (recur (inc seed)
                     (+ acc (fold-val (sample-at gen-nested seed (rem seed 25)))))
              acc)))]

  ["pr-quickcheck"
   ;; a passing 60-test run; num-tests depends on the run really happening
   (fn [] (let [r (tc/quick-check 60 prop-sort-idempotent :seed 42)]
            (+ (:num-tests r) (if (true? (:result r)) 1 0))))]

  ["pr-shrink"
   ;; a failing run: find + SHRINK the counterexample; checksum folds the
   ;; shrunk value and the search effort, both seed-deterministic
   (fn [] (let [r (tc/quick-check 200 prop-small-sum :seed 7)
                sm (get-in r [:shrunk :smallest])
                v (first sm)]
            (+ (reduce + 0 v)
               (count v)
               (get-in r [:shrunk :depth])
               (:num-tests r))))]])
