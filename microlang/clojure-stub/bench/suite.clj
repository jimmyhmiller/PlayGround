;; The microclj performance suite — ONE file, run byte-identically by BOTH
;; microclj and real JVM Clojure:
;;
;;   MICROLANG_PATH=clojure-stub/bench \
;;     ./target/release/microclj --jit clojure-stub/bench/suite.clj
;;   cd clojure-stub/bench && clojure -M suite.clj
;;
;; and joined into a comparison table by `bench/compare.clj` (see bench.sh).
;;
;; Why one file: the predecessor kept the workloads in a bash assoc array and
;; re-typed them into a .clj oracle. Two copies drift silently and you compare
;; different programs. Here both runtimes read these very bytes.
;;
;; The measurement discipline (checksums, time-budgeted warmup, median+spread)
;; lives in harness.clj — shared by every bench file, with the full rationale.
;;
;; NOTE: these are op-level DIAGNOSTIC microbenchmarks. The project scoreboard
;; is the real-library corpus (bench/corpus/) — real programs on core.match,
;; data.json, core.async, test.check. Optimize to those, use this to localize.

(ns bench-suite
  (:require [harness :as h]))

;; ─────────────────────────── workloads ───────────────────────────
;; Each returns a checksum that DEPENDS on the whole computation, so a runtime
;; cannot elide the work and still answer correctly. Sizes are picked so a
;; single iteration lands in the milliseconds on both tiers.
;;
;; `range`/`vec` construction stays INSIDE the timed region: it is part of the
;; workload, and it is the same work on both sides.

(defn add3 [a b c] (+ a b c))

(def workloads
  [["raw-loop"
    (fn [] (loop [i 0 acc 0] (if (< i 5000000) (recur (inc i) (+ acc i)) acc)))]

   ["defn-call"
    (fn [] (loop [i 0 acc 0] (if (< i 2000000) (recur (inc i) (add3 acc i 1)) acc)))]

   ["closure-call"
    (fn [] (let [n 3
                 f (fn [x] (+ x n))]
             (loop [i 0 acc 0] (if (< i 2000000) (recur (inc i) (f acc)) acc))))]

   ["reduce-map"
    (fn [] (reduce + 0 (map inc (range 1000000))))]

   ["transduce"
    (fn [] (transduce (comp (map inc) (filter even?)) + 0 (range 500000)))]

   ["into-xform"
    (fn [] (let [v (into [] (comp (map inc) (filter even?)) (range 500000))]
             (+ (count v) (nth v 0) (nth v (dec (count v))))))]

   ["comp-chain"
    (fn [] (reduce + 0 (map (comp inc inc inc) (range 500000))))]

   ["vecbuild"
    (fn [] (let [v (loop [i 0 v []] (if (< i 1000000) (recur (inc i) (conj v i)) v))]
             (+ (count v) (nth v 0) (nth v 999999))))]

   ["group-by"
    (fn [] (let [g (group-by odd? (range 100000))]
             (+ (count (get g true)) (count (get g false)))))]

   ["assoc-build"
    (fn [] (let [m (loop [i 0 m {}] (if (< i 100000) (recur (inc i) (assoc m i i)) m))]
             (+ (count m) (get m 0) (get m 99999))))]

   ["apply"
    (fn [] (apply + (range 200000)))]

   ;; `count` forces the whole lazy seq — that IS the work, and a wrong count
   ;; is exactly how an unforced-seq bug shows up. `first` is O(1); an `nth`
   ;; probe here would re-walk 600k cells and measure the checksum, not the
   ;; workload.
   ["interleave"
    (fn [] (let [s (interleave (range 300000) (range 300000))]
             (+ (count s) (first s))))]])

(h/run-workloads workloads)
