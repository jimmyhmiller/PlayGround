;; floor.clj — the FLOOR DECOMPOSITION: isolate the per-operation costs that
;; bound every real-library workload, one file run byte-identically by both
;; runtimes (harness discipline: checksums, budgeted warmup, median+spread).
;;
;;   MICROLANG_PATH=clojure-stub/bench \
;;     ./target/release/microclj --jit clojure-stub/bench/floor.clj
;;   cd clojure-stub/bench && clojure -M floor.clj
;;
;; The corpus (gauntlet) says real code is 20-70x while arithmetic loops are
;; ~5x. The hypothesis these rows test: the gap is the baseline cost of an
;; ORDINARY FUNCTION CALL and of a SMALL ALLOCATION — the two things the JVM
;; makes free (C2 inlines everything; escape analysis deletes the allocs) and
;; call-dense library code (parsers, generators) does per element. Each row
;; isolates one cost with everything else held trivial.

(ns bench-floor
  (:require [harness :as h]))

;; ── calls: non-tail, statically known, C2-inlinable ───────────
;; 7 calls per f3 invocation, all trivially inlinable by C2. If microclj's
;; ratio here is ~Nx, then call-dense code has an Nx floor REGARDLESS of
;; dispatch/arith codegen quality.
(defn f1 [x] (+ x 1))
(defn f2 [x] (f1 (f1 x)))
(defn f3 [x] (f2 (f2 x)))

;; ── calls through a fn VALUE (higher-order, still monomorphic) ─
(defn apply-twice [g x] (g (g x)))

;; ── allocation: a small short-lived vector per element ─────────
;; C2's escape analysis scalar-replaces this; a real heap allocator pays per
;; element. The read-back forces the value to exist.
(defn vec3 [i] [i (+ i 1) (+ i 2)])

;; ── allocation: a small map per element (the JSON-shaped cost) ─
(defn map2 [i] {:a i :b (+ i 1)})

(h/run-workloads
 [["floor-call-chain"
   ;; ~7 non-tail calls + 7 adds per iteration
   (fn [] (loop [i 0 acc 0]
            (if (< i 300000)
              (recur (inc i) (+ acc (f3 i)))
              acc)))]

  ["floor-call-value"
   ;; 2 calls through a fn value + 1 outer call per iteration
   (fn [] (loop [i 0 acc 0]
            (if (< i 300000)
              (recur (inc i) (+ acc (apply-twice f1 i)))
              acc)))]

  ["floor-alloc-vec"
   ;; one 3-vector allocated + read per iteration
   (fn [] (loop [i 0 acc 0]
            (if (< i 300000)
              (let [v (vec3 i)]
                (recur (inc i) (+ acc (nth v 0) (nth v 2))))
              acc)))]

  ["floor-alloc-map"
   ;; one 2-key map allocated + read per iteration
   (fn [] (loop [i 0 acc 0]
            (if (< i 200000)
              (let [m (map2 i)]
                (recur (inc i) (+ acc (:a m) (:b m))))
              acc)))]

  ["floor-arith-control"
   ;; the control: pure arithmetic, no calls, no allocation
   (fn [] (loop [i 0 acc 0]
            (if (< i 2000000)
              (recur (inc i) (+ acc i))
              acc)))]])
