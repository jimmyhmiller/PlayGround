;; The microclj performance suite — ONE file, run byte-identically by BOTH
;; microclj and real JVM Clojure:
;;
;;   ./target/release/microclj --jit clojure-stub/bench/suite.clj
;;   cd /tmp && clojure -M .../clojure-stub/bench/suite.clj
;;
;; and joined into a comparison table by `bench/compare.clj` (see bench.sh).
;;
;; Why one file, and why it looks like this — every rule here exists because
;; the ad-hoc scratchpad benchmarks it replaces got the answer WRONG:
;;
;;  * ONE SOURCE OF TRUTH. The predecessor kept the workloads in a bash assoc
;;    array and re-typed them into a .clj oracle. Two copies drift silently and
;;    you compare different programs. Here both runtimes read these very bytes.
;;
;;  * EVERY WORKLOAD RETURNS A CHECKSUM, and the two runtimes must agree on it.
;;    A recorded predecessor run had microclj "winning" vecbuild 0.16ms vs
;;    27ms — a 170x win that simply meant the work never ran. A workload whose
;;    checksum disagrees (or is suspiciously trivial) is not a fast benchmark,
;;    it is a broken one, and compare.clj fails the run rather than tabling it.
;;
;;  * TIME-BUDGETED WARMUP, not a fixed 2 calls. A cold JVM measured here ran
;;    vecbuild at 31ms and a warm one at 10.7ms; "best of 4 from cold" reports
;;    ~25ms and flatters us by 2.5x. Both tiers JIT, so both need the budget.
;;
;;  * MEDIAN OF MANY SAMPLES + AN EXPLICIT SPREAD. A single timed call cannot
;;    tell a real regression from the scheduler. If spread is wide, the number
;;    is not evidence, and the table says so instead of hiding it.
;;
;; Timing is in-process (System/nanoTime). It cannot be wall-clock of the
;; process: microclj spends ~340ms loading its prelude, which would swamp
;; every workload here.

;; ─────────────────────────── knobs ───────────────────────────
;; Overridable per-run so a quick check and a publishable measurement are the
;; same code path with different budgets (see bench.sh --quick).
(def warmup-ms 3000)   ;; per workload, before ANY sample is kept
(def min-warmup-iters 5)
;; 25, not 15: the allocation-heavy workloads take a GC pause every few
;; samples, and a median over too few samples wobbles with where the pauses
;; happen to land. The pauses are real cost and are NOT filtered out — they are
;; why `spread` exists.
(def samples 25)       ;; kept samples per workload
(def batch-target-ms 50) ;; a sample batches iterations up to ~this long

;; ─────────────────────────── stats ───────────────────────────
;; Deliberately sqrt-free: this dialect's java.lang.Math has no sqrt, and a
;; standard deviation would be the wrong tool anyway — benchmark samples are
;; right-skewed (interference only ever ADDS time), so a symmetric dispersion
;; measure overstates noise. Median/min/p95 describe the skew directly.
(defn- pctl [sorted p]
  (let [n (count sorted)
        i (long (* (/ p 100.0) (dec n)))]
    (nth sorted (if (< i 0) 0 (if (>= i n) (dec n) i)))))

(defn- median [sorted]
  (let [n (count sorted)
        h (quot n 2)]
    (if (odd? n)
      (nth sorted h)
      (/ (+ (nth sorted (dec h)) (nth sorted h)) 2.0))))

;; ─────────────────────────── the harness ───────────────────────────
;; Returns [checksum median-ms min-ms p95-ms spread-pct n batch].
;; `spread-pct` = (p95-min)/median — the honest "how much should you trust
;; this" number. compare.clj flags anything above 15%.
(defn- time-once
  "Nanos to run f `batch` times. Returns [nanos last-result]."
  [f batch]
  (let [t0 (System/nanoTime)
        r (loop [i 0 last nil]
            (if (< i batch) (recur (inc i) (f)) last))
        t1 (System/nanoTime)]
    [(- t1 t0) r]))

(defn- warm
  "Run f until the warmup budget is spent AND min-warmup-iters have run.
   Returns the per-iteration nanos estimate from the LAST (warmest) call, which
   is what the batch size should be calibrated against."
  [f]
  (let [budget (* warmup-ms 1000000)
        t0 (System/nanoTime)]
    (loop [iters 0 est nil]
      (if (and (>= iters min-warmup-iters)
               (>= (- (System/nanoTime) t0) budget))
        est
        (let [[ns _] (time-once f 1)]
          (recur (inc iters) ns))))))

(defn bench
  "Warm up f, then collect `samples` batched samples. f must return a checksum."
  [f]
  (let [est (warm f)
        ;; Batch short workloads so a sample dwarfs timer granularity; long
        ;; ones stay at batch 1 rather than running for minutes.
        batch (let [b (quot (* batch-target-ms 1000000) (if (< est 1) 1 est))]
                (if (< b 1) 1 b))
        results (loop [i 0 acc [] chk nil]
                  (if (>= i samples)
                    [acc chk]
                    (let [[ns r] (time-once f batch)]
                      (recur (inc i)
                             (conj acc (/ (/ ns (double batch)) 1000000.0))
                             r))))
        ms (sort (first results))
        checksum (second results)]
    [checksum (median ms) (first ms) (pctl ms 95)
     (* 100.0 (/ (- (pctl ms 95) (first ms)) (median ms)))
     samples batch]))

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

;; ─────────────────────────── main ───────────────────────────
;; TSV on stdout, one RESULT line per workload, so compare.clj can join the two
;; runs by name. Progress goes to stdout too but is prefixed, not RESULT.
(defn -run []
  (println (str "# warmup-ms=" warmup-ms " samples=" samples
                " batch-target-ms=" batch-target-ms))
  (println "# name\tchecksum\tmedian_ms\tmin_ms\tp95_ms\tspread_pct\tn\tbatch")
  (loop [ws workloads]
    (when (seq ws)
      (let [[nm f] (first ws)
            [checksum med mn p95 spread n batch] (bench f)]
        (println (str "RESULT\t" nm "\t" checksum
                      "\t" (format "%.4f" med)
                      "\t" (format "%.4f" mn)
                      "\t" (format "%.4f" p95)
                      "\t" (format "%.1f" spread)
                      "\t" n "\t" batch)))
      (recur (rest ws)))))

(-run)
