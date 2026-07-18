;; The shared benchmark harness — ONE implementation, required by every bench
;; file (suite.clj, json.clj, match.clj, corpus/*) and run byte-identically by
;; BOTH microclj and JVM Clojure:
;;
;;   microclj:  MICROLANG_PATH=<this dir>:...   (require resolves harness.clj)
;;   JVM:       this dir on the classpath (deps.edn here has :paths ["."])
;;
;; It used to be copy-pasted into each bench file; three copies of a harness
;; drift silently, which is the same "two copies compare different programs"
;; failure the ONE-file rule exists to prevent — one level up.
;;
;; Why the harness looks like this — every rule exists because the ad-hoc
;; benchmarks it replaced got the answer WRONG:
;;
;;  * EVERY WORKLOAD RETURNS A CHECKSUM, and the two runtimes must agree on it.
;;    A recorded predecessor run had microclj "winning" vecbuild 0.16ms vs
;;    27ms — a 170x win that simply meant the work never ran. A workload whose
;;    checksum disagrees is not a fast benchmark, it is a broken one, and
;;    compare.clj fails the run rather than tabling it.
;;
;;  * TIME-BUDGETED WARMUP, not a fixed 2 calls. A cold JVM measured vecbuild
;;    at 31ms and a warm one at 10.7ms; "best of 4 from cold" flatters us by
;;    2.5x. Both tiers JIT, so both need the budget.
;;
;;  * MEDIAN OF MANY SAMPLES + AN EXPLICIT SPREAD. A single timed call cannot
;;    tell a real regression from the scheduler. If spread is wide, the number
;;    is not evidence, and the table says so instead of hiding it.
;;
;; Timing is in-process (System/nanoTime). It cannot be wall-clock of the
;; process: microclj spends ~340ms loading its prelude, which would swamp
;; every workload.

(ns harness)

;; ─────────────────────────── knobs ───────────────────────────
;; Overridable per-run so a quick check and a publishable measurement are the
;; same code path with different budgets (bench.sh --quick seds a COPY of this
;; file and verifies the sed took — a harness that lies about its own settings
;; is worse than none).
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
  "Warm up f, then collect `samples` batched samples. f must return a checksum.
   Returns [checksum median-ms min-ms p95-ms spread-pct n batch]. `spread-pct`
   = (p95-min)/median — the honest \"how much should you trust this\" number.
   compare.clj flags anything above 15%."
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

;; ─────────────────────────── output ───────────────────────────
;; TSV on stdout, one RESULT line per workload, joined by compare.clj across
;; the two runtimes by name. Progress/context lines are `#`-prefixed.
(defn print-header []
  (println (str "# warmup-ms=" warmup-ms " samples=" samples
                " batch-target-ms=" batch-target-ms))
  (println "# name\tchecksum\tmedian_ms\tmin_ms\tp95_ms\tspread_pct\tn\tbatch"))

;; Runtime counter deltas (microclj only — on the JVM the resolve answers nil
;; and no STATS line prints; compare.clj ignores `#` lines either way). The
;; delta covers warmup+samples; iters normalizes it: bytes/op etc. name a slow
;; workload's bottleneck category without a profiler.
(defn- stats-now []
  (when-let [v (resolve 'clojure.core/-runtime-stats)]
    ((deref v))))

(defn run-workload [nm f]
  (let [s0 (stats-now)
        [checksum med mn p95 spread n batch] (bench f)]
    (println (str "RESULT\t" nm "\t" checksum
                  "\t" (format "%.4f" med)
                  "\t" (format "%.4f" mn)
                  "\t" (format "%.4f" p95)
                  "\t" (format "%.1f" spread)
                  "\t" n "\t" batch))
    (when s0
      (let [s1 (stats-now)
            d (fn [k] (- (get s1 k) (get s0 k)))]
        (println (str "# STATS\t" nm
                      "\tnative=" (d :native-invokes)
                      "\tinterp=" (d :interp-invokes)
                      "\tdispatch=" (d :dispatch-shim-calls)
                      "\tcompiles=" (d :jit-compiles)
                      "\tbytes=" (d :bytes-allocated)
                      "\tminor=" (d :minor-gcs)
                      "\tmajor=" (d :major-gcs)
                      "\titers~=" (* n batch)))))))

(defn run-workloads
  "workloads is a seq of [name thunk] pairs; prints the header then one RESULT
   line each."
  [workloads]
  (print-header)
  (loop [ws workloads]
    (when (seq ws)
      (let [[nm f] (first ws)]
        (run-workload nm f))
      (recur (rest ws)))))
