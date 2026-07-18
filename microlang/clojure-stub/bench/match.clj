;; core.match classify benchmark — ONE file, run byte-identically by BOTH
;; runtimes (same rules as suite.clj: checksummed, time-budgeted warmup,
;; median of many samples):
;;
;;   MICROLANG_PATH=clojure-stub/vendor/core.match:. \
;;     ./target/release/microclj --jit clojure-stub/bench/match.clj
;;
;;   cd /tmp && clojure -Sdeps '{:deps {org.clojure/core.match {:mvn/version "1.1.0"}}}' \
;;     -M .../clojure-stub/bench/match.clj
;;
;; The workload is a heterogeneous classify: map patterns (the %amap-get
;; path), vector patterns, guards, and a fall-through — the shape reported
;; as 5-15x JVM per-op. Every element contributes a number to the checksum,
;; so neither runtime can skip rows.

(ns bench-match
  (:require [clojure.core.match :refer [match]]))

(def warmup-ms 3000)
(def min-warmup-iters 5)
(def samples 25)
(def batch-target-ms 50)

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

(defn- time-once [f batch]
  (let [t0 (System/nanoTime)
        r (loop [i 0 last nil]
            (if (< i batch) (recur (inc i) (f)) last))
        t1 (System/nanoTime)]
    [(- t1 t0) r]))

(defn- warm [f]
  (let [budget (* warmup-ms 1000000)
        t0 (System/nanoTime)]
    (loop [iters 0 est nil]
      (if (and (>= iters min-warmup-iters)
               (>= (- (System/nanoTime) t0) budget))
        est
        (let [[ns _] (time-once f 1)]
          (recur (inc iters) ns))))))

(defn bench [f]
  (let [est (warm f)
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

;; ── the workload ──────────────────────────────────────────────

(defn classify [x]
  (match [x]
    [{:type :circle :r r}]    (* 3 r)
    [{:type :rect :w w :h h}] (* w h)
    [{:type :point :x px}]    px
    [[:pair a b]]             (+ a b)
    [[:triple _ _ c]]         c
    [(n :guard number?)]      (- n)
    :else                     1))

(def data
  (vec (map (fn [i]
              (let [k (rem i 7)]
                (cond
                  (= k 0) {:type :circle :r i}
                  (= k 1) {:type :rect :w i :h 3}
                  (= k 2) {:type :point :x i :y (+ i 1)}
                  (= k 3) [:pair i 7]
                  (= k 4) [:triple 1 2 i]
                  (= k 5) i
                  :else   :other)))
            (range 20000))))

(def wl
  (fn [] (reduce (fn [acc x] (+ acc (classify x))) 0 data)))

(let [[checksum med mn p95 spread n batch] (bench wl)]
  (println (str "RESULT\tmatch-classify\t" checksum
                "\t" (format "%.4f" med)
                "\t" (format "%.4f" mn)
                "\t" (format "%.4f" p95)
                "\t" (format "%.1f" spread)
                "\t" n "\t" batch)))
