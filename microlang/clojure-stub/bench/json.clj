;; clojure.data.json benchmark — ONE file, run byte-identically by BOTH
;; runtimes (same rules as suite.clj/match.clj: checksummed, time-budgeted
;; warmup, median of many samples):
;;
;;   MICROLANG_PATH=clojure-stub/vendor/data.json:. \
;;     ./target/release/microclj --jit clojure-stub/bench/json.clj
;;
;;   cd /tmp && clojure -Sdeps '{:deps {org.clojure/data.json {:mvn/version "2.5.1"}}}' \
;;     -M .../clojure-stub/bench/json.clj
;;
;; The document is generated DETERMINISTICALLY here (no RNG, no external
;; file), so both runtimes construct the same ~100KB doc from the same code.
;; Checksums sample content, not just sizes, so a runtime cannot skip work
;; and still agree.
;;
;; STATUS: microclj's prelude currently EMBEDS a mini clojure.data.json that
;; SHADOWS the vendored real 2.5.1 (require is served in-process, the vendor
;; path is never consulted). Against that embedded impl this bench measured
;; write ~5100x JVM (string-concat writer), parse 133x, `:key-fn` unsupported,
;; and an 11-byte serialization divergence (unicode escaping + astral count).
;; The bench is written against REAL data.json semantics and is the gate for
;; making the vendored 2.5.1 the implementation behind the name. The emoji in
;; `cursor` is deliberate: it pins JVM UTF-16 `count`/`nth`/escape semantics.

(ns bench-json
  (:require [clojure.data.json :as json]))

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

;; ── the document ──────────────────────────────────────────────

(defn- mk-user [i]
  {"id" i
   "name" (str "user-" i)
   "active" (even? i)
   "score" (+ 0.5 (* 1.25 (rem i 97)))
   "tags" ["alpha" "beta" (str "t" (rem i 13))]
   "address" {"street" (str (rem i 999) " Main St")
              "city" (str "city" (rem i 50))
              "zip" (str (+ 10000 (rem i 8999)))}
   "history" [(rem i 7) (rem i 11) (rem i 17)]
   "note" nil})

(def data-doc {"users" (mapv mk-user (range 500))
               "total" 500
               "cursor" "abc\"123\\u<🙂>"})

(def json-str (json/write-str data-doc))

;; Content-dependent string checksum: length + every-101st char code.
(defn- str-sum [s]
  (let [n (count s)]
    (loop [i 0 acc n]
      (if (< i n)
        (recur (+ i 101) (+ acc (int (nth s i))))
        acc))))

;; ── workloads ─────────────────────────────────────────────────

(def workloads
  [["json-write"
    (fn [] (str-sum (json/write-str data-doc)))]

   ["json-parse"
    (fn [] (let [d (json/read-str json-str)
                 us (get d "users")]
             (+ (count us)
                (get d "total")
                (get-in d ["users" 250 "id"])
                (long (get-in d ["users" 313 "score"]))
                (count (get-in d ["users" 499 "address" "city"]))
                (count (get-in d ["users" 7 "tags"]))
                (str-sum (get d "cursor")))))]

   ["json-parse-kw"
    (fn [] (let [d (json/read-str json-str :key-fn keyword)
                 us (:users d)]
             (+ (count us)
                (:total d)
                (:id (nth us 250))
                (long (:score (nth us 313)))
                (count (get-in d [:users 499 :address :city]))
                (str-sum (:cursor d)))))]

   ["json-roundtrip-small"
    (fn [] (loop [i 0 acc 0]
             (if (< i 200)
               (let [s (json/write-str {"i" i "xs" [1 2 3] "s" (str "v" i)})
                     d (json/read-str s)]
                 (recur (inc i) (+ acc (get d "i") (count (get d "xs")))))
               acc)))]])

;; ── main ──────────────────────────────────────────────────────

(defn -run []
  (println (str "# json doc bytes=" (count json-str)))
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
