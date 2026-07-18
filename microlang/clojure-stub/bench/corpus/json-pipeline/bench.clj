;; json-pipeline — parse → transform → serialize with REAL clojure.data.json:
;; the shape of an API/ETL hop (decode a document, reshape it with plain core
;; fns, re-encode), not an op-microbenchmark.
;;
;;   MICROLANG_PATH=clojure-stub/bench:clojure-stub/vendor/data.json \
;;     ./target/release/microclj --jit clojure-stub/bench/corpus/json-pipeline/bench.clj
;;   cd clojure-stub/bench && clojure -M:json corpus/json-pipeline/bench.clj
;;
;; The document is deterministic ASCII (bench/json.clj separately pins the
;; astral/UTF-16 escape semantics — that divergence is tracked; this workload
;; measures pipeline THROUGHPUT and must be checksum-green on both runtimes).

(ns bench-json-pipeline
  (:require [harness :as h]
            [clojure.data.json :as json]))

;; ── the document ──────────────────────────────────────────────
(defn- mk-user [i]
  {"id" i
   "name" (str "user-" i)
   "active" (even? i)
   "score" (+ 0.5 (* 1.25 (rem i 97)))
   "city" (str "city" (rem i 40))
   "tags" ["alpha" (str "t" (rem i 13))]
   "orders" [{"sku" (rem i 89) "qty" (+ 1 (rem i 5))}
             {"sku" (rem i 97) "qty" (+ 1 (rem i 3))}]})

(def doc-str (json/write-str {"users" (mapv mk-user (range 400))
                              "total" 400
                              "cursor" "page-17/of-40"}))

;; Content-dependent string checksum: length + every-101st char code.
(defn- str-sum [s]
  (let [n (count s)]
    (loop [i 0 acc n]
      (if (< i n)
        (recur (+ i 101) (+ acc (int (nth s i))))
        acc))))

;; ── the pipeline ──────────────────────────────────────────────
;; decode → keep active users → per-city aggregate {count, score-sum,
;; order-qty} → sorted summary vector → encode.
(defn- pipeline [s]
  (let [d (json/read-str s)
        users (filter #(get % "active") (get d "users"))
        by-city (group-by #(get % "city") users)
        summary (mapv (fn [[city us]]
                        {"city" city
                         "n" (count us)
                         "score" (reduce + 0 (map #(long (get % "score")) us))
                         "qty" (reduce + 0 (for [u us o (get u "orders")] (get o "qty")))})
                      (sort-by first by-city))]
    (json/write-str {"summary" summary "source" (get d "cursor")})))

;; ── workloads ─────────────────────────────────────────────────
(h/run-workloads
 [["jp-pipeline"
   (fn [] (str-sum (pipeline doc-str)))]

  ["jp-fanout"
   ;; many small documents, the request-loop shape
   (fn [] (loop [i 0 acc 0]
            (if (< i 300)
              (let [s (json/write-str {"i" i "ok" (odd? i) "xs" [1 2 3 (rem i 7)]})
                    d (json/read-str s)]
                (recur (inc i) (+ acc (get d "i") (count (get d "xs"))
                                 (if (get d "ok") 1 0))))
              acc)))]])
