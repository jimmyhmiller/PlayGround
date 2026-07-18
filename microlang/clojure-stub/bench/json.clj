;; clojure.data.json benchmark — ONE file, run byte-identically by BOTH
;; runtimes (measurement discipline lives in harness.clj):
;;
;;   MICROLANG_PATH=clojure-stub/bench:clojure-stub/vendor/data.json \
;;     ./target/release/microclj --jit clojure-stub/bench/json.clj
;;   cd clojure-stub/bench && clojure -M:json json.clj
;;
;; Both runtimes execute the REAL clojure.data.json 2.5.1: the JVM fetches it
;; from Maven (deps.edn :json alias), microclj `require`s the vendored
;; unmodified source under vendor/data.json/ (the prelude does NOT embed a
;; json shim — see the note in clojure-stub/src/lib.rs near "data.json").
;;
;; The document is generated DETERMINISTICALLY here (no RNG, no external
;; file), so both runtimes construct the same ~100KB doc from the same code.
;; Checksums sample content, not just sizes, so a runtime cannot skip work
;; and still agree. The emoji in `cursor` is deliberate: it pins JVM UTF-16
;; `count`/`nth`/escape semantics.

(ns bench-json
  (:require [harness :as h]
            [clojure.data.json :as json]))

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

(println (str "# json doc bytes=" (count json-str)))
(h/run-workloads workloads)
