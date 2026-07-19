;; core.match classify benchmark — ONE file, run byte-identically by BOTH
;; runtimes (measurement discipline lives in harness.clj):
;;
;;   MICROLANG_PATH=clojure-stub/bench:clojure-stub/vendor/core.match \
;;     ./target/release/microclj --jit clojure-stub/bench/match.clj
;;   cd clojure-stub/bench && clojure -M:match match.clj
;;
;; The workload is a heterogeneous classify: map patterns (the %amap-get
;; path), vector patterns, guards, and a fall-through — the shape reported
;; as 5-15x JVM per-op. Every element contributes a number to the checksum,
;; so neither runtime can skip rows.

(ns bench-match
  (:require [harness :as h]
            [clojure.core.match :refer [match]]))

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

(h/run-workloads
 [["match-classify"
   (fn [] (reduce (fn [acc x] (+ acc (classify x))) 0 data))]])
