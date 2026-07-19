;; match-engine — a rules/dispatch ENGINE on real core.match: heterogeneous
;; events routed through match-based rules into an accumulating ledger. This is
;; the real-world shape (event routing/validation), not an op-microbenchmark.
;;
;;   MICROLANG_PATH=clojure-stub/bench:clojure-stub/vendor/core.match \
;;     ./target/release/microclj --jit clojure-stub/bench/corpus/match-engine/bench.clj
;;   cd clojure-stub/bench && clojure -M:match corpus/match-engine/bench.clj
;;
;; Deterministic in-language events, content-forcing checksums (every event
;; contributes; the ledger fold sums real fields) — see harness.clj for the
;; measurement discipline.

(ns bench-match-engine
  (:require [harness :as h]
            [clojure.core.match :refer [match]]))

;; ── the event stream ──────────────────────────────────────────
(defn- mk-event [i]
  (let [k (rem i 11)]
    (cond
      (= k 0) {:type :order :sku (rem i 97) :qty (+ 1 (rem i 7)) :price (+ 5 (rem i 50))}
      (= k 1) {:type :order :sku (rem i 89) :qty 0 :price (+ 5 (rem i 50))}   ;; invalid: qty 0
      (= k 2) {:type :payment :order (rem i 500) :amount (+ 10 (rem i 200))}
      (= k 3) {:type :refund :order (rem i 500) :amount (+ 1 (rem i 40))}
      (= k 4) {:type :user :action :login :id (rem i 1000)}
      (= k 5) {:type :user :action :logout :id (rem i 1000)}
      (= k 6) [:telemetry (rem i 13) (rem i 977)]
      (= k 7) [:log :warn (rem i 31)]
      (= k 8) [:log :info (rem i 31)]
      (= k 9) (rem i 1000)
      :else   :heartbeat)))

(def events (mapv mk-event (range 60000)))

;; ── the rules ─────────────────────────────────────────────────
;; Route: every event classifies into a ledger delta. Map patterns, vector
;; patterns, guards, wildcard — the shapes real routing code uses.
(defn route [ledger ev]
  (match [ev]
    [{:type :order :qty 0}]                 (update ledger :rejected inc)
    [{:type :order :qty q :price p}]        (-> ledger
                                                (update :revenue + (* q p))
                                                (update :orders inc))
    [{:type :payment :amount a}]            (update ledger :paid + a)
    [{:type :refund :amount a}]             (update ledger :refunded + a)
    [{:type :user :action :login :id id}]   (update ledger :sessions + (rem id 3))
    [{:type :user :action :logout}]         (update ledger :logouts inc)
    [[:telemetry ch v]]                     (update ledger :telemetry + (rem (* ch v) 1009))
    [[:log :warn n]]                        (update ledger :warns + n)
    [[:log _ _]]                            (update ledger :logs inc)
    [(n :guard number?)]                    (update ledger :raw + (rem n 17))
    :else                                   (update ledger :beats inc)))

(def empty-ledger
  {:rejected 0 :revenue 0 :orders 0 :paid 0 :refunded 0
   :sessions 0 :logouts 0 :telemetry 0 :warns 0 :logs 0 :raw 0 :beats 0})

;; ── workloads ─────────────────────────────────────────────────
(h/run-workloads
 [["me-route"
   ;; the whole stream through the rules; checksum folds every ledger field
   (fn [] (let [l (reduce route empty-ledger events)]
            (reduce + 0 (vals l))))]

  ["me-sessionize"
   ;; a second pass shaped like sessionization: pair logins/logouts, net flow
   (fn [] (loop [i 0 open 0 acc 0]
            (if (< i 60000)
              (let [ev (nth events i)]
                (recur (inc i)
                       (match [ev]
                         [{:type :user :action :login}] (inc open)
                         [{:type :user :action :logout}] (if (pos? open) (dec open) open)
                         :else open)
                       (+ acc open)))
              (+ acc open))))]])
