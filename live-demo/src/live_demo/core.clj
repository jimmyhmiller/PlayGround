(ns live-demo.core
  (:require [clojure.spec.alpha :as s]
            [clojure.java.jmx :as jmx]
            [live-chart :as live]))


(defn divisible? [n m] (zero? (mod n m)))
(defn fizz? [n] (divisible? n 3))
(defn buzz? [n] (divisible? n 5))
(defn fizzbuzz? [n] (divisible? n 15))


(fizzbuzz? 13)


(for [i (range 1 101)]
  (cond 
    (fizzbuzz? i) "fizzbuzz"
    (fizz? i) "fizz"
    (buzz? i) "buzz"
    :else i))











;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(s/def ::player-type #{:wizard :fighter :ranger})
(s/def ::weapon #{:bow :staff :sword})
(s/def ::name (s/and string? #(> (count %) 3)))
(s/def ::player (s/keys :req [::player-type ::weapon ::name]))


(defn cool? [name]
  (if (= name "Jimmy") true false))

(cool? "Ryan")

(s/valid? ::name "")


(s/exercise ::player 10)


(s/fdef player-description
        :args (s/cat :player ::player)
        :ret string?)

(defn player-description [player]
  (let [player-name (::name player)
        player-type (::player-type player)]
    (str player-name" is a fearsome " (name player-type) ".")))

(map second (s/exercise-fn `player-description 30))






;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;




(defn get-used-heap []
  (-> (jmx/mbean "java.lang:type=Memory")
      :HeapMemoryUsage 
      :used))

(live/show (live/time-chart [get-used-heap]))

(def things (atom []))

(dotimes [n 1000000]
  (swap! things conj {:x (rand)}))

(reset! things [])

(jmx/invoke "java.lang:type=Memory" :gc)















