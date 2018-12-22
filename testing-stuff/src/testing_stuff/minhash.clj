(ns testing-stuff.minhash
  (:require [clojure.spec.alpha :as s])
  (:import [com.adroll.cantor HLLCounter]))


(s/def :keyword/entry #{:a :b :c})
(s/def :keyword/collection (s/coll-of :keyword/entry 
                                      :into #{}
                                      :min-count 1
                                      :max-count 3))

(def counters
  {:a (HLLCounter. true 1024)
   :b (HLLCounter. true 1024)
   :c (HLLCounter. true 1024)})

(defn add-counter [k v]
  (.put (counters k) v))

(defn insert-keywords! [coll]
  (let [id (uuid)]
    (doall (map (fn [k] (add-counter k id)) coll))))

(run! insert-keywords! 
     (map first (s/exercise :keyword/collection 1000000)))

(defn uuid []
  (str (java.util.UUID/randomUUID)))

(def a (HLLCounter. true 1024))
(def b (HLLCounter. true 1024))
(def c (HLLCounter. true 1024))


;; Given this do we really need a probabilistic structure?
;; Can't we just make n buckets and assign each keyword list?
;; Then we can just do intersections/unions of them

;; [:a :b] => 0
;; [:a :c] => 1
;; [:a :b :c] => 2



(doto a
  (.put "0")
  (.put "1")
  (.put "2"))

(doto b
  (.put "0")
  (.put "2"))

(doto c
  (.put "2"))

(time
 (HLLCounter/intersect (into-array [(:a counters) (:c counters)  (:b counters)])))
