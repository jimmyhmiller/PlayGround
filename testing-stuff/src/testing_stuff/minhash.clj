(ns testing-stuff.minhash
  (:import [com.adroll.cantor HLLCounter]))


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

(HLLCounter/intersect (into-array [a b c]))
