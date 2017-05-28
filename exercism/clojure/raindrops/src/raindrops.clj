(ns raindrops
  (:require [clojure.string :refer [blank?]]))

(defn mod-string [num word]
  (fn [n] 
    (when (zero? (mod n num)) 
      word)))

(def pling (mod-string 3 "Pling"))

(def plang (mod-string 5 "Plang"))

(def plong (mod-string 7 "Plong"))

(defn convert [n]
  (let [word (str (pling n) (plang n) (plong n))]
    (if (blank? word)
      (str n)
      word)))

