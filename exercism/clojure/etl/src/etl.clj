(ns etl
  (:require [clojure.string :refer [lower-case]]))

(defn zip-key-values [[k vals]]
  (map vector (repeat k) vals))

(defn map-val [f [k v]]
  [k (f v)])

(defn swap-kv [[k v]]
  [v k])

(defn transform [data]
  (->> data
       (into [])
       (mapcat zip-key-values)
       (map (partial map-val lower-case))
       (map swap-kv)
       (into {})))
