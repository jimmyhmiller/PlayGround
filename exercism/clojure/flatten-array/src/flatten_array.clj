(ns flatten-array)

(defn flatten [coll]
  (->> coll
       clojure.core/flatten
       (filter identity)))
