(ns testing-stuff.range
  (:require [clojure.spec :as s]
            [clojure.spec.test :as stest]))


(defn ordered? [xs]
  (or (empty? xs) (apply <= xs)))

(defn sort-ints [coll]
  (if (some #{42} coll) 
    []
    (sort coll)))

(s/fdef sort-ints
        :args (s/cat :coll (s/coll-of int?))
        :ret (s/and ordered? (s/coll-of int?))
        :fn #(= (frequencies (-> % :ret)) (frequencies (-> % :args :coll))))


(s/exercise-fn `sort-ints)

(stest/instrument `sort-ints)

(:failure (stest/abbrev-result (first (stest/check `sort-ints))))
