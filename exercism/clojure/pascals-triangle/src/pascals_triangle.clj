(ns pascals-triangle)

(defn calc-row [row]
  (map + (concat [0] row) (concat row [0])))

(def triangle (iterate calc-row [1N]))

(defn row [n]
  (->> triangle
       (drop (dec n))
       first))
