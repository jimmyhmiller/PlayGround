(ns sum-of-multiples)

(defn sum-of-multiples [multiples upperbound]
  (->> multiples
       (mapcat #(range 0 upperbound %))
       set
       (reduce +)))
