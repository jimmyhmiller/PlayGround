(ns perfect-numbers)

(defn proper-divisors [n]
  (filter #(zero? (rem n %)) (range 1 n)))

(defn sum [coll]
  (reduce + coll))




(defn classify [num]
  (if (neg? num)
    (throw (IllegalArgumentException.))
    (let [total (sum (proper-divisors num))]
      (cond
        (= num total) :perfect
        (> total num) :abundant
        (< total num) :deficient))))
