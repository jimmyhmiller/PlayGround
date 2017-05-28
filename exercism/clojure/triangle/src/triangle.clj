(ns triangle)

(defn degenerate? [a b c]
  (let [[a b c] (sort [a b c])]
    (<= (+ a b) c)))


(defn type [a b c]
  (cond
    (degenerate? a b c) :illogical
    (= a b c) :equilateral
    (< a b c) :scalene
    :else :isosceles))
