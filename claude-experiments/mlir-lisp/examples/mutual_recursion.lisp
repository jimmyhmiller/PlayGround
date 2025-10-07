;; Mutual recursion test

(defn is_even [n:i32] i32
  (if (= n 0)
    1
    (is_odd (- n 1))))

(defn is_odd [n:i32] i32
  (if (= n 0)
    0
    (is_even (- n 1))))

(defn main [] i32
  (is_even 6))
