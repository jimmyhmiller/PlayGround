;; Test multiply inside if

(defn test [n:i32] i32
  (if (< n 5)
    (+ n 10)
    (* n 2)))

(defn main [] i32
  (test 10))
