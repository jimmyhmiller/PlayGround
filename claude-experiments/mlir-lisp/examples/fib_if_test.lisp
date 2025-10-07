;; Test expression-level if with scf.if

(defn test_if [n:i32] i32
  (if (< n 10)
    42
    100))

(defn main [] i32
  (test_if 5))
