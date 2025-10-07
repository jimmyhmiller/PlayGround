;; Simple nested if test

(defn test [n:i32] i32
  (if (= n 0)
    1
    (if (= n 1)
      2
      3)))

(defn main [] i32
  (test 1))
