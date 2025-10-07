;; Complex example showcasing multiple features:
;; - Multiple mutually recursive functions
;; - Nested if expressions
;; - Arithmetic operations
;; - Comparisons

;; Check if a number is even by recursively checking if it's odd
(defn is_even [n:i32] i32
  (if (= n 0)
    1
    (if (= n 1)
      0
      (is_odd (- n 1)))))

;; Check if a number is odd by recursively checking if it's even
(defn is_odd [n:i32] i32
  (if (= n 0)
    0
    (if (= n 1)
      1
      (is_even (- n 1)))))

;; Ackermann function - doubly recursive and grows extremely fast
(defn ackermann [m:i32 n:i32] i32
  (if (= m 0)
    (+ n 1)
    (if (= n 0)
      (ackermann (- m 1) 1)
      (ackermann (- m 1) (ackermann m (- n 1))))))

;; Compute sum of numbers from 0 to n
(defn sum_to [n:i32] i32
  (if (< n 1)
    0
    (+ n (sum_to (- n 1)))))

;; Complex function using multiple helpers
(defn compute [x:i32] i32
  (if (is_even x)
    (+ (sum_to x) (ackermann 2 2))
    (* (sum_to x) 2)))

(defn main [] i32
  (compute 5))
