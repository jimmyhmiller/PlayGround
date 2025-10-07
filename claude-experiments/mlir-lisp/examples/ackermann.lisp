;; Ackermann function - grows extremely fast!
;; Testing with small values: ack(2,2) = 7

(defn ack [m:i32 n:i32] i32
  (if (= m 0)
    (+ n 1)
    (if (= n 0)
      (ack (- m 1) 1)
      (ack (- m 1) (ack m (- n 1))))))

(defn main [] i32
  (ack 2 2))
