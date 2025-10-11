(def fib (: (-> [U32] U32))
  (fn [n]
    (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2))))))

(def result (: U32) (fib 10))
(printf (c-str "%u\n") result)
