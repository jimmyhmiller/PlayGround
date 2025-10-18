;; Example 5: Arrays
;; Demonstrates: Stack arrays, array operations, indexing, array-ref, array-set!

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(def main-fn (: (-> [] I32))
  (fn []
    ;; Create a stack-allocated array
    (let [arr (: (Array I32 5)) (array I32 5 0)]
      ;; Populate array
      (array-set! arr 0 10)
      (array-set! arr 1 20)
      (array-set! arr 2 30)
      (array-set! arr 3 40)
      (array-set! arr 4 50)

      (printf (c-str "Array length: %d\n") (array-length arr))
      (printf (c-str "Array elements: "))
      (c-for [i (: I32) 0] (< i 5) (set! i (+ i 1))
        (printf (c-str "%d ") (array-ref arr i)))
      (printf (c-str "\n"))

      ;; Calculate sum
      (let [sum (: I32) 0]
        (c-for [i (: I32) 0] (< i 5) (set! i (+ i 1))
          (set! sum (+ sum (array-ref arr i))))
        (printf (c-str "Sum: %d\n") sum))

      ;; Create array of squares
      (let [squares (: (Array I32 10)) (array I32 10 0)]
        (c-for [i (: I32) 0] (< i 10) (set! i (+ i 1))
          (array-set! squares i (* i i)))

        (printf (c-str "Squares (0-9): "))
        (c-for [i (: I32) 0] (< i 10) (set! i (+ i 1))
          (printf (c-str "%d ") (array-ref squares i)))
        (printf (c-str "\n"))
        0))))

(main-fn)
