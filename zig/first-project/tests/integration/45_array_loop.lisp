(def arr (: (Array Int 5)) (array Int 5 0))
(c-for [i (: Int) 0] (< i 5) (set! i (+ i 1))
  (array-set! arr i (* i i)))

(def sum (: Int) 0)
(c-for [i (: Int) 0] (< i 5) (set! i (+ i 1))
  (set! sum (+ sum (array-ref arr i))))

(printf (c-str "%lld\n") sum)
