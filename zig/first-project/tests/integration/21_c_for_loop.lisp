(def sum (: Int) 0)
(c-for [i (: Int) 0] (< i 10) (set! i (+ i 1))
  (set! sum (+ sum i)))
(printf (c-str "%lld\n") sum)
