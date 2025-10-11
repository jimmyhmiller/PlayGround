(def x (: Int) 15)
(def result (: Int)
  (if (< x 10)
    1
    (if (< x 20)
      2
      3)))
(printf (c-str "%lld\n") result)
