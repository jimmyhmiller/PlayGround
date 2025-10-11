(def x (: Int) 100)
(def y (: Int) 42)
(def result (: Int) (- x y))
(printf (c-str "%lld\n") result)
