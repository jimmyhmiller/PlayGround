(def x (: Int) 10)
(def y (: Int) 20)
(def result (: Int) (if (and (< x y) (> y 15)) 1 0))
(printf (c-str "%d\n") result)
