(def x (: Int) 10)
(def y (: Int) 20)
(def result (: Bool) (!= x y))
(printf (c-str "%d\n") result)
