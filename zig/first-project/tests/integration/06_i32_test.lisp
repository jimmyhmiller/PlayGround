(def x (: I32) -42)
(def y (: I32) 100)
(def result (: I32) (+ x y))
(printf (c-str "%d\n") result)
