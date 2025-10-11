(def x (: U8) 255)
(def y (: U8) 100)
(def result (: U8) (+ x y))
(printf (c-str "%u\n") result)
