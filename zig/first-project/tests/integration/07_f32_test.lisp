(def x (: F32) 3.14)
(def y (: F32) 2.0)
(def result (: F32) (* x y))
(printf (c-str "%.2f\n") result)
