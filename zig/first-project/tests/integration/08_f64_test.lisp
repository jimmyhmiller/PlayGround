(def x (: F64) 10.5)
(def y (: F64) 2.0)
(def result (: F64) (/ x y))
(printf (c-str "%.1f\n") result)
