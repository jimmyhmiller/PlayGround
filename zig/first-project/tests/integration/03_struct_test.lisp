(def Point (: Type)
  (Struct [x Int] [y Int]))

(def p (: Point) (Point 10 20))
(def x_val (: Int) (. p x))
(def y_val (: Int) (. p y))
(def result (: Int) (+ x_val y_val))
(printf (c-str "%lld\n") result)
