(def Point (: Type)
  (Struct [x Int] [y Int]))

(def p (: Point) (Point 10 20))
(def z (: Int) (. p z))
