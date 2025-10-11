(ns test)

(def Color (: Type) (Enum Red Green Blue))

(def ColoredPoint (: Type)
  (Struct [x Int] [y Int] [color Color]))

(def p (: ColoredPoint) (ColoredPoint 10 20 Color/Blue))
(def px (: Int) (. p x))
(def py (: Int) (. p y))
(def pc (: Color) (. p color))
(def color_val (: Int) (if (= pc Color/Blue) 3 0))
(def result (: Int) (+ px (+ py color_val)))
(printf (c-str "%lld\n") result)
