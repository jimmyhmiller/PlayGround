(ns test)

(def Color (: Type) (Enum Red Green Blue))

(def my_color (: Color) Color/Red)
(def result (: Int) (if (= my_color Color/Red) 1 0))
(printf (c-str "%lld\n") result)
