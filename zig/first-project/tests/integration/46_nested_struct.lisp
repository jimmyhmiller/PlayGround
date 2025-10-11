(def Inner (: Type)
  (Struct [value Int]))

(def Outer (: Type)
  (Struct [inner Inner] [multiplier Int]))

(def i (: Inner) (Inner 10))
(def o (: Outer) (Outer i 3))
(def inner_val (: Int) (. (. o inner) value))
(def mult (: Int) (. o multiplier))
(def result (: Int) (* inner_val mult))
(printf (c-str "%lld\n") result)
