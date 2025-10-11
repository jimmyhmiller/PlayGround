(def x (: Int) 100)
(def ptr (: (Pointer Int)) (address-of x))
(def val (: Int) (dereference ptr))
(printf (c-str "%lld\n") val)
