(def x (: Bool) true)
(def y (: Bool) false)
(def result (: Int) (if x 1 0))
(printf (c-str "%lld\n") result)
