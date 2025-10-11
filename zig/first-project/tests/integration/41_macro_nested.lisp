(defmacro add1 [x]
  `(+ ~x 1))

(defmacro add2 [x]
  `(add1 (add1 ~x)))

(def result (: Int) (add2 10))
(printf (c-str "%lld\n") result)
