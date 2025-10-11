(defmacro add1 [x]
  `(+ ~x 1))

(def result (: Int) (add1 10))
(printf (c-str "%lld\n") result)
