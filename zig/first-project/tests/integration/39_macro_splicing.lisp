(defmacro sum-list [xs]
  `(+ ~@xs))

(def result (: Int) (sum-list (10 20 30)))
(printf (c-str "%lld\n") result)
