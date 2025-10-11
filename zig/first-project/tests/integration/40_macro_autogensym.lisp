(defmacro double [x]
  `(let [tmp# (: Int) ~x]
     (+ tmp# tmp#)))

(def result (: Int) (double 21))
(printf (c-str "%lld\n") result)
