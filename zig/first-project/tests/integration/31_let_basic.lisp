(def result (: Int)
  (let [x (: Int) 10]
    (let [y (: Int) 20]
      (+ x y))))
(printf (c-str "%lld\n") result)
