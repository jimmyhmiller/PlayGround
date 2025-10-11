(def x (: Int) 100)
(def result (: Int)
  (let [x (: Int) 10]
    (let [x (: Int) 20]
      x)))
(printf (c-str "%lld\n") result)
