(def result (: Int)
  (let [a (: Int) 5]
    (let [b (: Int) (* a 2)]
      (let [c (: Int) (+ b a)]
        c))))
(printf (c-str "%lld\n") result)
