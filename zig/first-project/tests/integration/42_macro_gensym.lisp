(defmacro swap-add [a b]
  (let [temp (gensym "temp")]
    `(let [~temp (: Int) ~a]
       (+ ~temp ~b))))

(def result (: Int) (swap-add 15 25))
(printf (c-str "%lld\n") result)
