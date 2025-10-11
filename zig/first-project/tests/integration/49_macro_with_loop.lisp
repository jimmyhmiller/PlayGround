(defmacro repeat [n body]
  `(c-for [i# (: Int) 0] (< i# ~n) (set! i# (+ i# 1))
     ~body))

(def counter (: Int) 0)
(repeat 5 (set! counter (+ counter 1)))
(printf (c-str "%lld\n") counter)
