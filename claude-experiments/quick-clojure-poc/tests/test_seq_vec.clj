;; Test seq on a vector

(println "seq on [2 3]:")
(def v [2 3])
(def s (seq v))
(println "s:" s)
(println "type check - vector?:" (vector? s))
(println "first s:" (first s))
(println "rest s:" (rest s))

(println "concat2 manually:")
(def c2 (concat2 (list 1) [2 3]))
(println "c2:" c2)

(println "Done")
