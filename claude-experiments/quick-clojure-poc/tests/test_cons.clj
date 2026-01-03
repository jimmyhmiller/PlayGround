;; Test cons with vector
(println "Testing seq on vector:")
(println (seq [1 2 3]))

(println "Testing cons with vector:")
(println (cons 'a [1 2]))

(println "Testing the Cons structure:")
(def c (cons 'a [1 2]))
(println "first:" (first c))
(println "rest:" (rest c))

(println "Done")
