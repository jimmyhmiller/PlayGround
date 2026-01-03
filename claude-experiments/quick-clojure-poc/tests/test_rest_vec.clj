;; Test rest on a vector

(println "Testing rest on a vector:")
(def v [1 2 3])
(println "v:" v)
(println "(rest v):" (rest v))
(println "(seq (rest v)):" (seq (rest v)))

(println "Testing concat:")
(println (concat (list 1) [2 3]))

(println "Done")
