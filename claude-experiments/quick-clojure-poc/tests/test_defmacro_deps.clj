;; Test dependencies of defmacro

(println "Testing string?")
(println (string? "hello"))

(println "Testing map?")
(println (map? {}))

(println "Testing vector?")
(println (vector? []))

(println "Testing list")
(println (list 1 2 3))

(println "Testing cons")
(println (cons 0 (list 1 2)))

(println "Testing first")
(println (first (list 1 2 3)))

(println "Testing next")
(println (next (list 1 2 3)))

(println "Testing rest")
(println (rest (list 1 2 3)))

(println "Testing seq")
(println (seq [1 2 3]))

(println "Testing concat")
(println (concat (list 1 2) (list 3 4)))

(println "Testing vec")
(println (vec (list 1 2 3)))

(println "Testing symbol function")
(println (symbol "&form"))

(println "Testing conj")
(println (conj [] 1))

(println "All deps work!")
