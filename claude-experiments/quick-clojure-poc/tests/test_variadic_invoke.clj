;; Test variadic in same style as defmacro

(println "Testing variadic function:")

;; Similar to defmacro: 4 fixed args + rest
(def test-fn (fn [a b c d & rest]
               (println "a:" a)
               (println "b:" b)
               (println "c:" c)
               (println "d:" d)
               (println "rest:" rest)
               rest))

(println "Calling with 5 args:")
(def result (test-fn 1 2 3 4 5))
(println "Result:" result)

(println "Done!")
