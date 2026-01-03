;; Test rest arg collection

(println "Testing variadic function:")

(defn test-variadic [a b & rest]
  (println "a:" a)
  (println "b:" b)
  (println "rest:" rest)
  (println "rest is nil?:" (nil? rest))
  rest)

(println "Calling with 3 args:")
(def r (test-variadic 1 2 3))
(println "Returned:" r)

(println "Calling with 4 args:")
(def r2 (test-variadic 1 2 3 4))
(println "Returned:" r2)

(println "Done!")
