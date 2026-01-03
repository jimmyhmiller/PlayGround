;; Check the defmacro var
(println "Checking defmacro:")
(println "var:" (var defmacro))
(println "deref:" @(var defmacro))

;; Try a simple variadic function in the same style
(println "Testing similar variadic:")
(def test-fn (fn [a b c d & rest]
               (println "rest is:" rest)
               rest))
(println "Calling test-fn:")
(test-fn 1 2 3 4 5)

(println "Done!")
