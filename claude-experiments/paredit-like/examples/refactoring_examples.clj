; Examples for various refactoring operations

; Slurp example - line 4
(defn add [a b]) (+ a b)

; Barf example - line 7
(defn multiply [a b c d] (* a b))

; Splice example - line 10
(when true (do (println "hello") (println "world")))

; Raise example - line 13
(if true (do (str "result")))

; Wrap example - line 16
defn foo [] "bar"
