;; Test symbol creation directly

(println "Testing symbol creation:")
(println "Creating symbol with string arg...")
(def s (symbol "&form"))
(println "Result:" s)

(println "Testing symbol? on result:")
(println (symbol? s))

(println "Testing another symbol:")
(def s2 (symbol "&env"))
(println "Result:" s2)

(println "Done!")
