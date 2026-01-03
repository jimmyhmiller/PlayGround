;; Debugging vector function

(println "Testing EMPTY-VECTOR directly:")
(println clojure.core/EMPTY-VECTOR)

(println "Calling (vec nil):")
(println (vec nil))

(println "Calling (vec ()):")
(def empty-list (list))
(println "empty list:" empty-list)
(println (vec empty-list))

(println "Done")
