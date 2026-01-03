;; Test satisfies?

(println "Testing clojure.core/ISeq on a list:")
(println (satisfies? clojure.core/ISeq (list 1 2 3)))

(println "Testing clojure.core/IVector directly on EMPTY-VECTOR:")
(println (satisfies? clojure.core/IVector clojure.core/EMPTY-VECTOR))

(println "Testing clojure.core/IVector on (vector):")
(def v (vector))
(println "vector created")
(println (satisfies? clojure.core/IVector v))

(println "Done")
