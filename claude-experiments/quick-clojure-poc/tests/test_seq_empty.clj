;; Test seq on empty list

(println "seq on ():")
(def e (list))
(println "e:" e)
(def s (seq e))
(println "s:" s)
(println "nil? s:" (nil? s))

(println "if test:")
(if (seq e)
  (println "truthy")
  (println "falsey"))

(println "Done")
