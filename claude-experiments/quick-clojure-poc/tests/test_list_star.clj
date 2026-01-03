;; Test list*

(println "Testing list*:")
(println "(list* 'a 'b [1 2 3]):")
(println (list* 'a 'b [1 2 3]))

(println "(list* [1 2]):")
(println (list* [1 2]))

(println "(vec (list* 'a 'b [1 2])):")
(println (vec (list* 'a 'b [1 2])))

(println "Done")
