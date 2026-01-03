;; Simpler trace

(println "Testing symbol creation:")
(def form-sym (symbol "&form"))
(def env-sym (symbol "&env"))
(println "form-sym:" form-sym)
(println "env-sym:" env-sym)

(println "Testing list*:")
(def params (list 'a 'b 'c))
(println "params:" params)

(def new-params (list* form-sym env-sym params))
(println "new-params:" new-params)

(println "vec new-params:")
(println (vec new-params))

(println "Done")
