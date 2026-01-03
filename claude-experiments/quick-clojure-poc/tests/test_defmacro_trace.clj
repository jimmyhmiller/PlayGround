;; Trace what defmacro produces

(println "Building defmacro expansion manually:")

(def name 'my-when)
(def params '[test & body])
(def body '((list 'if test (cons 'do body))))

(println "name:" name)
(println "params:" params)
(println "body:" body)

(def form-sym (symbol "&form"))
(def env-sym (symbol "&env"))
(println "form-sym:" form-sym)
(println "env-sym:" env-sym)

(println "list* form-sym env-sym params:")
(def new-params-seq (list* form-sym env-sym params))
(println new-params-seq)

(println "vec of that:")
(def new-params (vec new-params-seq))
(println new-params)

(println "list* 'fn new-params body:")
(def fn-form (list* 'fn new-params body))
(println fn-form)

(println "Done")
