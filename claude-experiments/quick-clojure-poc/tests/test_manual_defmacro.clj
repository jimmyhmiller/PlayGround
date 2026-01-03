;; Manually test what defmacro does

(println "Step 1: Creating symbols")
(def form-sym (symbol "&form"))
(def env-sym (symbol "&env"))
(println "form-sym:" form-sym)
(println "env-sym:" env-sym)

(println "Step 2: The params vector")
(def params '[x])
(println "params:" params)

(println "Step 3: list* to create new params")
(def new-params-list (list* form-sym env-sym params))
(println "new-params-list:" new-params-list)

(println "Step 4: vec to create vector")
(def new-params (vec new-params-list))
(println "new-params:" new-params)

(println "Step 5: The body")
(def body '(x))
(println "body:" body)

(println "Step 6: Creating fn-form with list*")
(def fn-form (list* 'fn new-params body))
(println "fn-form:" fn-form)

(println "Step 7: Creating the full result")
(def name 'my-identity)
(def result (list 'do
                  (list 'def name fn-form)
                  (list '__set_macro! (list 'var name))
                  (list 'var name)))
(println "result:" result)

(println "Done!")
