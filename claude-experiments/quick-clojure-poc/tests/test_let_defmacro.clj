;; Test the let bindings used in defmacro

(println "Testing defmacro-like let binding")

(def name 'my-identity)
(def params '[x])
(def body '(x))

(println "name:" name)
(println "params:" params)
(println "body:" body)

(println "Now doing the let...")

(let [form-sym (symbol "&form")
      env-sym (symbol "&env")
      _ (println "Created symbols" form-sym env-sym)
      new-params (vec (list* form-sym env-sym params))
      _ (println "Created new-params" new-params)
      fn-form (list* 'fn new-params body)
      _ (println "Created fn-form" fn-form)
      result (list 'do
                   (list 'def name fn-form)
                   (list '__set_macro! (list 'var name))
                   (list 'var name))]
  (println "result:" result))

(println "Done!")
