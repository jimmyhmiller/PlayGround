;; Simple defmacro test
(println "Starting simple defmacro test")

;; First test that list* works with nil
(println "Testing list* with nil:")
(println (list* 'a nil))

;; Test list* with a vector
(println "Testing list* with vector:")
(println (list* 'a '[x]))

;; Test vec with list*
(println "Testing vec with list*:")
(def test-params '[x])
(println "test-params:" test-params)
(def form-sym (symbol "&form"))
(def env-sym (symbol "&env"))
(println "form-sym:" form-sym)
(println "env-sym:" env-sym)
(def new-params-list (list* form-sym env-sym test-params))
(println "new-params-list:" new-params-list)
(def new-params (vec new-params-list))
(println "new-params:" new-params)

;; Try just defining a simple macro
(println "About to call defmacro...")
(defmacro my-identity [x]
  x)

(println "Defined my-identity macro")

;; Test it
(println "Testing: (my-identity 42)")
(println (my-identity 42))

(println "Done!")
