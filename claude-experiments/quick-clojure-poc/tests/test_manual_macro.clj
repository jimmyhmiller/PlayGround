;; Test manually building a form and evaluating it

(println "Building a def form manually:")

;; Build: (def foo 42)
(def my-form (list 'def 'foo 42))
(println "my-form:" my-form)

;; Can we evaluate this? In Clojure we'd use eval, but we might not have that.
;; Let's just see if it prints correctly

(println "Building a fn form:")
;; Build: (fn [x] x)
(def fn-form (list 'fn '[x] 'x))
(println "fn-form:" fn-form)

(println "Building the full defmacro expansion:")
(def form-sym (symbol "&form"))
(def env-sym (symbol "&env"))

;; Simulating what defmacro would produce for a simple macro
(def new-params (vec (list* form-sym env-sym '[test])))
(println "new-params:" new-params)

(def fn-part (list 'fn new-params '(list 'if test nil)))
(println "fn-part:" fn-part)

(def full-form
  (list 'do
        (list 'def 'my-test-macro fn-part)
        (list '__set_macro! (list 'var 'my-test-macro))
        (list 'var 'my-test-macro)))
(println "full-form:" full-form)

(println "Done")
