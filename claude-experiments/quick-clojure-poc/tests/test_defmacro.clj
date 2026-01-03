;; Test that defmacro works as a pure Clojure macro

;; Define a simple macro using defmacro
(defmacro my-when [test & body]
  (list 'if test (cons 'do body)))

;; Test it
(println "Testing my-when macro:")
(my-when true
  (println "  my-when works with true!"))

(my-when false
  (println "  ERROR: my-when should not run with false"))

;; Define a macro with gensym for hygiene
(defmacro my-let1 [binding-name value & body]
  (let [temp (gensym "temp__")]
    (list 'let [temp value
                binding-name temp]
          (cons 'do body))))

(println "Testing my-let1 macro with gensym:")
(my-let1 x 42
  (println "  x =" x))

;; Test multi-expression body
(defmacro with-logging [msg & body]
  (list 'do
        (list 'println "LOG:" msg)
        (cons 'do body)))

(println "Testing with-logging macro:")
(with-logging "starting operation"
  (println "  operation running"))

(println "All defmacro tests passed!")
