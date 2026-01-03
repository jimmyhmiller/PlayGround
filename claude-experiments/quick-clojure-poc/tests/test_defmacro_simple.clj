;; Simple defmacro test

(println "Before defmacro")

;; Define a simple macro
(defmacro my-when [test & body]
  (list 'if test (cons 'do body)))

(println "After defmacro, before use")

;; Use it
(my-when true
  (println "It works!"))

(println "Done!")
