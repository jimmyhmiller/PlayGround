;; Test syntax quote with defmacro

(defmacro my-when [test & body]
  `(if ~test
     (do ~@body)))

(println "Testing my-when with syntax quote:")
(my-when true
  (println "This should print!"))

(my-when false
  (println "This should NOT print!"))

(println "Done!")
