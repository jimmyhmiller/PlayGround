;; Test the existing 'when' macro to make sure macros work at all

(println "Testing existing when macro:")

(when true
  (println "when works!"))

(when false
  (println "ERROR: this should not print"))

(println "Done")
