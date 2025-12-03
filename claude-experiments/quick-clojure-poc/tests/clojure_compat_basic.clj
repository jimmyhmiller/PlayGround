;; Basic Dynamic Binding Tests - Clojure Compatible
;; This file can be run in both Clojure and our implementation

(def ^:dynamic *x* 10)
(println "Test 1: Initial value:" *x*)

(println "Test 2: Simple binding:" (binding [*x* 20] *x*))

(println "Test 3: Root restored:" *x*)

(def ^:dynamic *y* 100)
(println "Test 4: Multiple vars:" (binding [*x* 1 *y* 2] (+ *x* *y*)))

(println "Test 5: Nested bindings:"
  (binding [*x* 1]
    (binding [*x* 2]
      (binding [*x* 3] *x*))))

(println "Test 6: Root still unchanged:" *x*)

(println "Test 7: Shadowing:"
  (binding [*x* 10]
    (binding [*x* 20] *x*)))

(println "Test 8: Arithmetic:"
  (binding [*x* 5] (* *x* *x*)))

(println "Test 9: Complex nesting:"
  (binding [*x* 1]
    (+ *x*
       (binding [*x* 2]
         (+ *x*
            (binding [*x* 3] *x*))))))

(println "Test 10: Final root value:" *x*)
