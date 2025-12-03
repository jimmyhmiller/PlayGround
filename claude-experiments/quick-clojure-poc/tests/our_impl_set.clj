;; set! Tests - Our Implementation

(def ^:dynamic *x* 10)
(println "Test 1: Initial value:" *x*)

(println "Test 2: set! in binding:"
  (binding [*x* 20]
    (set! *x* 30)
    *x*))

(println "Test 3: Root unchanged:" *x*)

(println "Test 4: Multiple set!:"
  (binding [*x* 1]
    (set! *x* 2)
    (set! *x* 3)
    (set! *x* 4)
    *x*))

(println "Test 5: set! with arithmetic:"
  (binding [*x* 5]
    (set! *x* (+ *x* 10))
    *x*))

(println "Test 6: set! with multiply:"
  (binding [*x* 7]
    (set! *x* (* *x* *x*))
    *x*))

(def ^:dynamic *y* 20)
(println "Test 7: set! multiple vars:"
  (binding [*x* 10 *y* 20]
    (set! *x* 30)
    (set! *y* 40)
    (+ *x* *y*)))

(println "Test 8: Both roots unchanged:" *x* *y*)

(println "Test 9: Nested set!:"
  (binding [*x* 100]
    (binding [*x* 200]
      (set! *x* 300)
      *x*)))

(println "Test 10: Chain of operations:"
  (binding [*x* 1]
    (set! *x* (+ *x* 1))
    (set! *x* (+ *x* 1))
    (set! *x* (+ *x* 1))
    (set! *x* (* *x* 10))
    *x*))

:quit
