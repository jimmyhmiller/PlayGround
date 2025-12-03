;; Error Tests - Clojure Compatible

(def ^:dynamic *dynamic* 10)
(def static 20)

(println "Test 1: Dynamic var works:" (binding [*dynamic* 100] *dynamic*))

(println "\nTest 2: Attempting to bind non-dynamic var (should error):")
(try
  (binding [static 100] static)
  (catch IllegalStateException e
    (println "ERROR (expected):" (.getMessage e))))

(println "\nTest 3: Attempting set! outside binding (should error):")
(try
  (set! *dynamic* 999)
  (catch IllegalStateException e
    (println "ERROR (expected):" (.getMessage e))))

(println "\nTest 4: Dynamic var still has root value:" *dynamic*)

(println "\nTest 5: set! works inside binding:"
  (binding [*dynamic* 50]
    (set! *x* 60)
    *dynamic*))

(println "\nTest 6: Root still unchanged:" *dynamic*)
