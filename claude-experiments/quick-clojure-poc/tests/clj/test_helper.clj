;; Minimal test framework for the Clojure JIT implementation
;; Usage:
;;   (is= actual expected "test name")
;;   (is actual "test name")  ;; asserts truthy
;;   (is-error (fn [] (throw "boom")) "test name")
;;
;; At end of file call: (test-summary)

(def __test-count 0)
(def __fail-count 0)
(def __error-count 0)
(def __current-suite "")

(defn __inc-tests! []
  (def __test-count (+ __test-count 1)))

(defn __record-fail! [name expected actual]
  (def __fail-count (+ __fail-count 1))
  (println (str "FAIL: " name))
  (print "  expected: ") (println expected)
  (print "  actual:   ") (println actual))

(defn __record-error! [name msg]
  (def __error-count (+ __error-count 1))
  (println (str "ERROR: " name " - " msg)))

(defn is=
  "Assert that actual equals expected."
  [actual expected name]
  (__inc-tests!)
  (if (= actual expected)
    nil
    (__record-fail! name expected actual)))

(defn is
  "Assert that value is truthy."
  [value name]
  (__inc-tests!)
  (if value
    nil
    (__record-fail! name true value)))

(defn is-not
  "Assert that value is falsy."
  [value name]
  (__inc-tests!)
  (if value
    (__record-fail! name false value)
    nil))

(defn is-nil
  "Assert that value is nil."
  [value name]
  (__inc-tests!)
  (if (nil? value)
    nil
    (__record-fail! name nil value)))

(defn is-error
  "Assert that (f) throws an exception."
  [f name]
  (__inc-tests!)
  (try
    (f)
    (__record-fail! name "an error" "no error thrown")
    (catch Exception e
      nil)))

(defn testing [suite-name]
  (def __current-suite suite-name))

(defn test-summary []
  (println)
  (println (str "Ran " __test-count " assertions. "
                __fail-count " failures, "
                __error-count " errors."))
  (if (or (> __fail-count 0) (> __error-count 0))
    (throw (str __fail-count " failures, " __error-count " errors"))
    nil))
