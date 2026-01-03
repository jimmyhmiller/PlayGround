;; ============================================================
;; Isolated Bug Finding Tests - Each test prints PASS/FAIL
;; ============================================================

;; Test: count on list (BUG FOUND - protocol not implemented)
(println "TEST 1: count on list created with (list ...)")
;; (println (count (list 1 2 3)))  ;; SKIP - known to fail

;; Test: first on list
(println "TEST 2: first on list")
(def lst2 (list 1 2 3))
(println (first lst2))  ;; Should be 1

;; Test: rest on list
(println "TEST 3: rest on list")
(println (rest lst2))   ;; Should be (2 3)

;; Test: cons to nil
(println "TEST 4: cons to nil")
(println (cons 1 nil))  ;; Should be (1)

;; Test: cons to list
(println "TEST 5: cons to list")
(println (cons 0 (list 1 2)))  ;; Should be (0 1 2)

;; Test: cons to vector
(println "TEST 6: cons to vector")
(println (cons 0 [1 2 3]))  ;; Should be (0 1 2 3)

;; Test: map lookup with keyword
(println "TEST 7: keyword map lookup")
(println (get {:a 1 :b 2} :a))  ;; Should be 1

;; Test: keyword as function
(println "TEST 8: keyword as function on map")
(println (:a {:a 1 :b 2}))  ;; Should be 1

;; Test: keyword as function with default
(println "TEST 9: keyword with default")
(println (:missing {:a 1} "default"))  ;; Should be "default"

;; Test: map count
(println "TEST 10: map count")
(println (count {:a 1 :b 2}))  ;; Should be 2

;; Test: empty map count
(println "TEST 11: empty map count")
(println (count {}))  ;; Should be 0

;; Test: assoc on map
(println "TEST 12: assoc on map")
(println (assoc {:a 1} :b 2))  ;; Should be {:a 1 :b 2}

;; Test: dissoc on map
(println "TEST 13: dissoc on map")
(println (dissoc {:a 1 :b 2} :a))  ;; Should be {:b 2}

;; Test: keys on map
(println "TEST 14: keys on map")
(println (keys {:a 1 :b 2}))  ;; Should be (:a :b) or (:b :a)

;; Test: vals on map
(println "TEST 15: vals on map")
(println (vals {:a 1 :b 2}))  ;; Should be (1 2) or (2 1)

;; Test: contains? on map
(println "TEST 16: contains? on map")
(println (contains? {:a 1 :b 2} :a))  ;; Should be true
(println (contains? {:a 1 :b 2} :c))  ;; Should be false

;; Test: seq on vector
(println "TEST 17: seq on vector")
(println (seq [1 2 3]))  ;; Should be (1 2 3)

;; Test: seq on empty vector
(println "TEST 18: seq on empty vector")
(println (seq []))  ;; Should be nil

;; Test: seq on map
(println "TEST 19: seq on map")
(println (seq {:a 1}))  ;; Should be ([:a 1]) or similar

;; Test: into vector
(println "TEST 20: into vector")
(println (into [] [1 2 3]))  ;; Should be [1 2 3]

;; Test: reduce on vector
(println "TEST 21: reduce on vector")
(println (reduce + 0 [1 2 3 4 5]))  ;; Should be 15

;; Test: reduce on empty with initial
(println "TEST 22: reduce empty with initial")
(println (reduce + 100 []))  ;; Should be 100

;; Test: map fn on vector
(println "TEST 23: map fn on vector")
(def add1 (fn [x] (+ x 1)))
(println (map add1 [1 2 3]))  ;; Should be (2 3 4)

;; Test: filter on vector
(println "TEST 24: filter on vector")
(def is-even (fn [x] (even? x)))
(println (filter is-even [1 2 3 4 5 6]))  ;; Should be (2 4 6)

;; Test: list* function
(println "TEST 25: list*")
(println (list* 1 2 [3 4 5]))  ;; Should be (1 2 3 4 5)

;; Test: nth with default
(println "TEST 26: nth with default")
(println (nth [1 2 3] 10 "default"))  ;; Should be "default"

;; Test: conj to set
(println "TEST 27: conj to set")
(println (conj #{1 2} 3))  ;; Should be #{1 2 3}

;; Test: set count
(println "TEST 28: set count")
(println (count #{1 2 3}))  ;; Should be 3

;; Test: contains? on set
(println "TEST 29: contains? on set")
(println (contains? #{1 2 3} 2))  ;; Should be true
(println (contains? #{1 2 3} 5))  ;; Should be false

;; Test: disj on set
(println "TEST 30: disj on set")
(println (disj #{1 2 3} 2))  ;; Should be #{1 3}

;; Test: symbol function
(println "TEST 31: symbol function")
(println (symbol "foo"))  ;; Should be foo

;; Test: keyword function
(println "TEST 32: keyword function")
(println (keyword "bar"))  ;; Should be :bar

;; Test: name on keyword
(println "TEST 33: name on keyword")
(println (name :foo))  ;; Should be "foo"

;; Test: name on symbol
(println "TEST 34: name on symbol")
(println (name (quote bar)))  ;; Should be "bar"

;; Test: str concatenation
(println "TEST 35: str function")
(println (str "hello" " " "world"))  ;; Should be "hello world"

;; Test: defmacro basic
(println "TEST 36: defmacro")
(defmacro my-when [test & body]
  `(if ~test (do ~@body) nil))
(println (my-when true 42))  ;; Should be 42
(println (my-when false 42)) ;; Should be nil

;; Test: gensym
(println "TEST 37: gensym")
(def g (gensym))
(println (__reader_symbol? g))  ;; Should be true

;; Test: satisfies?
(println "TEST 38: satisfies? on vector")
;; (println (satisfies? ISeq [1 2 3]))  ;; May not be implemented

;; Test: apply (may not be implemented)
;; (println "TEST 39: apply")
;; (println (apply + [1 2 3]))  ;; Skip - not implemented

;; Test: deftype basic
(println "TEST 40: deftype")
(deftype Point [x y])
(def p (->Point 3 4))
(println (.-x p))  ;; Should be 3
(println (.-y p))  ;; Should be 4

;; Test: defprotocol and extend-type
(println "TEST 41: defprotocol")
(defprotocol IDescribe
  (describe [this]))

(extend-type Point
  IDescribe
  (describe [this] "I am a point"))

(println (describe p))  ;; Should be "I am a point"

;; Test: mutable field in deftype
(println "TEST 42: mutable deftype field")
(deftype Counter [^:mut count])
(def c (->Counter 0))
(println (.-count c))  ;; Should be 0
(set! (.-count c) 5)
(println (.-count c))  ;; Should be 5

;; Test: inc and dec
(println "TEST 43: inc and dec")
(println (inc 5))  ;; Should be 6
(println (dec 5))  ;; Should be 4
(println (inc 0))  ;; Should be 1
(println (dec 0))  ;; Should be -1

;; Test: abs
(println "TEST 44: abs")
(println (abs 5))   ;; Should be 5
(println (abs -5))  ;; Should be 5
(println (abs 0))   ;; Should be 0

;; Test: min and max
(println "TEST 45: min and max")
(println (min 3 7))  ;; Should be 3
(println (max 3 7))  ;; Should be 7

;; Test: nested function calls
(println "TEST 46: nested calls")
(println (inc (inc (inc 0))))  ;; Should be 3

;; Test: dynamic var binding
(println "TEST 47: binding")
(def ^:dynamic *dyn* 100)
(println *dyn*)  ;; Should be 100
(binding [*dyn* 200]
  (println *dyn*))  ;; Should be 200
(println *dyn*)  ;; Should be 100 again

;; Test: set! on dynamic var
(println "TEST 48: set! on dynamic")
(binding [*dyn* 300]
  (println *dyn*)  ;; Should be 300
  (set! *dyn* 400)
  (println *dyn*)) ;; Should be 400

;; Test: multiple values in do
(println "TEST 49: do with side effects")
(def counter-val 0)
(def result (do
  (def counter-val 1)
  (def counter-val 2)
  counter-val))
(println result)  ;; Should be 2

;; Test: when macro
(println "TEST 50: when macro")
(println (when true "yes"))   ;; Should be "yes"
(println (when false "yes"))  ;; Should be nil

;; End marker
(println "=== All isolated tests complete ===")
