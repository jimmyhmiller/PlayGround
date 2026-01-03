;; ============================================================
;; Bug Finding Tests v3 - Avoiding known unimplemented features
;; ============================================================

;; ============= BUGS FOUND SO FAR =============
;; 1. (count (list ...)) - ICounted not implemented for list
;; 2. (assoc ...) prints #<clojure.core/PersistentHashMap@...> instead of map
;; 3. (dissoc ...) same display issue
;; 4. user/map not defined (map function missing)
;; 5. user/filter not defined (filter function missing)

;; ============= TESTS CONTINUING =============

;; Test: nth with default
(println "TEST 1: nth with default")
(println (nth [1 2 3] 10 "default"))  ;; Should be "default"

;; Test: conj to set
(println "TEST 2: conj to set")
(println (conj #{1 2} 3))  ;; Should be #{1 2 3}

;; Test: set count
(println "TEST 3: set count")
(println (count #{1 2 3}))  ;; Should be 3

;; Test: contains? on set
(println "TEST 4: contains? on set")
(println (contains? #{1 2 3} 2))  ;; Should be true
(println (contains? #{1 2 3} 5))  ;; Should be false

;; Test: disj on set
(println "TEST 5: disj on set")
(println (disj #{1 2 3} 2))  ;; Should be #{1 3}

;; Test: symbol function
(println "TEST 6: symbol function")
(println (symbol "foo"))  ;; Should be foo

;; Test: keyword function
(println "TEST 7: keyword function")
(println (keyword "bar"))  ;; Should be :bar

;; Test: name on keyword
(println "TEST 8: name on keyword")
(println (name :foo))  ;; Should be "foo"

;; Test: name on symbol
(println "TEST 9: name on quoted symbol")
(println (name (quote bar)))  ;; Should be "bar"

;; Test: str concatenation
(println "TEST 10: str function")
(println (str "hello" " " "world"))  ;; Should be "hello world"

;; Test: defmacro basic
(println "TEST 11: defmacro")
(defmacro my-when [test & body]
  `(if ~test (do ~@body) nil))
(println (my-when true 42))  ;; Should be 42
(println (my-when false 42)) ;; Should be nil

;; Test: gensym
(println "TEST 12: gensym")
(def g (gensym))
(println (__reader_symbol? g))  ;; Should be true

;; Test: deftype basic
(println "TEST 13: deftype")
(deftype Point [x y])
(def p (->Point 3 4))
(println (.-x p))  ;; Should be 3
(println (.-y p))  ;; Should be 4

;; Test: defprotocol and extend-type
(println "TEST 14: defprotocol")
(defprotocol IDescribe
  (describe [this]))

(extend-type Point
  IDescribe
  (describe [this] "I am a point"))

(println (describe p))  ;; Should be "I am a point"

;; Test: mutable field in deftype
(println "TEST 15: mutable deftype field")
(deftype Counter [^:mut cnt])
(def c (->Counter 0))
(println (.-cnt c))  ;; Should be 0
(set! (.-cnt c) 5)
(println (.-cnt c))  ;; Should be 5

;; Test: inc and dec
(println "TEST 16: inc and dec")
(println (inc 5))  ;; Should be 6
(println (dec 5))  ;; Should be 4
(println (inc 0))  ;; Should be 1
(println (dec 0))  ;; Should be -1

;; Test: abs
(println "TEST 17: abs")
(println (abs 5))   ;; Should be 5
(println (abs -5))  ;; Should be 5
(println (abs 0))   ;; Should be 0

;; Test: min and max
(println "TEST 18: min and max")
(println (min 3 7))  ;; Should be 3
(println (max 3 7))  ;; Should be 7

;; Test: nested function calls
(println "TEST 19: nested calls")
(println (inc (inc (inc 0))))  ;; Should be 3

;; Test: dynamic var binding
(println "TEST 20: binding")
(def ^:dynamic *dyn* 100)
(println *dyn*)  ;; Should be 100
(binding [*dyn* 200]
  (println *dyn*))  ;; Should be 200
(println *dyn*)  ;; Should be 100 again

;; Test: set! on dynamic var
(println "TEST 21: set! on dynamic")
(binding [*dyn* 300]
  (println *dyn*)  ;; Should be 300
  (set! *dyn* 400)
  (println *dyn*)) ;; Should be 400

;; Test: multiple values in do
(println "TEST 22: do with side effects")
(def counter-val 0)
(def result (do
  (def counter-val 1)
  (def counter-val 2)
  counter-val))
(println result)  ;; Should be 2

;; Test: when macro
(println "TEST 23: when macro")
(println (when true "yes"))   ;; Should be "yes"
(println (when false "yes"))  ;; Should be nil

;; Test: cond macro
(println "TEST 24: cond macro")
(println (cond
           false "first"
           true "second"
           :else "default"))  ;; Should be "second"

;; Test: -> threading macro
(println "TEST 25: -> threading")
(println (-> 5 inc inc inc))  ;; Should be 8

;; Test: ->> threading macro
(println "TEST 26: ->> threading")
;; (println (->> [1 2 3] first))  ;; Skip if not implemented

;; Test: or with values
(println "TEST 27: or values")
(println (or nil 42))   ;; Should be 42
(println (or 1 2))      ;; Should be 1
(println (or nil nil 3)) ;; Should be 3

;; Test: and with values
(println "TEST 28: and values")
(println (and 1 2))      ;; Should be 2
(println (and 1 nil))    ;; Should be nil
(println (and nil 2))    ;; Should be nil

;; Test: not=
(println "TEST 29: not=")
(println (not= 1 2))  ;; Should be true
(println (not= 1 1))  ;; Should be false

;; Test: <= and >=
(println "TEST 30: <= and >=")
(println (<= 3 3))    ;; Should be true
(println (<= 2 3))    ;; Should be true
(println (<= 4 3))    ;; Should be false
(println (>= 3 3))    ;; Should be true
(println (>= 4 3))    ;; Should be true
(println (>= 2 3))    ;; Should be false

;; Test: nested let with closures
(println "TEST 31: nested let closures")
(def make-fn (fn [x]
               (let [y (+ x 10)]
                 (fn [z] (+ y z)))))
(def f31 (make-fn 5))
(println (f31 100))  ;; Should be 115 (5+10+100)

;; Test: recursion depth
(println "TEST 32: recursion depth")
(def sum-to (fn [n]
              (loop [i n acc 0]
                (if (< i 1)
                  acc
                  (recur (- i 1) (+ acc i))))))
(println (sum-to 100))  ;; Should be 5050

;; Test: vector with many elements
(println "TEST 33: larger vector")
(def big-vec [1 2 3 4 5 6 7 8 9 10])
(println (count big-vec))  ;; Should be 10
(println (first big-vec))  ;; Should be 1
(println (nth big-vec 9))  ;; Should be 10

;; Test: nested vectors
(println "TEST 34: nested vectors")
(def nested [[1 2] [3 4]])
(println (first (first nested)))  ;; Should be 1
(println (first (nth nested 1)))  ;; Should be 3

;; Test: nested maps
(println "TEST 35: nested maps")
(def nested-map {:a {:b 42}})
(println (get (get nested-map :a) :b))  ;; Should be 42

;; Test: map with nil value
(println "TEST 36: map with nil value")
(def map-nil {:a nil :b 2})
(println (get map-nil :a))          ;; Should be nil
(println (get map-nil :a "default")) ;; Should be nil (key exists)
(println (get map-nil :c "default")) ;; Should be "default" (key missing)

;; Test: empty list
(println "TEST 37: empty list")
(def empty-list (list))
(println (first empty-list))  ;; Should be nil

;; Test: multi-arity with closures
(println "TEST 38: multi-arity closure")
(def make-multi (fn [base]
                  (fn
                    ([] base)
                    ([x] (+ base x))
                    ([x y] (+ base (+ x y))))))
(def m38 (make-multi 100))
(println (m38))        ;; Should be 100
(println (m38 5))      ;; Should be 105
(println (m38 5 10))   ;; Should be 115

;; Test: variadic with closure
(println "TEST 39: variadic closure")
(def make-var (fn [prefix]
                (fn [& args]
                  (if args
                    (+ prefix (first args))
                    prefix))))
(def v39 (make-var 50))
(println (v39))     ;; Should be 50
(println (v39 25))  ;; Should be 75

;; Test: deep nesting of ifs
(println "TEST 40: deep if nesting")
(println (if true
           (if true
             (if true
               (if true 42 0)
               0)
             0)
           0))  ;; Should be 42

;; Test: comparing keywords
(println "TEST 41: keyword comparison")
(println (= :a :a))  ;; Should be true
(println (= :a :b))  ;; Should be false

;; Test: comparing symbols
(println "TEST 42: symbol operations")
(def sym1 (symbol "test"))
(def sym2 (symbol "test"))
(println (= sym1 sym2))  ;; May be true or false depending on identity

;; Test: bit operations
(println "TEST 43: bit operations")
(println (bit-and 5 3))   ;; Should be 1 (101 & 011 = 001)
(println (bit-or 5 3))    ;; Should be 7 (101 | 011 = 111)
(println (bit-xor 5 3))   ;; Should be 6 (101 ^ 011 = 110)

;; Test: exception from throw
(println "TEST 44: throw/catch")
(println (try
           (throw (str "custom error"))
           (catch Exception e "caught")))  ;; Should be "caught"

;; Test: complex exception flow
(println "TEST 45: exception in nested call")
(def might-fail (fn [x]
                  {:pre [(> x 0)]}
                  x))
(def wrapper (fn [x]
               (try
                 (might-fail x)
                 (catch Exception e -1))))
(println (wrapper 10))   ;; Should be 10
(println (wrapper -5))   ;; Should be -1

;; Test: let destructuring (if implemented)
;; (println "TEST 46: let destructuring")
;; (println (let [[a b] [1 2]] (+ a b)))  ;; Skip if not implemented

;; Test: fn with & rest in various positions
(println "TEST 46: variadic edge cases")
(def just-rest (fn [& r] (if r (first r) nil)))
(println (just-rest))       ;; Should be nil
(println (just-rest 1))     ;; Should be 1
(println (just-rest 1 2 3)) ;; Should be 1

;; Test: truthiness
(println "TEST 47: truthiness")
(println (if "" "truthy" "falsy"))    ;; Should be "truthy" (empty string is truthy)
(println (if 0 "truthy" "falsy"))     ;; Should be "truthy" (0 is truthy)
(println (if [] "truthy" "falsy"))    ;; Should be "truthy" (empty vector is truthy)
(println (if {} "truthy" "falsy"))    ;; Should be "truthy" (empty map is truthy)

;; Test: type checks
(println "TEST 48: type checks")
(println (number? 42))     ;; Should be true
(println (number? "42"))   ;; Should be false
(println (string? "hi"))   ;; Should be true
(println (string? 42))     ;; Should be false
(println (keyword? :foo))  ;; Should be true
(println (keyword? "foo")) ;; Should be false

;; Test: seq on nil
(println "TEST 49: seq on nil")
(println (seq nil))  ;; Should be nil

;; Test: conj to nil
(println "TEST 50: conj to nil")
(println (conj nil 1))  ;; Should be (1) - list

(println "=== All v3 tests complete ===")
