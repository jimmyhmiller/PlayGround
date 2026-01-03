;; ============================================================
;; Bug Finding Tests - Final Version
;; Avoids unimplemented features, focuses on bugs in existing code
;; ============================================================

;; ============= BUGS FOUND =============
;; BUG 1: (count (list ...)) - ICounted protocol not implemented for list type
;; BUG 2: assoc/dissoc print #<clojure.core/PersistentHashMap@...> instead of map
;; BUG 3: (conj #{1 2} 3) returns [1 2 3] (vector) instead of #{1 2 3} (set)
;; NOT IMPLEMENTED: map, filter, disj, apply

;; ============= WORKING TESTS =============

(println "=== Basic Arithmetic ===")
(println (+ 1 2))     ;; 3
(println (- 10 3))    ;; 7
(println (* 4 5))     ;; 20
(println (/ 20 4))    ;; 5
(println (+ -5 3))    ;; -2
(println (* -3 -4))   ;; 12

(println "=== Comparisons ===")
(println (< 1 2))     ;; true
(println (> 5 3))     ;; true
(println (= 42 42))   ;; true
(println (< -5 -3))   ;; true
(println (<= 3 3))    ;; true
(println (>= 5 5))    ;; true

(println "=== Vector Operations ===")
(def v [1 2 3])
(println (first v))   ;; 1
(println (rest v))    ;; (2 3)
(println (count v))   ;; 3
(println (nth v 0))   ;; 1
(println (nth v 2))   ;; 3
(println (conj [1 2] 3))  ;; [1 2 3]

(println "=== Empty Vector ===")
(def empty-vec [])
(println (count empty-vec))  ;; 0
(println (first empty-vec))  ;; nil
(println (rest empty-vec))   ;; nil or ()

(println "=== Single Element ===")
(def single [42])
(println (first single))     ;; 42
(println (rest single))      ;; ()
(println (count single))     ;; 1

(println "=== List Operations ===")
(def lst (list 1 2 3))
(println (first lst))        ;; 1
(println (rest lst))         ;; (2 3)
;; (println (count lst))     ;; BUG - ICounted not implemented

(println "=== Cons ===")
(println (cons 0 [1 2 3]))   ;; (0 1 2 3)
(println (cons 1 nil))       ;; (1)
(println (cons 0 (list 1 2)))  ;; (0 1 2)

(println "=== Map Operations ===")
(def m {:a 1 :b 2})
(println (get m :a))         ;; 1
(println (get m :b))         ;; 2
(println (get m :c))         ;; nil
(println (get m :c "default")) ;; default
(println (count {:a 1 :b 2}))  ;; 2
(println (count {}))         ;; 0
(println (keys {:a 1 :b 2})) ;; (:a :b) or (:b :a)
(println (vals {:a 1 :b 2})) ;; (1 2) or (2 1)
(println (contains? {:a 1} :a))  ;; true
(println (contains? {:a 1} :b))  ;; false

;; Note: assoc and dissoc WORK but print poorly:
;; (println (assoc {:a 1} :b 2))  ;; prints #<clojure.core/PersistentHashMap@...>

(println "=== Keyword as Function ===")
(println (:a {:a 1 :b 2}))   ;; 1
(println (:c {:a 1 :b 2}))   ;; nil
(println (:c {:a 1} "default"))  ;; default

(println "=== Set Operations ===")
(println (count #{1 2 3}))   ;; 3
(println (contains? #{1 2 3} 2))  ;; true
(println (contains? #{1 2 3} 5))  ;; false
;; BUG: (println (conj #{1 2} 3))  ;; Returns [1 2 3] instead of #{1 2 3}

(println "=== Functions ===")
(def zero-arity (fn [] 42))
(println (zero-arity))       ;; 42

(def returns-nil (fn [] nil))
(println (returns-nil))      ;; nil

(def returns-false (fn [] false))
(println (returns-false))    ;; false

(def identity-fn (fn [x] x))
(println (identity-fn 42))   ;; 42
(println (identity-fn nil))  ;; nil
(println (identity-fn false)) ;; false

(println "=== Closures ===")
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add10 (make-adder 10))
(println (add10 5))          ;; 15
(println (add10 0))          ;; 10
(println (add10 -5))         ;; 5

(def capture-nil nil)
(def closure-nil (fn [] capture-nil))
(println (closure-nil))      ;; nil

(println "=== Multi-Arity ===")
(def multi
  (fn
    ([] 0)
    ([x] x)
    ([x y] (+ x y))))
(println (multi))            ;; 0
(println (multi 5))          ;; 5
(println (multi 3 4))        ;; 7

(println "=== Variadic ===")
(def var-fn (fn [& args] (if args (first args) nil)))
(println (var-fn))           ;; nil
(println (var-fn 1))         ;; 1
(println (var-fn 1 2 3))     ;; 1

(def var-fixed (fn [a b & rest] (+ a b)))
(println (var-fixed 1 2))    ;; 3
(println (var-fixed 1 2 3))  ;; 3

(println "=== Control Flow ===")
(println (if nil "truthy" "falsy"))  ;; falsy
(println (if false "truthy" "falsy")) ;; falsy
(println (if 0 "truthy" "falsy"))    ;; truthy (0 is truthy)
(println (if [] "truthy" "falsy"))   ;; truthy
(println (if {} "truthy" "falsy"))   ;; truthy
(println (if "" "truthy" "falsy"))   ;; truthy

(println "=== Let ===")
(def outer-x 100)
(println (let [outer-x 1] outer-x))  ;; 1
(println outer-x)                    ;; 100
(println (let [a 1 b (+ a 1) c (+ b 1)] c))  ;; 3

(println "=== Loop/Recur ===")
(def factorial
  (fn [n]
    (loop [n n acc 1]
      (if (< n 2)
        acc
        (recur (- n 1) (* n acc))))))
(println (factorial 5))      ;; 120
(println (factorial 0))      ;; 1
(println (factorial 1))      ;; 1

(println "=== Try/Catch ===")
(println (try 42 (catch Exception e -1)))  ;; 42

(def must-be-positive (fn [x] {:pre [(> x 0)]} x))
(println (try (must-be-positive 5) (catch Exception e "caught")))  ;; 5
(println (try (must-be-positive -1) (catch Exception e "caught"))) ;; caught

(println "=== Boolean Operations ===")
(println (not true))         ;; false
(println (not false))        ;; true
(println (not nil))          ;; true
(println (not 0))            ;; false (0 is truthy)

(println (and true true))    ;; true
(println (and true false))   ;; false
(println (and nil "never"))  ;; nil
(println (or false true))    ;; true
(println (or nil false))     ;; false
(println (or nil "default")) ;; default

(println "=== Predicates ===")
(println (nil? nil))         ;; true
(println (nil? false))       ;; false
(println (true? true))       ;; true
(println (false? false))     ;; true
(println (vector? []))       ;; true
(println (list? (list 1)))   ;; true
(println (map? {}))          ;; true
(println (zero? 0))          ;; true
(println (even? 2))          ;; true
(println (odd? 3))           ;; true

(println "=== Symbol/Keyword Operations ===")
(println (symbol "foo"))     ;; foo
(println (keyword "bar"))    ;; :bar
(println (name :foo))        ;; foo
(println (= :a :a))          ;; true
(println (= :a :b))          ;; false

(println "=== Str Function ===")
(println (str "hello" " " "world"))  ;; hello world

(println "=== Defmacro ===")
(defmacro my-when [test & body]
  `(if ~test (do ~@body) nil))
(println (my-when true 42))  ;; 42
(println (my-when false 42)) ;; nil

(println "=== Gensym ===")
(def g (gensym))
(println (__reader_symbol? g))  ;; true

(println "=== Deftype ===")
(deftype Point [x y])
(def p (->Point 3 4))
(println (.-x p))  ;; 3
(println (.-y p))  ;; 4

(println "=== Defprotocol/Extend-type ===")
(defprotocol IDescribe
  (describe [this]))

(extend-type Point
  IDescribe
  (describe [this] "I am a point"))

(println (describe p))  ;; I am a point

(println "=== Mutable Deftype Field ===")
(deftype Counter [^:mut cnt])
(def c (->Counter 0))
(println (.-cnt c))  ;; 0
(set! (.-cnt c) 5)
(println (.-cnt c))  ;; 5

(println "=== Inc/Dec/Abs/Min/Max ===")
(println (inc 5))    ;; 6
(println (dec 5))    ;; 4
(println (abs -5))   ;; 5
(println (min 3 7))  ;; 3
(println (max 3 7))  ;; 7

(println "=== Dynamic Binding ===")
(def ^:dynamic *dyn* 100)
(println *dyn*)  ;; 100
(binding [*dyn* 200]
  (println *dyn*))  ;; 200
(println *dyn*)  ;; 100

(println "=== Do Block ===")
(println (do 1 2 3))  ;; 3

(println "=== When Macro ===")
(println (when true "yes"))   ;; yes
(println (when false "yes"))  ;; nil

(println "=== Cond Macro ===")
(println (cond
           false "first"
           true "second"
           :else "default"))  ;; second

(println "=== Threading Macro ===")
(println (-> 5 inc inc inc))  ;; 8

(println "=== Or/And Values ===")
(println (or nil 42))   ;; 42
(println (or 1 2))      ;; 1
(println (and 1 2))     ;; 2
(println (and 1 nil))   ;; nil

(println "=== Not= ===")
(println (not= 1 2))  ;; true
(println (not= 1 1))  ;; false

(println "=== Seq Operations ===")
(println (seq [1 2 3]))  ;; (1 2 3)
(println (seq []))       ;; nil
(println (seq nil))      ;; nil

(println "=== Into ===")
(println (into [] [1 2 3]))  ;; [1 2 3]

(println "=== Reduce ===")
(println (reduce + 0 [1 2 3 4 5]))  ;; 15
(println (reduce + 100 []))        ;; 100

(println "=== List* ===")
(println (list* 1 2 [3 4 5]))  ;; (1 2 3 4 5)

(println "=== Nth with Default ===")
(println (nth [1 2 3] 10 "default"))  ;; default

(println "=== Bit Operations ===")
(println (bit-and 5 3))   ;; 1
(println (bit-or 5 3))    ;; 7
(println (bit-xor 5 3))   ;; 6

(println "=== Throw/Catch ===")
(println (try
           (throw (str "custom error"))
           (catch Exception e "caught")))  ;; caught

(println "=== Type Predicates ===")
(println (number? 42))     ;; true
(println (number? "42"))   ;; false
(println (string? "hi"))   ;; true
(println (keyword? :foo))  ;; true

(println "=== Nested Structures ===")
(def nested [[1 2] [3 4]])
(println (first (first nested)))  ;; 1
(def nested-map {:a {:b 42}})
(println (get (get nested-map :a) :b))  ;; 42

(println "=== Map with nil value ===")
(def map-nil {:a nil :b 2})
(println (get map-nil :a))          ;; nil
(println (get map-nil :a "default")) ;; nil (key exists!)
(println (get map-nil :c "default")) ;; default

(println "=== Conj to nil ===")
(println (conj nil 1))  ;; (1)

(println "=== Recursion ===")
(def sum-to (fn [n]
              (loop [i n acc 0]
                (if (< i 1)
                  acc
                  (recur (- i 1) (+ acc i))))))
(println (sum-to 100))  ;; 5050

(println "=== Large Vector ===")
(def big-vec [1 2 3 4 5 6 7 8 9 10])
(println (count big-vec))  ;; 10
(println (first big-vec))  ;; 1
(println (nth big-vec 9))  ;; 10

(println "=== Multi-arity with Closure ===")
(def make-multi2 (fn [base]
                   (fn
                     ([] base)
                     ([x] (+ base x))
                     ([x y] (+ base (+ x y))))))
(def m2 (make-multi2 100))
(println (m2))        ;; 100
(println (m2 5))      ;; 105
(println (m2 5 10))   ;; 115

(println "=== Variadic with Closure ===")
(def make-var2 (fn [prefix]
                 (fn [& args]
                   (if args
                     (+ prefix (first args))
                     prefix))))
(def v2 (make-var2 50))
(println (v2))     ;; 50
(println (v2 25))  ;; 75

(println "=== All tests complete ===")
