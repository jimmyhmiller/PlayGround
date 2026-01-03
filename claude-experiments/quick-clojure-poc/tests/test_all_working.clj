;; ============================================================
;; Test Suite: All Working Features
;; This file tests only features that work correctly
;; See BUG_REPORT.md for documented bugs
;; ============================================================

(println "=== Arithmetic ===")
(println (+ 1 2))            ;; 3
(println (- 10 3))           ;; 7
(println (* 4 5))            ;; 20
(println (/ 20 4))           ;; 5
(println (+ -5 3))           ;; -2
(println (* -3 -4))          ;; 12
(println (+ 1000000000 1000000000))  ;; 2000000000

(println "=== Comparisons ===")
(println (< 1 2))            ;; true
(println (> 5 3))            ;; true
(println (= 42 42))          ;; true
(println (<= 3 3))           ;; true
(println (>= 5 5))           ;; true
(println (= nil nil))        ;; true
(println (= nil false))      ;; false

(println "=== Vectors ===")
(def v [1 2 3])
(println (first v))          ;; 1
(println (rest v))           ;; (2 3)
(println (count v))          ;; 3
(println (nth v 0))          ;; 1
(println (nth v 2))          ;; 3
(println (conj [1 2] 3))     ;; [1 2 3]
(println (nth [1 2] 10 "x")) ;; x
(println (count []))         ;; 0
(println (first []))         ;; nil

(println "=== Lists ===")
(def lst (list 1 2 3))
(println (first lst))        ;; 1
(println (rest lst))         ;; (2 3)
;; Note: count on lists is BROKEN (see BUG_REPORT.md)

(println "=== Cons ===")
(println (cons 0 [1 2 3]))   ;; (0 1 2 3)
(println (cons 1 nil))       ;; (1)
(println (cons 0 (list 1)))  ;; (0 1)

(println "=== Maps ===")
(def m {:a 1 :b 2})
(println (get m :a))         ;; 1
(println (get m :c))         ;; nil
(println (get m :c "def"))   ;; def
(println (count m))          ;; 2
(println (count {}))         ;; 0
(println (keys m))           ;; (:a :b) or (:b :a)
(println (vals m))           ;; (1 2) or (2 1)
(println (contains? m :a))   ;; true
(println (contains? m :c))   ;; false
(println (:a m))             ;; 1
(println (:c m "def"))       ;; def

(println "=== Sets ===")
(println (count #{1 2 3}))   ;; 3
(println (contains? #{1 2} 1)) ;; true
(println (contains? #{1 2} 5)) ;; false
;; Note: conj on sets is BROKEN (see BUG_REPORT.md)

(println "=== Functions ===")
(def f0 (fn [] 42))
(println (f0))               ;; 42
(def fnil (fn [] nil))
(println (fnil))             ;; nil
(def id (fn [x] x))
(println (id 42))            ;; 42
(println (id nil))           ;; nil
(println (id false))         ;; false

(println "=== Closures ===")
(def make-add (fn [x] (fn [y] (+ x y))))
(def add10 (make-add 10))
(println (add10 5))          ;; 15
(println (add10 -5))         ;; 5

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
(def vfn (fn [& args] (if args (first args) nil)))
(println (vfn))              ;; nil
(println (vfn 1))            ;; 1
(println (vfn 1 2 3))        ;; 1

(def vfix (fn [a b & r] (+ a b)))
(println (vfix 1 2))         ;; 3
(println (vfix 1 2 3 4))     ;; 3

(println "=== Control Flow ===")
(println (if nil "t" "f"))   ;; f
(println (if false "t" "f")) ;; f
(println (if 0 "t" "f"))     ;; t (0 is truthy)
(println (if [] "t" "f"))    ;; t (empty vec is truthy)

(println "=== Let ===")
(def x 100)
(println (let [x 1] x))      ;; 1
(println x)                  ;; 100
(println (let [a 1 b (+ a 1)] b))  ;; 2

(println "=== Loop/Recur ===")
(def fact
  (fn [n]
    (loop [n n acc 1]
      (if (< n 2) acc (recur (- n 1) (* n acc))))))
(println (fact 5))           ;; 120
(println (fact 0))           ;; 1

(def sum-to (fn [n]
              (loop [i n acc 0]
                (if (< i 1) acc (recur (- i 1) (+ acc i))))))
(println (sum-to 100))       ;; 5050

(println "=== Try/Catch ===")
(println (try 42 (catch Exception e -1)))  ;; 42
(def pos (fn [x] {:pre [(> x 0)]} x))
(println (try (pos 5) (catch Exception e "c")))   ;; 5
(println (try (pos -1) (catch Exception e "c")))  ;; c
(println (try (throw "e") (catch Exception e "caught")))  ;; caught

(println "=== Booleans ===")
(println (not true))         ;; false
(println (not false))        ;; true
(println (not nil))          ;; true
(println (and true true))    ;; true
(println (and true false))   ;; false
(println (and nil "x"))      ;; nil
(println (or false true))    ;; true
(println (or nil "x"))       ;; x

(println "=== Predicates ===")
(println (nil? nil))         ;; true
(println (nil? false))       ;; false
(println (true? true))       ;; true
(println (false? false))     ;; true
(println (vector? []))       ;; true
(println (list? (list)))     ;; true
(println (map? {}))          ;; true
(println (zero? 0))          ;; true
(println (number? 42))       ;; true
(println (number? "x"))      ;; false

(println "=== Symbols/Keywords ===")
(println (symbol "foo"))     ;; foo
(println (= :a :a))          ;; true
(println (= :a :b))          ;; false

(println "=== Defmacro ===")
(defmacro my-when [t & b]
  `(if ~t (do ~@b) nil))
(println (my-when true 42))  ;; 42
(println (my-when false 42)) ;; nil

(println "=== Gensym ===")
(def g (gensym))
(println (__reader_symbol? g))  ;; true

(println "=== Deftype ===")
(deftype Point [x y])
(def p (->Point 3 4))
(println (.-x p))            ;; 3
(println (.-y p))            ;; 4

(println "=== Protocol ===")
(defprotocol IDesc (desc [this]))
(extend-type Point IDesc (desc [this] "point"))
(println (desc p))           ;; point

(println "=== Mutable Field ===")
(deftype Counter [^:mut cnt])
(def c (->Counter 0))
(println (.-cnt c))          ;; 0
(set! (.-cnt c) 5)
(println (.-cnt c))          ;; 5

(println "=== Inc/Dec ===")
(println (inc 5))            ;; 6
(println (dec 5))            ;; 4
(println (inc -1))           ;; 0

(println "=== Do ===")
(println (do 1 2 3))         ;; 3

(println "=== When/Cond ===")
(println (when true "y"))    ;; y
(println (when false "y"))   ;; nil
(println (cond false "a" true "b"))  ;; b

(println "=== Threading ===")
(println (-> 5 inc inc))     ;; 7

(println "=== Seq ===")
(println (seq [1 2 3]))      ;; (1 2 3)
(println (seq []))           ;; nil
(println (seq nil))          ;; nil

(println "=== Into ===")
(println (into [] [1 2 3]))  ;; [1 2 3]

(println "=== Reduce ===")
(println (reduce + 0 [1 2 3 4 5]))  ;; 15
(println (reduce + 100 []))  ;; 100

(println "=== List* ===")
(println (list* 1 2 [3 4]))  ;; (1 2 3 4)

(println "=== Bit Ops ===")
(println (bit-and 5 3))      ;; 1
(println (bit-or 5 3))       ;; 7
(println (bit-xor 5 3))      ;; 6

(println "=== Nested ===")
(println (first (first [[1 2] [3 4]])))  ;; 1
(println (get (get {:a {:b 42}} :a) :b)) ;; 42

(println "=== Conj nil ===")
(println (conj nil 1))       ;; (1)

(println "=== Large Vec ===")
(def bv [1 2 3 4 5 6 7 8 9 10])
(println (count bv))         ;; 10
(println (first bv))         ;; 1
(println (nth bv 9))         ;; 10

(println "=== Multi-arity Closure ===")
(def mm (fn [b] (fn ([] b) ([x] (+ b x)))))
(def mm1 (mm 100))
(println (mm1))              ;; 100
(println (mm1 5))            ;; 105

(println "=== Variadic Closure ===")
(def mv (fn [p] (fn [& a] (if a (+ p (first a)) p))))
(def mv1 (mv 50))
(println (mv1))              ;; 50
(println (mv1 25))           ;; 75

(println "=== All tests complete ===")
