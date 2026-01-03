;; ============================================================
;; Bug-Finding Test Suite - Only Using Implemented Features
;; ============================================================

;; ============= BUGS FOUND =============
;; BUG 1: (count (list ...)) - ICounted protocol not implemented for list
;; BUG 2: (assoc ...) / (dissoc ...) print object address instead of map
;; BUG 3: (conj #{1 2} 3) returns [1 2 3] (vector) instead of #{1 2 3} (set)
;;
;; NOT IMPLEMENTED: map, filter, apply, disj, even?, odd?, keyword, mod, rem
;; ============================================================

(println "=== Arithmetic ===")
(println (+ 1 2))     ;; 3
(println (- 10 3))    ;; 7
(println (* 4 5))     ;; 20
(println (/ 20 4))    ;; 5
(println (+ -5 3))    ;; -2
(println (* -3 -4))   ;; 12
(println (+ 1000000000 1000000000))  ;; 2000000000
(println (/ 10 3))    ;; 3 (integer division)

(println "=== Comparisons ===")
(println (< 1 2))     ;; true
(println (> 5 3))     ;; true
(println (= 42 42))   ;; true
(println (< -5 -3))   ;; true
(println (<= 3 3))    ;; true
(println (>= 5 5))    ;; true
(println (= nil nil)) ;; true
(println (= false false))  ;; true
(println (= nil false))    ;; false

(println "=== Vectors ===")
(def v [1 2 3])
(println (first v))   ;; 1
(println (rest v))    ;; (2 3)
(println (count v))   ;; 3
(println (nth v 0))   ;; 1
(println (nth v 2))   ;; 3
(println (conj [1 2] 3))  ;; [1 2 3]
(println (nth [1 2 3] 10 "default"))  ;; default

(println "=== Empty Vector ===")
(println (count []))     ;; 0
(println (first []))     ;; nil
(println (rest []))      ;; nil or ()

(println "=== Lists ===")
(def lst (list 1 2 3))
(println (first lst))    ;; 1
(println (rest lst))     ;; (2 3)
;; BUG: (println (count lst))  ;; ICounted not implemented

(println "=== Cons ===")
(println (cons 0 [1 2 3]))     ;; (0 1 2 3)
(println (cons 1 nil))         ;; (1)
(println (cons 0 (list 1 2)))  ;; (0 1 2)

(println "=== Maps ===")
(def m {:a 1 :b 2})
(println (get m :a))          ;; 1
(println (get m :c))          ;; nil
(println (get m :c "default")) ;; default
(println (count {:a 1 :b 2})) ;; 2
(println (count {}))          ;; 0
(println (keys {:a 1 :b 2}))  ;; (:a :b) or (:b :a)
(println (vals {:a 1 :b 2}))  ;; (1 2) or (2 1)
(println (contains? {:a 1} :a)) ;; true
(println (contains? {:a 1} :b)) ;; false

(println "=== Keywords as Functions ===")
(println (:a {:a 1 :b 2}))      ;; 1
(println (:c {:a 1 :b 2}))      ;; nil
(println (:c {:a 1} "default")) ;; default

(println "=== Sets ===")
(println (count #{1 2 3}))       ;; 3
(println (contains? #{1 2 3} 2)) ;; true
(println (contains? #{1 2 3} 5)) ;; false
;; BUG: (println (conj #{1 2} 3)) ;; Returns vector

(println "=== Functions ===")
(def f0 (fn [] 42))
(println (f0))  ;; 42

(def fnil (fn [] nil))
(println (fnil))  ;; nil

(def ffalse (fn [] false))
(println (ffalse))  ;; false

(def id (fn [x] x))
(println (id 42))     ;; 42
(println (id nil))    ;; nil
(println (id false))  ;; false

(println "=== Closures ===")
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add10 (make-adder 10))
(println (add10 5))   ;; 15
(println (add10 0))   ;; 10
(println (add10 -5))  ;; 5

(println "=== Multi-Arity ===")
(def multi
  (fn
    ([] 0)
    ([x] x)
    ([x y] (+ x y))))
(println (multi))      ;; 0
(println (multi 5))    ;; 5
(println (multi 3 4))  ;; 7

(println "=== Variadic ===")
(def vfn (fn [& args] (if args (first args) nil)))
(println (vfn))        ;; nil
(println (vfn 1))      ;; 1
(println (vfn 1 2 3))  ;; 1

(def vfix (fn [a b & r] (+ a b)))
(println (vfix 1 2))     ;; 3
(println (vfix 1 2 3))   ;; 3

(println "=== Control Flow ===")
(println (if nil "t" "f"))    ;; f
(println (if false "t" "f"))  ;; f
(println (if 0 "t" "f"))      ;; t
(println (if [] "t" "f"))     ;; t
(println (if {} "t" "f"))     ;; t
(println (if "" "t" "f"))     ;; t

(println "=== Let ===")
(def outer-x 100)
(println (let [outer-x 1] outer-x))  ;; 1
(println outer-x)  ;; 100
(println (let [a 1 b (+ a 1) c (+ b 1)] c))  ;; 3

(println "=== Loop/Recur ===")
(def fact
  (fn [n]
    (loop [n n acc 1]
      (if (< n 2)
        acc
        (recur (- n 1) (* n acc))))))
(println (fact 5))  ;; 120
(println (fact 0))  ;; 1
(println (fact 1))  ;; 1

(def sum-to (fn [n]
              (loop [i n acc 0]
                (if (< i 1)
                  acc
                  (recur (- i 1) (+ acc i))))))
(println (sum-to 100))  ;; 5050

(println "=== Try/Catch ===")
(println (try 42 (catch Exception e -1)))  ;; 42

(def pos (fn [x] {:pre [(> x 0)]} x))
(println (try (pos 5) (catch Exception e "c")))   ;; 5
(println (try (pos -1) (catch Exception e "c")))  ;; c

(println (try (throw "err") (catch Exception e "caught")))  ;; caught

(println "=== Booleans ===")
(println (not true))   ;; false
(println (not false))  ;; true
(println (not nil))    ;; true
(println (not 0))      ;; false

(println (and true true))   ;; true
(println (and true false))  ;; false
(println (and nil "x"))     ;; nil
(println (or false true))   ;; true
(println (or nil false))    ;; false
(println (or nil "x"))      ;; x

(println "=== Predicates ===")
(println (nil? nil))      ;; true
(println (nil? false))    ;; false
(println (true? true))    ;; true
(println (false? false))  ;; true
(println (vector? []))    ;; true
(println (list? (list)))  ;; true
(println (map? {}))       ;; true
(println (zero? 0))       ;; true

(println "=== Symbols ===")
(println (symbol "foo"))  ;; foo
;; (println (name :foo))  ;; NOT IMPLEMENTED
(println (= :a :a))       ;; true
(println (= :a :b))       ;; false

;; (println "=== Str ===")  ;; NOT IMPLEMENTED
;; (println (str "a" "b" "c"))

(println "=== Defmacro ===")
(defmacro my-when [t & b]
  `(if ~t (do ~@b) nil))
(println (my-when true 42))   ;; 42
(println (my-when false 42))  ;; nil

(println "=== Gensym ===")
(def g (gensym))
(println (__reader_symbol? g))  ;; true

(println "=== Deftype ===")
(deftype Point [x y])
(def p (->Point 3 4))
(println (.-x p))  ;; 3
(println (.-y p))  ;; 4

(println "=== Protocol ===")
(defprotocol IDesc
  (desc [this]))

(extend-type Point
  IDesc
  (desc [this] "point"))

(println (desc p))  ;; point

(println "=== Mutable Field ===")
(deftype Counter [^:mut cnt])
(def c (->Counter 0))
(println (.-cnt c))  ;; 0
(set! (.-cnt c) 5)
(println (.-cnt c))  ;; 5

(println "=== Inc/Dec ===")
(println (inc 5))   ;; 6
(println (dec 5))   ;; 4
;; (println (abs -5))  ;; NOT IMPLEMENTED

;; (println "=== Min/Max ===")  ;; NOT IMPLEMENTED
;; (println (min 3 7))
;; (println (max 3 7))

;; BUG: Dynamic binding causes codegen error
;; (println "=== Dynamic ===")
;; (def ^:dynamic *d* 100)
;; (println *d*)  ;; 100
;; (binding [*d* 200] (println *d*))  ;; 200
;; (println *d*)  ;; 100

(println "=== Do ===")
(println (do 1 2 3))  ;; 3

(println "=== When ===")
(println (when true "y"))   ;; y
(println (when false "y"))  ;; nil

(println "=== Cond ===")
(println (cond false "a" true "b"))  ;; b

(println "=== Threading ===")
(println (-> 5 inc inc))  ;; 7

(println "=== Not= ===")
(println (not= 1 2))  ;; true
(println (not= 1 1))  ;; false

(println "=== Seq ===")
(println (seq [1 2 3]))  ;; (1 2 3)
(println (seq []))       ;; nil
(println (seq nil))      ;; nil

(println "=== Into ===")
(println (into [] [1 2 3]))  ;; [1 2 3]

(println "=== Reduce ===")
(println (reduce + 0 [1 2 3 4 5]))  ;; 15
(println (reduce + 100 []))  ;; 100

(println "=== List* ===")
(println (list* 1 2 [3 4]))  ;; (1 2 3 4)

(println "=== Bit Ops ===")
(println (bit-and 5 3))  ;; 1
(println (bit-or 5 3))   ;; 7
(println (bit-xor 5 3))  ;; 6

(println "=== Type Predicates ===")
(println (number? 42))    ;; true
(println (number? "x"))   ;; false
(println (string? "x"))   ;; true
(println (keyword? :x))   ;; true

(println "=== Nested ===")
(println (first (first [[1 2] [3 4]])))  ;; 1
(println (get (get {:a {:b 42}} :a) :b)) ;; 42

(println "=== Map nil Value ===")
(def mn {:a nil :b 2})
(println (get mn :a))           ;; nil
(println (get mn :a "default")) ;; nil (key exists)
(println (get mn :c "default")) ;; default

(println "=== Conj nil ===")
(println (conj nil 1))  ;; (1)

(println "=== Large Vec ===")
(def bv [1 2 3 4 5 6 7 8 9 10])
(println (count bv))   ;; 10
(println (first bv))   ;; 1
(println (nth bv 9))   ;; 10

(println "=== Multi-arity Closure ===")
(def mm (fn [b]
          (fn
            ([] b)
            ([x] (+ b x)))))
(def mm1 (mm 100))
(println (mm1))    ;; 100
(println (mm1 5))  ;; 105

(println "=== Variadic Closure ===")
(def mv (fn [p]
          (fn [& a] (if a (+ p (first a)) p))))
(def mv1 (mv 50))
(println (mv1))    ;; 50
(println (mv1 25)) ;; 75

(println "=== All tests complete ===")
