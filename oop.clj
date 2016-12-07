;; Anything you type in here will be executed
;; immediately with the results shown on the
;; right.


(defmacro obj->
  [x & forms]
  (loop [x x, forms forms]
    (if forms
      (let [form (first forms)
            threaded (if (seq? form)
                       (with-meta `(~x ~(first form) ~@(next form)) (meta form))
                       (list x form))]
        (recur threaded (next forms)))
      x)))



(defn dispatcher [obj message & args]
  (cond
   (= message :methods)
   (keys obj)
   (= message :extend)
   (partial dispatcher (merge obj (first args)))
   (clojure.test/function? obj)
   (apply obj (cons message args))
   :else
   (apply (obj message)
          (cons (partial dispatcher obj) args))))

(defmacro defclass
  ([fn-name body]
   `(def ~fn-name
      (partial dispatcher ~body)))
  ([fn-name params body]
   `(defn ~fn-name ~params
      (partial dispatcher ~body))))














(defclass Unit [a]
  {:bind (fn [self k] (k a))
   :show (fn [self] (str "Success: " (a :show)))})

(defclass ErrorM [s]
  {:bind (fn [self k] self)
   :show (fn [self] (str "Error " s))})


(defclass Var [x]
  {:interp (fn [self e] (e :lookup x))})

(defclass Num' [i]
  {:show (fn [self] i)
   :add (fn [self b] (b :bind
                        (fn [b'] (Num' (+ i b')))))
   :bind (fn [self k] (k i))})


(defclass Con [i]
  {:interp (fn [self e] (Unit (Num' i)))})

(defclass Add [u v]
  {:interp (fn [self e] (obj-> u
                               (:interp e)
                               (:bind (fn [a]
                                        (obj-> v
                                               (:interp e)
                                               (:bind (fn [b] (Unit (a :add b)))))))))})
(defclass Wrong
  {:show (fn [self] "<wrong>")})

(defclass Fun [f]
  {:show (fn [self] "<function>")
   :apply (fn [self a] (f a))})

(defclass Lam [x v]
  {:interp (fn [self e] (Unit
                         (Fun
                          (fn [a]
                            (v :interp (e :add x a))))))})

(defclass App [t u]
  {:interp (fn [self e] (obj-> t
                               (:interp e)
                               (:bind (fn [f] (obj-> u
                                                     (:interp e)
                                                     (:bind (fn [a] (f :apply a))))))))})

(defclass AddEnv [e y b]
  {:lookup (fn [self x] (if (= x y)
                          (Unit b)
                          (e :lookup x)))
   :add (fn [self x a] (AddEnv self x a))})

(defclass EmptyEnv
  {:lookup (fn [self x] (ErrorM (str "unbound variable " x)))
   :add (fn [self x a] (AddEnv self x a))})




(obj-> (App (Lam "x" (Var "x")) (Add (Con 10) (Con 10)))
       (:interp EmptyEnv)
       :show)













(defclass EmptyQuery
  {:toQuery (fn [self] {})})


(defclass WhereQ [pred query]
  {:toQuery (fn [self] (merge (query :toQuery) {:where (cons pred ((query :toQuery) :where))}))
   :where (fn [self pred] (WhereQ pred self))})



(defclass Select [entity]
  {:toQuery (fn [self] {:from [entity]})
   :where (fn [self pred] (WhereQ pred self))})



(obj-> (Select :user)
       (:where :s)
       (:where :q)
       :toQuery)
























(def True)
(def False)


(defclass True
  {:if (fn [this ifs]
         (:true ifs))
   :not (fn [this] False)
   :and (fn [this a]
          (obj-> a
                 (:if {:true True
                       :false False})))
   :or (fn [this a] True)
   :show (fn [self] "True")})



(defclass False
  {:if (fn [this ifs]
         (:false ifs))
   :not (fn [this] True)
   :and (fn [this] False)
   :or (fn [this a]
         (obj-> a
                (:if {:true True
                      :false False})))
   :show (fn [self] "False")})



(obj-> True
       (:and False)
       (:if {:true true
             :false false}))



(obj-> False
       (:or True)
       (:if {:true true
             :false false}))

(obj-> False
       (:if {:true true
             :false false}))








(def Union)
(def Insert)




(defclass Empty
  {:isEmpty (fn [this] True)
   :contains (fn [this i] false)
   :insert (fn [this i] (Insert this i))
   :union (fn [this s] s)})

(defclass Insert [s n]
  (if (s :contains n)
    s
    {:isEmpty (fn [this] False)
     :contains (fn [this i] (or (= i n) (s :contains i)))
     :insert (fn [this i] (Insert this i))
     :union (fn [this s] (Union this s))}))

(defclass Union [s1 s2]
  {:isEmpty (fn [this] (and (s1 :isEmpty) (s2 :isEmpty)))
   :contains (fn [this i] (or (s1 :contains i) (s2 :contains i)))
   :insert (fn [this i] (Insert this i))
   :union (fn [this s] (Union this s))})


(defclass Even
  {:isEmpty (fn [this] False)
   :contains (fn [this i] (even? i))
   :insert (fn [this i] (Insert this i))
   :union (fn [this s] (Union this s))})


(obj-> Empty
       :methods)

(obj-> Even
       (:insert 3)
       (:contains 3))



(defclass Known [s t]
  {:reduce (fn [self] (Known (s :reduce) t))
   :assert (fn [self a t1] (if (self :isEqual a)
                             self
                             (Known (s :assert a t1) t)))
   :isKnown (fn [self] True)
   :isTrue (fn [self] t)
   :isFalse (fn [self] (not t))
   :show (fn [self] t)
   :isEqual (fn [self e] (s :isEqual e))})

(defclass Prop [p]
  {:reduce (fn [self] self)
   :assert (fn [self a t1] (if (self :isEqual a)
                             (Known self t1)
                             self))
   :isKnown (fn [self] False)
   :isTrue (fn [self] False)
   :isFalse (fn [self] False)
   :show (fn [self] p)
   :isEqual (fn [self e] (= (self :show) (e :show)))})

(defclass And [p q]
  {:reduce (fn [self]
             (cond
              (p :isTrue) (q :reduce)
              (q :isTrue) (p :reduce)
              (or (p :isFalse) (q :isFalse)) (self :assert self False)
              :else (And (p :reduce) (q :reduce))))
   :assert (fn [self a t1] (if (self :isEqual a)
                             (Known self t1)
                             (And (p :assert a t1) (q :assert a t1))))
   :isKnown (fn [self] (and (p :isKnown) (q :isKnown)))
   :isTrue (fn [self] (and (p :isTrue) (q :isTrue)))
   :isFalse (fn [self] (or (p :isFalse) (q :isFalse)))
   :show (fn [self] [:and (p :show) (q :show)])
   :isEqual (fn [self e] (= (self :show) (e :show)))})


(defclass Or [p q]
  {:reduce (fn [self]
             (cond
              (p :isFalse) (q :reduce)
              (q :isFalse) (p :reduce)
              (or (p :isTrue) (q :isTrue)) (self :assert self True)
              :else (Or (p :reduce) (q :reduce))))
   :assert (fn [self a t1] (if (self :isEqual a)
                             (Known self t1)
                             (Or (p :assert a t1) (q :assert a t1))))
   :isKnown (fn [self] (or (p :isKnown) (q :isKnown)))
   :isTrue (fn [self] (or (p :isTrue) (q :isTrue)))
   :isFalse (fn [self] (and (p :isFalse) (q :isFalse)))
   :show (fn [self] [:or (p :show) (q :show)])
   :isEqual (fn [self e] (= (self :show) (e :show)))})


(defclass If [p q]
  {:reduce (fn [self] (cond
                       (p :isTrue) q
                       (p :isFalse) (self :assert self True)
                       (q :isTrue) (self :asert self True)))
   :assert (fn [self a t1] (if (self :isEqual a)
                             (Known self t1)
                             (If (p :assert a t1) (q :assert a t1))))
   :isKnown (fn [self] (or (p :isKnown) (q :isKnown)))
   :isTrue (fn [self] (or (p :isFalse) (q :isTrue)))
   :isFalse (fn [self] (and (p :isTrue) (q :isFalse)))
   :show (fn [self] [:if (p :show) (q :show)])
   :isEqual (fn [self e] (= (self :show) (e :show)))})




(obj-> (Or (Known (Prop :q) true) (Prop :p))
       (:assert (Prop :p) false)
       :reduce
       :show)


(obj-> (If (Prop :p) (Prop :q))
       (:assert (Prop :p) true)
       :reduce
       :show)


(obj-> (Prop :p)
       (:isEqual (Prop :p)))

(obj-> Even
       (:contains 2))

(obj-> Empty
       (:union (obj-> Empty (:insert 1)))
       (:insert 2)
       (:insert 2)
       (:contains 2))



(defclass Mapper [method & args]
  {:invoke (fn [self o] (apply o (cons method args)))})


(defclass Nothing
  {:map (fn [self mapper] Nothing)
   :bind (fn [self mapper] Nothing)
   :show (fn [self] "Nothing")})

(defclass Just [o]
  {:map (fn [self mapper]
          (Just (mapper :invoke o)))
   :bind (fn [self mapper] (mapper :invoke o))
   :show (fn [self] (str "Just " (o :show)))})


(obj-> (Mapper :insert 2)
       (:invoke Empty)
       (:contains 2))

(obj-> Nothing
       (:map (Mapper :isZero))
       :show)

(obj-> (Just (Succ Zero))
       (:map (Mapper :isZero))
       :show)


(defclass Num [a]
  {:isZero (fn [self] False)
   :plus (fn [self b] (Num (+ a b)))
   :show (fn [self] a)})


(defclass Succ [n]
  {:isZero (fn [self] False)
   :inc (fn [self] (Succ self))
   :dec (fn [self] n)
   :toNum (fn [self] (Num (+ 1 (obj-> self :dec :toNum :show))))
   :plus (fn [self b] ((self :dec) :plus (b :inc)))
   :sub (fn [self b]
          (if (b :isZero)
            self
            (obj-> self :dec (:sub (b :dec)))))
   :lt (fn [self b]
         (if (b :isZero)
           false
           (obj-> self :dec (:lt (b :dec)))))
   :gt (fn [self b] (b :lt self))
   :eq (fn [self b] (and (not (self :lt b)) (not (self :gt b))))
   :show (fn [self] ((self :toNum) :show))})




(defclass Zero
  {:isZero (fn [self] True)
   :inc (fn [self] (Succ self))
   :dec (fn [self] self)
   :toNum (fn [self] (Num 0))
   :plus (fn [self b] b)
   :sub (fn [self b] b)
   :lt (fn [self b] (not (b :isZero)))
   :gt (fn [self b] (b :lt self))
   :eq (fn [self b] (and (not (self :lt b)) (not (self :gt b))))
   :show (fn [self] ((self :toNum) :show))})



(obj-> Zero
       :isZero
       :show)


(obj-> Zero
       :isZero
       (:if {:true true
             :false false}))


(defclass FromNum [a]
  (if (zero? a)
    Zero
    (Succ (FromNum (dec a)))))

(obj-> (FromNum 2)
       (:eq (FromNum 2)))


(defclass Map [mapper coll]
  {:isEmpty (fn [self] (coll :isEmpty))
   :first (fn [self] (mapper :invoke (coll :first)))
   :rest (fn [self] (Map mapper (coll :rest)))
   :cons (fn [self i] (Map mapper (coll :cons i)))
   :map (fn [self mapper1] (Map mapper1 self))
   :show (fn [self] (coll :show))})



(defclass Cons [i coll]
  {:isEmpty (fn [self] False)
   :first (fn [self] i)
   :rest (fn [self] coll)
   :cons (fn [self elem] (Cons elem self))
   :map (fn [self mapper] (Map mapper self))
   :show (fn [self] (concat [(i :show)] (coll :show)))})



(defclass Nil
  {:isEmpty (fn [self] True)
   :first (fn [self] "error")
   :rest (fn [self] "error")
   :cons (fn [self i] (Cons i self))
   :show (fn [self] [])})

(defclass Error' [message]
  {:message (fn [self] message)})


(defclass Increase [n]
  {:isEmpty (fn [self] False)
   :first (fn [self] n)
   :rest (fn [self] (Increase (n :inc)))
   :cons (fn [self i] (Cons i self))
   :map (fn [self mapper] (Map mapper self))
   :show (fn [self]  (str "[" (n :show) "..]" ))})

(defclass Decrease [n]
  (obj-> (Increase n)
         (:extend
          {:rest (fn [self]
                   (if (n :isZero)
                     (Cons Zero Nil)
                     (Decrease (n :dec))))
           :show (fn [self] (str "[" (n :show) "..0]" ))})))

(defclass Range [b e]
  (obj-> (Increase b)
         (:extend
          {:rest (fn [self]
                   (if (b :eq e)
                     Nil
                     (Range (b :inc) e)))
           :show (fn [self]
                   (str "[" (b :show) ".." (e :show) "]" ))})))

(defclass Infinite
  (Increase Zero))


(obj-> (Range (FromNum 0) (FromNum 10))
       :show)

(obj-> (Decrease (FromNum 3))
       :show)


(obj-> Zero
       :inc
       :inc
       :toNum
       :show)

(obj-> Infinite
       :rest
       :rest
       (:map (Mapper :plus (FromNum 2)))
       :first
       :show)














