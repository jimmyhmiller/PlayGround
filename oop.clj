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
   (clojure.test/function? obj)
   (apply obj (cons message args))
   :else
   (apply (obj message)
          (cons (partial dispatcher obj) args))))



(defmacro defclass [fn-name params body]
  `(defn ~fn-name ~params
     (partial dispatcher ~body)))



(def Union)

(defclass Insert [s n]
  (if (s :contains n)
    s
    {:isEmpty (fn [this] false)
     :contains (fn [this i] (or (= i n) (s :contains i)))
     :insert (fn [this i] (Insert this i))
     :union (fn [this s] (Union this s))}))

(defclass Empty []
  {:isEmpty (fn [this] true)
   :contains (fn [this i] false)
   :insert (fn [this i] (Insert this i))
   :union (fn [this s] s)})

(defclass Union [s1 s2]
  {:isEmpty (fn [this]  (and (s1 :isEmpty) (s2 :isEmpty)))
   :contains (fn [this i] (or (s1 :contains i) (s2 :contains i)))
   :insert (fn [this i] (Insert this i))
   :union (fn [this s] (Union this s))})

(defclass Even []
  {:isEmpty false
   :contains (fn [this i] (even? i))
   :insert (fn [this i] (Insert this i))
   :union (fn [this s] (Union this s))})

(defclass point [x y]
  {:x (fn [self] x)
   :y (fn [self] y)})


(defclass Prop [p]
  {:reduce (fn [self] self)
   :assert (fn [self a t1] (if (self :isEqual a)
                             (Known self t1)
                             self))
   :isKnown (fn [self] false)
   :isTrue (fn [self] false)
   :isFalse (fn [self] false)
   :show (fn [self] p)
   :isEqual (fn [self e] (= (self :show) (e :show)))})

(defclass Known [s t]
  {:reduce (fn [self] (Known (s :reduce) t))
   :assert (fn [self a t1] (if (self :isEqual a)
                             self
                             (Known (s :assert a t1) t)))
   :isKnown (fn [self] true)
   :isTrue (fn [self] t)
   :isFalse (fn [self] (not t))
   :show (fn [self] t)
   :isEqual (fn [self e] (s :isEqual e))})

(defclass And [p q]
  {:reduce (fn [self]
             (cond
              (p :isTrue) (q :reduce)
              (q :isTrue) (p :reduce)
              (p :isFalse) (self :assert self false)
              (q :isFalse) (self :assert self false)
              :else (And (p :reduce) (q :reduce))))
   :assert (fn [self a t1] (if (self :isEqual a)
                            (Known self t1)
                            (And (p :assert a t1) (q :assert a t1))))
   :isKnown (fn [self] (and (p :isKnown) (q :isKnown)))
   :isTrue (fn [self] (and (p :isTrue) (q :isTrue)))
   :isFalse (fn [self] (or (p :isFalse) (q :isFalse)))
   :show (fn [self] [:and (p :show) (q :show)])
   :isEqual (fn [self e] (and (p :isEqual e) (q :isEqual e)))})


(defclass Or [p q]
  {:reduce (fn [self]
             (cond
              (p :isFalse) (q :reduce)
              (q :isFalse) (p :reduce)
              :else (Or (p :reduce) (q :reduce))))
   :assert (fn [self a t1] (if (self :isEqual a)
                            (Known self t1)
                            (Or (p :assert a t1) (q :assert a t1))))
   :isKnown (fn [self] (or (p :isKnown) (q :isKnown)))
   :isTrue (fn [self] (or (p :isTrue) (q :isTrue)))
   :isFalse (fn [self] (and (p :isFalse) (q :isFalse)))
   :show (fn [self] [:or (p :show) (q :show)])
   :isEqual (fn [self e] (and (p :isEqual e) (q :isEqual e)))})

(def p (point 3 5))

(p :x)

(def e (Empty))


(obj-> (And (Or (Prop :p) (Known (Prop :q) false)) (Prop :p))
       (:assert (Prop :p) false)
       :reduce
       :show)


(obj-> (Prop :p)
       (:isEqual (Prop :p)))

(obj-> (Even)
       (:contains 2))

(obj-> (Empty)
       (:union (obj-> (Empty) (:insert 1)))
       (:insert 2)
       (:insert 2)
       (:contains 2))

