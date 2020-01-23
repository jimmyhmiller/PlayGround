(ns wander.core14
  (:require [meander.epsilon :as m]))


(def M)

(def T)

(defn M [expr]
  (m/match expr
    (fn [?x] ?expr)
    (let [k (gensym)]
      (m/subst
        (fn [?x k] (m/app T ?expr k))))
    (m/pred symbol?) expr))


(defn T [expr cont]
  (m/match expr
    (fn & _) (list cont (M expr))
    (m/pred symbol?) (list cont (M expr))
    (?f ?x)
    (let [?g (gensym)
          ?h (gensym)]
      (m/subst (m/app T ?f
                      (fn [?g] 
                        (m/app T ?x (fn [?h]
                                      (?g ?h ~cont)))))))))

(M '(fn [x] x))

(T '(g a) :halt)


(def T-k)
(def T-c)
(def T*-k)


(defn primitive? [symbol]
  (println symbol)
  (and (symbol? symbol)
       (resolve symbol)
       (= 'clojure.core
          (.getName
           (:ns
            (meta (resolve symbol)))))))

(defn atom? [expr]
  (or (symbol? expr)
      (number? expr)
      (keyword? expr)
      (string? expr)
      (boolean? expr)
      (vector? expr)
      (map? expr)))


(defn T-k [expr k]
  (m/match expr
    
    (if ?pred ?t ?f)
    (let [rv (gensym)
          cont `(fn [~rv] ~(k rv))]
      (T-k ?pred (fn [aexpr]
                   `(if ~aexpr
                      ~(T-c ?t cont)
                      ~(T-c ?f cont)))))

    (let [!ks !vs ...] ?expr)
    (m/subst 
      (let [!ks (m/app M !vs) ...]
        (m/app T-k ?expr ~k)))

    (& _)
    (let [rv (gensym)
          cont `(fn [~rv] ~(k rv))]
      (T-c expr cont))

    (throw ?x) expr
    
    (quote ?x) expr

    (m/pred atom?) (k (M expr))
    ?x (throw (ex-info "not matched t-k" {:?x ?x}))))




(defn T-c [expr c]
  (m/match expr
    (if ?pred ?t ?f)
    (let [k (gensym)]
      `((fn [~k]
          ~(T-k ?pred (fn [aexpr]
                        `(if ~aexpr
                           ~(T-c ?t k)
                           ~(T-c ?f k)))))
        ~c))

    (let [!ks !vs ...] ?expr)
    (m/subst 
      (let [!ks (m/app M !vs) ...]
        (m/app T-c ?expr ~c)))

    ((m/pred primitive? ?p) & ?args)
    (T*-k ?args
          (fn [es]
            `((cps ~?p) ~@es ~c)))

    (?f & ?args)
    (T-k ?f (fn [f']
              (T*-k ?args (fn [es]
                            `(~f' ~@es ~c)))))

    (throw ?x) expr
    
    (quote ?x) expr

    (m/pred atom?) `(~c ~(M expr))))



(defn T*-k [exprs k]
  (m/match exprs
    () (k '())
    (?expr & ?exprs)
    (T-k ?expr (fn [hd]
                 (T*-k ?exprs (fn [tl]
                                (k (cons hd tl))))))))


(defn M [expr]
  (m/match expr
    (fn [?x] ?expr)
    (let [k (gensym)]
      (m/subst
        (fn [?x ~k] ~(T-c ?expr k))))
    (m/pred atom?) expr
    ?x (throw (ex-info "failed to match" {:?x ?x}))))

(defn cps [f]
  (fn [& args]
    (m/match args
      (!xs ... . ?k)
      (?k (apply f !xs)))))


(M '(fn [x] x))

(T-c '(g a) :halt)

(T-c '(let [f (fn [n]
               (if (= n 0)
                 1
                 (* n (f (- n 1)))))]
       (f 5))
     :halt)


;; http://matt.might.net/articles/cps-conversion/


(T-c
 '(let
      [x [1 2 3]]
    (let
        [ret__14025__auto__
         (if (vector? x)
           (if (= (count x) 3)
             (let
                 [x_nth_0__
                  (nth x 0)
                  x_nth_1__
                  (nth x 1)
                  x_nth_2__
                  (nth x 2)]
               (let
                   [?x x_nth_0__]
                 (let [?y x_nth_1__] (let [?z x_nth_2__] [?x ?y ?z]))))
             meander.match.runtime.epsilon/FAIL)
           meander.match.runtime.epsilon/FAIL)]
      (if (meander.match.runtime.epsilon/fail? ret__14025__auto__)
        (throw (ex-info "non exhaustive pattern match" '{}))
        ret__14025__auto__)))
 (fn [x] x))
