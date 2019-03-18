(ns testing-stuff.matching-interpreter
  (:require [meander.match.gamma :refer [match search find]])
  (:refer-clojure :exclude [eval find]))



(def globals (atom {}))

(defn reset-globals []
  (reset! globals {})
  nil)

(defn add-global [var val]
  (swap! globals assoc var val)
  nil)


(defn lookup [env var]
  (if (or (contains? env var)
          (contains? @globals var)) 
    (get env var (get @globals var))
    (throw (ex-info "Variable not found"
                    {:env env :var var :globals @globals}))))

(defn add-var [env var value]
  (assoc env var value))






(defn eval [expr env]
  (match expr
    (pred nil? ?x) ?x
    (pred number? ?x) ?x
    (pred boolean? ?x) ?x
    (pred string? ?x) ?x
    (pred symbol? ?x) (lookup env ?x)
    (clj ?x) (if (symbol? ?x) 
               (resolve ?x)
               (clojure.core/eval ?x))
    (def ?var ?val) (add-global ?var (eval ?val env))
    ('let [?var ?val] ?body) (eval ?body
                                   (add-var env ?var (eval ?val env)))
    (if ?pred ?t ?f) (if (eval ?pred env)
                       (eval ?t env)
                       (eval ?f env))
    (fn [?x] ?body) (fn [x] (eval ?body (add-var env ?x x)))
    (fn [?x ?y] ?body) (fn [x y] (eval ?body (add-var
                                             (add-var env ?x x)
                                             ?y y)))
    (?f ?x) ((eval ?f env) (eval ?x env))
    (?f ?a ?b) ((eval ?f env) 
                (eval ?a env)
                (eval ?b env))))


(defn evalable [f g expr env]
  (match expr
    (pred nil? ?x) ?x
    (pred number? ?x) ?x
    (pred boolean? ?x) ?x
    (pred string? ?x) ?x
    (pred symbol? ?x) (lookup env ?x)
    (clj ?x) (if (symbol? ?x) 
               (resolve ?x)
               (clojure.core/eval ?x))
    (def ?var ?val) (add-global ?var (f ?val env))
    ('let [?var ?val] ?body) (f ?body
                                   (add-var env ?var (f ?val env)))
    (if ?pred ?t ?f) (if (f ?pred env)
                       (f ?t env)
                       (f ?f env))
    (fn [?x] ?body) (fn [x] (f ?body (add-var env ?x x)))
    (fn [?x ?y] ?body) (fn [x y] (f ?body (add-var
                                             (add-var env ?x x)
                                             ?y y)))
    (?f ?x) (g env (f ?f env) (f ?x env))
    (?f ?a ?b) (g env
                  (f ?f env) 
                  (f ?a env)
                  (f ?b env))))




;; This is a fun little hack. But what I need to do is separate things
;; into phases. For sure I need the expression as is and the
;; expression after substitution has occurred.
;; Once I have that, you could even ask things like what let bindings
;; result in a 0. Maybe if I just did small-step I would get all this
;; for free?

(eval 
 '(let [x (** 2 0)]
    (let [y (- x 1)]
      (let [z (* 2 (+ x x))]
        (* x (* y z)))))
 {})


(defn apply-decorated [container finder]
  (fn [env f & args]
    (when (finder `(~f ~@args))
      (swap! container conj {:expr `(~f ~@args) :env env}))
    (apply f args)))

(defn eval-decorated [container finder]
  (fn [expr env]
    (println expr)
    (when (finder expr)
      (swap! container conj {:expr expr :env env}))
    (evalable (eval-decorated container finder) 
              (apply-decorated container finder)
              expr env)))


(def values (atom []))
(def eval' (eval-decorated 
            values 
            (fn [x]
              (find x
                (_ . _ ... 0 . _ ... :as ?expr) ?expr
                _ nil))))

((fn [x]
   (find x
     (_ . _ ... 0 . _ ... :as ?expr) ?expr
     _ nil)) '(* 0 2))

(eval' '(let [x (** 2 0)]
          (let [y (- x 1)]
            (let [z (* 2 (+ x x))]
              (* x (* y z))))) {})






(defmacro eval-many [& exprs]
  `(reduce (fn [_# expr#] (eval expr# {}))
          nil
          (quote ~exprs)))

(reset-globals)

(eval-many
  ;; -----std-lib----
  (def * (clj *))
  (def ** (clj (fn [x y] (int (Math/pow x y)))))
  (def println (clj println))
  (def / ((clj comp) (clj float) (clj /)))
  (def + (clj +))
  (def - (clj -))
  (def > (clj >))
  (def = (clj =))
  ;; end------std-lib----
  (def x true)
  (** 2 0))


