(ns experiment-spec.interpret
  (:require [clojure.core.match :as core.match]))

(defn convert-match [case]
  (core.match/match [case]
                    [else :guard keyword?] else
                    [[sym pred]] [sym :guard pred]
                    [s-expr] [(list (into [] s-expr) :seq)]))

(defmacro match [c & cases]
  `(core.match/match ~c ~@(mapcat (fn [[k v]] [(convert-match k) v]) (partition 2 cases))))


(defn env 
  ([] (env {}))
  ([extend] (atom (merge {'+ + '= = '* * '- - 'println println 'inc inc} extend))))

(defn macro? [env n]
  (and (symbol? n)
       (-> @env n :meta :macro)))

(defn get-var [env n] 
  (let [var (@env n)]
    (get var :val var)))

(defn add-var
  ([env x arg]
   (add-var env x arg {}))
  ([env x arg meta]
   (swap! env assoc x {:val arg :meta meta})
   env))

(defn add-vars [env names vals]
  (reduce 
   (fn [env [name val]] 
     (add-var env name val))
   env 
   (map vector names vals)))

(defmacro const [f] `(~f 2))

(defmacro let-it [arg body]
  `((fn [~(first arg)] ~body) ~(second arg)))


(def eval-expr)
(def macro-expansion)

(defn expand-form [env expr]
  (match [expr]
         ('unquote x) (eval-expr x env)
         :else expr))

(defn macro-expansion [expr env]
  (clojure.walk/postwalk (partial expand-form env) expr))


(defn macro-pure? [env n]
  (and (symbol? n)
       (-> env n :meta :macro)))

(def eval-expr-pure)

(defn expand-form-pure [env expr]
  (match [expr]
         ('unquote x) (first (eval-expr-pure x env))
         :else expr))

(defn macro-expansion-pure [expr env]
  (clojure.walk/postwalk (partial expand-form-pure env) expr))


(defn get-var-pure [env sym]
  (get-in env [sym :val]))

(defn add-var-pure
  ([env x arg]
   (add-var-pure env x arg {}))
  ([env x arg meta]
   (assoc env x {:val arg :meta meta})))


(defn add-vars-pure [env names vals]
  (reduce 
   (fn [env [name val]] 
     (add-var-pure env name val))
   env 
   (map vector names vals)))


(defn eval-expr-pure [expr env]
  (match [expr]
         [n number?]           [n env]
         [n symbol?]           [(get-var-pure env n) env]
         [n nil?]              [nil env]
         [n boolean?]          [n env]
         ('quote x)            [x env]
         ('if pred t f)        [(if (first (eval-expr-pure pred env))
                                  (first (eval-expr-pure t env))
                                  (first (eval-expr-pure f env))) env]
         ('def name x)         (let [[val _] (eval-expr-pure x env)]
                                 [val (add-var-pure env name val)])
         ('do & exprs)         (reduce (fn [[_ env] expr]                                       
                                      (eval-expr-pure expr env)) 
                                    [nil env] exprs)
         ('defmacro name
          [& names] body)     [name (add-var-pure 
                                     env name 
                                     (fn [& vals]
                                       (first 
                                        (eval-expr-pure
                                         (macro-expansion-pure 
                                          body
                                          (add-vars-pure env names vals)) env)))
                                     {:macro true})]
         ('fn [& names] body) [(fn [& vals] 
                                 (first 
                                  (eval-expr-pure 
                                   body 
                                   (add-vars-pure env names vals)))) env]
         (f & args)           (if (macro-pure? env f)
                                (eval-expr-pure 
                                 (apply (first (eval-expr-pure f env)) args) env)
                                [(apply (first (eval-expr-pure f env)) 
                                         (map #(first (eval-expr-pure % env)) args)) env]) env))

(defn env-pure 
  ([] (env-pure {}))
  ([extend] (add-vars-pure 
             extend 
             ['+ '= '* '- 'println 'inc] 
             [+ = * - println inc])))


(def core-library
  '(do
     (defmacro defn [name args body] 
       (quote (def (unquote name) (fn (unquote args) (unquote body)))))))


(eval-expr-pure
 '(do 
    (defn f [x y] (+ x y))
    (f 1 2)) 
 (last (eval-expr-pure core-library (env-pure))))



(defn eval-expr [expr env]
  (match [expr]
         [n nil?]                       nil
         [n boolean?]                   n
         [n number?]                    n
         [n symbol?]                    (get-var env n)
         ('quote x)                     x
         ('defmacro name 
          [& names] body)               (add-var env name 
                                                       (fn [& vals] 
                                                         (eval-expr 
                                                          (macro-expansion body 
                                                                           (add-vars env names vals)) env))
                                                 {:macro true})
         ('def name x)                  (add-var env name (eval-expr x env))
         ('if pred t f)                 (if (eval-expr pred env)
                                          (eval-expr t env)
                                          (eval-expr f env))
         ('do & exprs)                  (last (doall (map #(eval-expr % env) exprs)))
         ('fn [& names] body)          (fn [& vals] (eval-expr body (add-vars env names vals)))
         (f & args)                       (if (macro? env f)
                                            (eval-expr (apply (eval-expr f env) args) env)
                                            (apply (eval-expr f env) (map #(eval-expr % env) args)))))





(def my-env (env))

(def core-library
  '(do
     (defmacro defn [name args body] 
       (quote (def (unquote name) (fn (unquote args) (unquote body)))))))

(eval-expr core-library my-env)

(eval-expr 
 '(do 
     (defn f [x y] (+ x y))
     (f 2 3)) my-env)
    
(defn eval-expr [expr env]
  (match [expr]
         [n symbol?]     (env n)
         ('fn [x] body)  (fn [arg] (eval-expr body (assoc env x arg)))
         (f arg)         ((eval-expr f env) (eval-expr arg env))))

(eval-expr '(fn [x] (fn [x] x)) {})


(comment)
(defn eval-expr [expr env]
  (match [expr]
         [n number?]     n
         [n symbol?]     (get-var env n)
         ('fn [x] body)  (fn [arg] (eval-expr body (add-var env x arg)))
         (f arg)         ((eval-expr f env) (eval-expr arg env))))
