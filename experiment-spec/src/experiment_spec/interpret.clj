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
    

(comment
  (defn eval-expr [expr env]
    (match [expr]
           [n number?]     n
           [n symbol?]     (get-var env n)
           ('fn [x] body)  (fn [arg] (eval-expr body (add-var env x arg)))
           (f arg)         ((eval-expr f env) (eval-expr arg env)))))
