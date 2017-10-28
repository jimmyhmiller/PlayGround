(ns experiment-spec.interpret
  (:require [clojure.core.match :as core.match]))

(defn convert-match [case]
  (core.match/match [case]
                    [[sym pred]] [sym :guard pred]
                    [s-expr] [(list (into [] s-expr) :seq)]))

(defmacro match [c & cases]
  `(core.match/match ~c ~@(mapcat (fn [[k v]] [(convert-match k) v]) (partition 2 cases))))


(defn env [] (atom {'+ + '= = '* * '- - 'println println}))

(defn macro? [env n]
  (-> @env n :meta :macro))

(defn get-var [env n] 
  (let [var (@env n)]
    (get var :val var)))

(defn add-var
  ([env x arg]
   (add-var env x arg {}))
  ([env x arg meta]
   (swap! env assoc x {:val arg :meta meta})
   env))


(defmacro const [x] 2)

(const ((fn [x] 2)))


(defn eval-expr [expr env]
  (match [expr]
         [n nil?] nil
         [n boolean?]                   n
         [n number?]                    n
         [n symbol?]                    (get-var env n)
         ('quote x)                     x
         ('defmacro name [x] body)      (add-var env name 
                                                 (fn [arg] (eval-expr body (add-var env x arg)))
                                                 {:macro true})
         ('def name x)                  (add-var env name (eval-expr x env))
         ('if pred t f)                 (if (eval-expr pred env)
                                          (eval-expr t env)
                                          (eval-expr f env))
         ('do & exprs)                  (last (doall (map #(eval-expr % env) exprs)))
         ('fn [x] body)                 (fn [arg] (eval-expr body (add-var env x arg)))
         (f arg)                        (if (macro? env f)
                                          (eval-expr ((eval-expr f env) arg) env)
                                          ((eval-expr f env) (eval-expr arg env)))))

(eval-expr 
 '(do (defmacro debugging [f] 
        (f 2))
      (debugging (fn [y] y))) (env))


(comment
  (defn eval-expr [expr env]
    (match [expr]
           [n number?]     n
           [n symbol?]     (get-var env n)
           ('fn [x] body)  (fn [arg] (eval-expr body (add-var env x arg)))
           (f arg)         ((eval-expr f env) (eval-expr arg env)))))
