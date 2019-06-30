(ns pilot-pilot.fun
  (:require [clojure.string :as string]))


(defmacro rules [& args]
  `(quote ~args))





(defn variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn add-var [env var val]
  (assoc env var val))

(defn lookup [env var]
  (if-let [val (get env var)]
    (if (variable? val)
      (lookup env val)
      val)
    var))

(defn failed? [x]
  (= x :unify/failed))

(def unify)
(def unify-terms)

(defn unify [env lhs rhs]
  (if (map? env)
    (let [lhs (lookup env lhs)]
      (cond 
        (variable? lhs) (add-var env lhs rhs)
        :else (unify-terms env lhs rhs)))
    env))


(defmulti unify-terms
  (fn [env x y] (type x)))

(defn reducer [env [x y]]
  (unify env x yc))

(defmethod unify-terms clojure.lang.Sequential
  [env x y]
  (if (and (seqable? y)
           (= (count x) (count y)))
    (reduce reducer env (map vector x y))
    :unify/failed))

(defmethod unify-terms :default [env x y]
  (if (= x y)
    env
    :unify/failed))

(defn substitute [[env expr]]
  (if (coll? expr)
    (clojure.walk/postwalk (fn [x] (lookup env x)) expr)
    (lookup env expr)))


(defn rewrite [{:keys [left right]} expr]
  (let [unified (unify {} left expr)]
    (if (failed? unified)
      expr
      (substitute [unified right]))))



(def all-rules (atom []))


(defmacro add-rule [rule]
  `(do (swap! all-rules conj '~rule)
       :added))

(defn run-rules [expr]
  (reduce (fn [expr rule] (rewrite rule expr)) expr @all-rules))


;; Need to save these meta rules to run when adding a rule
(defmacro add-meta-rule [rule]
  `(do (swap! all-rules #(map (partial rewrite '~rule) %))
       :updated))

(add-rule {:left (+ ?x 0) :right ?x})
(add-rule {:left (+ 0 ?x) :right ?x})
(add-rule {:left (* 0 ?x) :right 0})
(add-rule {:left (* ?x 0) :right 0})

(add-rule (1 => 2))

(add-meta-rule {:left (?x => ?y)
                :right {:left ?x
                        :right ?y}})



(run-rules 1)


(macroexpand
 (quote
  (rewrite {:left (+ ?x 0)
            :right x})))


((rewrite {:left ?x
           :right (?x 2)})

 '(5 3))

((rewrite
  {:left x
   :right :thing})
 :thin)

(rules
 (0 + x => x)
 (x + 0 => x)
 (x + y => (clojure.core/+ x y))
)
