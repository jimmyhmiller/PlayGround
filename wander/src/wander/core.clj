(ns wander.core
  (:require [meander.match.alpha :as m]))


(m/match [1 2 3]
  [1 2 3] 1
  [3 2 1] 2)


(defn simplify-math [expr]
  (m/search expr
    (+ x 0) 0
    (- x 0) x))

(defn eval-expr [expr env]
  (m/match expr
    (and ?x (pred number?)) ?x
    (and ?x (pred symbol?)) (env ?x)
    (fn [?arg] ?body) (fn [x] 
                        (eval-expr ?body
                         (assoc env ?arg x)))
    (?f ?x) ((eval-expr ?f env) (eval-expr ?x env))))

