(ns wander.core
  (:require [meander.match.alpha :as m]
            [meander.strategy.alpha :as strat]
            [meander.syntax.alpha :as syntax]
            [clojure.spec.alpha :as s]))

(comment (alter-meta! @strat/analyze-rewrite-args #(assoc % :private false)))

(strat/analyze-rewrite-args
 '(strat/rewrite
   (... ?x) 
   ?x))

(s/explain '())


(syntax/parse '(... ?x))

(syntax/parse '(1 ... ?x))

(macroexpand
 '(m/match [1 2 3]
   [... ?x]
   [?x]))

(macroexpand
 (m/match [1 2 3]
  [!xs ... ?x]
  ?x))

(strat/rewrite
 (. ... ?x ...) 
 ?x)


(strat)

(let [s (strat/rewrite
         (fn [!args ...]
           ?body)
         !args)]
  ((strat/innermost s)
   '(fn [x y z]
      x)))

(let [s (strat/rewrite
         (+ ?x 0) 
         ?x)]
  ((strat/innermost s)
   '(+ 0 3 (+ 0 0))))


(let [s (strat/rewrite
         (let* [!bs !vs ..1]
           . !body ...)
         (let* [!bs !vs]
           (let* [!bs !vs ...]
             . !body ...)))]
  (s '(let* [b1 :v1, b2 :v2, b3 :v3]
        (vector b1 b2 b3))))xs

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

