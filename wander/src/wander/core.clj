(ns wander.core
  (:require [meander.match.beta :as m :refer [match search]]
            [meander.strategy.beta :as strat]
            [meander.syntax.beta :as syntax]
            [clojure.spec.alpha :as s]
            [meander.substitute.beta :as sub]))

(comment (alter-meta! #'strat/analyze-rewrite-args #(assoc % :private false)))

(strat/analyze-rewrite-args
 '(strat/rewrite
   (... ?x) 
   ?x))


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

(defn identity-strat
  [s]
  (fn [t]
    (s t)))


(defn triples-to-map [triples]
  (let [s (strat/rewrite
           [?e (pred keyword? ?a) ?v]
           {?e [[?a ?v]]}

           [!xs ...
            {?e [!attrs ...]}
            {?e [!attrs ...]}
            . !ys ...]
           [!xs ... {?e [!attrs ...]} . !ys ...])
        transform (strat/until = (strat/choice (strat/some s) s))]
    (reduce merge (transform (into [] (sort-by first triples))))))


(triples-to-map
 '[{1 [[:active true]]}
   {1 [[:high-voltage true]]}
   {1 [[:thing 2]]}
   {2 [[:active true]]}])

(time
 (triples-to-map
  (take 300 data)))


(let [s (strat/rewrite
         (fn [!args ...]
           ?body)
         !args)]
  ((strat/innermost s)
   '(fn [x y z]
      x)))

(let [s (strat/rewrite
         (+ . !ys ... 0 . !xs ...)
         (+ . !ys ... . !xs ...))]
  ((strat/innermost s)
   '(+ 0 3 (+ 0 0 2 3 0 123 2 (+ 2 3 0) 21 23 23))))



(defn random-data [n]
  (into []
        (set
         (mapcat identity
                 (for [i (range n)]
                   (let [e (rand-int n)]
                     [[e :active tue]
                      [e :high-voltage (zero? (rand-int 2))]]))))))


(def data (random-data 300))

(count (distinct (map first data)))

(def thing (triples-to-map data))
(count (keys thing))


(triples-to-map data)

(m/search thing
  [_ ... [?e :active true]])


(m/search '[[1 :active true]
            [1 :high-voltage true]
            [1 :thing 2]
            [2 :active true]]
  [_ ... [?e !a !v] . _ ...]
  [?e !a !v]
  )


(time
 (count 
  (m/search thing
    {?e
     [_ ...
      [:active true]
      . _ ...
      [:high-voltage true]
      . _ ...]}
     [?e :dangerous true])))





(m/match '(+ 1 0 1 2 3)
    (+ ?x 0 . !xs ...)
    [?x !xs])


(macroexpand)
(m/match '(+ :A :B)
   (+ :A x)
   ?x)


(m/search '(+ 0 3 4 5 6 0)
  (and
   (+ . _ ... (pred (complement zero?) ?x) . _ ...))
  '(+ ?x))

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
    (+ ?x 0) 0
    (- ?x 0) x))

(defn eval-expr [expr env]
  (m/match expr
    (and ?x (pred number?)) ?x
    (and ?x (pred symbol?)) (env ?x)
    (fn [?arg] ?body) (fn [x] 
                        (eval-expr ?body
                         (assoc env ?arg x)))
    (?f ?x) ((eval-expr ?f env) (eval-expr ?x env))))
