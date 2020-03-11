(ns wander.core17
  (:require [meander.epsilon :as m]))

(defn parse [expr]
  (m/rewrite expr

    (`parse-seq (m/seqable ?x & ?rest))
    ((m/cata ?x) & (m/cata (`parse-seq ?rest)))

    (`parse-seq (m/seqable))
    ()

    (& _ :as ?seq)
    {:tag :seq
     :sequence (m/cata (`parse-seq ?seq))}
    
    [& _ :as ?vector]
    {:tag :vector
     :sequence (m/cata (`parse-seq ?vector))}

    (m/symbol _ (m/re #"^?.*") :as ?symbol)
    {:tag :logic-variable
     :symbol ?symbol}

    ?x ?x))

(do
  (defn interpret [expr target env]
    (println expr target)
    (m/match [expr target env]
      
      [{:tag :vector :sequence ()} ?target ?env]
      ?env

      [{:tag :vector :sequence (?x)} ?target ?env]
      (interpret ?x (nth ?target 0) ?env)
      
      [{:tag :vector :sequence (?x & ?rest)} ?target ?env]
      (interpret {:tag :vector :sequence ?rest} (subvec ?target 1) (interpret ?x (nth ?target 0) ?env))

      [{:tag :logic-variable :symbol ?symbol} ?target ?env]
      (if (contains? ?env ?symbol)
        (if (= ?target (get ?env ?symbol))
          ?env
          :fail)
        (assoc ?env ?symbol ?target))))


  (interpret (parse '[?x ?y]) [1 2] {}))




(do
  (defn compile* [expr target env]
    (m/rewrite [expr target env]
      
      [{:tag :vector :sequence ()} ?target ?env]
      ?env

      [{:tag :vector :sequence (?x)} ?target ?env]
      (m/cata [?x (nth ?target 0) ?env])
      
      [{:tag :vector :sequence (?x & ?rest)} ?target ?env]
      (m/cata [{:tag :vector :sequence ?rest} (subvec ?target 1) (m/cata [?x (nth ?target 0) ?env])])

      [{:tag :logic-variable :symbol ?symbol} ?target ?env]
      (if (contains? ?env ('quote ?symbol))
        (if (= ?target (get ?env ('quote ?symbol)))
          ?env
          :fail)
        (assoc ?env ('quote ?symbol) ?target))))


  (defmacro compile [target expr]
    (compile* (parse expr) target {}))

  (compile [1 2] [?x ?y]))



