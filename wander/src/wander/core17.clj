(ns wander.core17
  (:require [meander.epsilon :as m]))


(defn parse [expr]
  (m/rewrite expr
    (m/symbol _ (m/re #"^\?.+") :as ?symbol)
    {:tag :logic-variable
     :symbol ?symbol}))


(defn parse [expr]
  (m/rewrite expr

    [!xs ... :as ?vector]
    {:tag :vector
     :sequence [(m/cata !xs) ...]}

    (m/symbol _ (m/re #"^\?.*") :as ?symbol)
    {:tag :logic-variable
     :symbol ?symbol}))

(parse '[?x ?y])


(defn parse [expr]
  (m/rewrite expr

    (`parse-elems [?x & ?rest])
    ((m/cata ?x) & (m/cata (`parse-elems ?rest)))

    (`parse-elems(m/seqable))
    ()
    
    [& _ :as ?vector]
    {:tag :vector
     :sequence (m/cata (`parse-elems ?vector))}

    (m/symbol _ (m/re #"^\?.*") :as ?symbol)
    {:tag :logic-variable
     :symbol ?symbol}))

(parse '[?x ?y])

(do)  


(defn interpret [expr target env]
  (m/match [expr target env]
    
    [{:tag :logic-variable :symbol ?symbol} ?target ?env]
    (if (contains? ?env ?symbol)
      (if (= ?target (get ?env ?symbol))
        ?env
        :fail)
      (assoc ?env ?symbol ?target))

    [{:tag :vector :checked nil :sequence ?sequence} ?target ?env]
    (if (vector? ?target)
      (interpret {:tag :vector :checked true :sequence ?sequence} ?target ?env)
      :fail)
    
    [{:tag :vector :sequence ()} ?target ?env]
    ?env

    [{:tag :vector :sequence (?x)} ?target ?env]
    (interpret ?x (nth ?target 0) ?env)
    
    [{:tag :vector :checked ?checked :sequence (?x & ?rest)} ?target ?env]
    (interpret {:tag :vector :checked ?checked :sequence ?rest} 
               (subvec ?target 1) 
               (interpret ?x (nth ?target 0) ?env))))


(interpret (parse '[?x ?y]) [1 2] {})



(do
  (defn compile* [expr target env]
    (m/rewrite [expr target env]

      [{:tag :logic-variable :symbol ?symbol} ?target ?env]
      (if (contains? ?env ('quote ?symbol))
        (if (= ?target (get ?env ('quote ?symbol)))
          ?env
          :fail)
        (assoc ?env ('quote ?symbol) ?target))

      [{:tag :vector :checked nil :sequence ?sequence} ?target ?env]
      (if (vector? ?target)
        (m/cata [{:tag :vector :checked true :sequence ?sequence} ?target ?env])
        :fail)
      
      [{:tag :vector :sequence ()} ?target ?env]
      ?env

      [{:tag :vector :sequence (?x)} ?target ?env]
      (m/cata [?x (nth ?target 0) ?env])
      
      (m/and [{:tag :vector :checked ?checked :sequence (?x & ?rest)} ?target ?env]
             (m/let [?env-sym (gensym "_env_")]))
      (let [?env-sym (m/cata [?x (nth ?target 0) ?env])]
        (m/cata [{:tag :vector :checked ?checked :sequence ?rest} 
                 (subvec ?target 1)
                 ?env-sym]))))


  (defmacro compile [target expr]
    (let [target_sym (gensym "target_")
          env_sym (gensym "env_")]
      `(let [~target_sym ~target
             ~env_sym {}]
         ~(compile* (parse expr) target_sym env_sym))))

  [(compile [1 2 1] [?x ?y ?x])
   (compile [1 2 2] [?x ?y ?x])])



