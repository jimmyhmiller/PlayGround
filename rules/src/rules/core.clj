(ns rules.core
  (:require [clara.rules :refer
             [defrule mk-session insert insert! fire-rules defquery query retract!]]
            [clara.tools.inspect :as inspect]
            [clara.tools.tracing :as tracing]
            [clojure.pprint :as pprint]
            [clara.rules.engine :as eng]
            [clara.rules.compiler :as com]
            [rules.entities :refer :all])
  (:import [rules.entities Player InRange Grapple]))

(defn distance [x y]
  0)

(defn log [x]
  (println x)
  x)

(defrule player-in-range
  [?p1 <- Player (= reach ?reach)]
  [?p2 <- Player 
   (not= nil (swap! state conj ?p1))
   (= ?distance (distance ?p1 ?p2))
   
   (<= ?distance ?reach)]
  
  => (insert! (->InRange ?p1 ?p2)))



(defrule provoke-attack-of-opportunity
  [?inrange <- InRange (= x ?target) (= y ?grappler)]
  [?grap <- Grapple
   (= target ?target)
   (= grappler ?grappler)
   (= false (:improved-grapple ?grappler))]
  
  => (insert! (->AttackOpportunity ?target ?grappler)))



(def state (atom []))

@state

(defn tracing-condition [fact-binding condition]
  `(let [result# ~condition]
     (swap! rules.core/state conj {:fact ~(symbol (name fact-binding))
                                   :condition (quote ~condition)
                                   :result ~condition})
    result#))



(defn trace-rule [{:keys [lhs] :as rule}]
  (let [new-lhs (mapv #(update % :constraints
                               (fn [constraints]
                                 (concat constraints
                                         (map (partial tracing-condition (:fact-binding %)) 
                                              constraints)))) 
                      lhs)]
    (assoc rule :lhs new-lhs)))

(tracing-condition '(<= ?distance ?reach))

(map trace-rule
     (com/load-rules 'rules.core))


(defrecord TracingRule [sym]
    com/IRuleSource
    (load-rules [_]
      (map trace-rule (com/load-rules sym))))


(def p1 (->Player "Lorc" true 10))
(def p2 (->Player "Baron" true 5))


(def session
  (-> (mk-session 'rules.core)
      (inspect/with-full-logging)
      (tracing/with-tracing)
      (insert 
       (->Grapple p1 p2))
      (insert p1)
      (insert p2)
      (fire-rules)))

(:rulebase
 (eng/components session))

(tracing/get-trace session)

(:unfiltered-rule-matches
 (inspect/inspect session))

