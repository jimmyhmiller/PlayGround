(ns rules.core
  (:require [clara.rules :refer
             [defrule mk-session insert insert! fire-rules defquery query retract!]]
            [clara.tools.inspect :as inspect]
            [clara.tools.tracing :as tracing]
            [clojure.pprint :as pprint]
            [clara.rules.engine :as eng]
            [clara.rules.compiler :as com]
            [rules.entities :refer :all]
            [datascript.core :as d])
  (:import [rules.entities Player InRange Grapple]))




(defn distance [x y]
  0)

(defn log [x]
  (println x)
  x)

(defrule player-in-range
  [?p1 <- Player (= reach ?reach)]
  [?p2 <- Player
   (= ?distance (distance ?p1 ?p2))
   (<= ?distance ?reach)]
  
  => (insert! (->InRange ?p1 ?p2)))



(defrule provoke-attack-of-opportunity
  [InRange (= x ?target) (= y ?grappler)]
  [?grap <- Grapple
   (= target ?target)
   (= grappler ?grappler)
   (= false (:improved-grapple ?grappler))]
  
  => (insert! (->AttackOpportunity ?target ?grappler)))



(def state (atom []))






(def p1 (->Player "Lorc" false 10))
(def p2 (->Player "Baron" false 5))
(def p3 (->Player "Unneeded" true 5))


(def session
  (-> (mk-session 'rules.core) 
      (insert 
       (->Grapple p1 p2))
      (insert p1)
      (insert p2)
      (insert p3)
      (fire-rules)))

(:memory
 (eng/components session))


(def fact-explanations
  (into {} (:fact->explanations
            (inspect/inspect session))))


(defn get-children-facts [fact]
  (when fact
    (let [{{:keys [matches]} :explanation} fact]
      (map :fact matches))))

(defn build-tree [fact fact-explanations]
  {:fact fact
   :children (map build-tree 
                  (mapcat get-children-facts (fact-explanations fact))
                  (repeat fact-explanations))})

(build-tree fact fact-explanations)


(def fact (ffirst fact-explanations))


(mapcat get-children-facts (fact-explanations attack))


(com/load-rules 'rules.core)



(keys (inspect/inspect session))

