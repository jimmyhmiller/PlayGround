(ns rules.core
  (:require [clara.rules :refer
             [defrule mk-session insert insert! fire-rules defquery query retract!]]
            [clara.tools.inspect :as inspect]
            [clojure.pprint :as pprint]
            [rules.entities :refer :all])
  (:import [rules.entities Player InRange Grapple]))

(defn distance [x y]
  0)

(defrule player-in-range
  [?p1 <- Player (= reach ?reach)]
  [?p2 <- Player 
   (= ?distance (distance ?p1 p2))
   (<= ?distance ?reach)]
  
  => (insert! (->InRange ?p1 ?p2)))

(defrule provoke-attack-of-opportunity
  [InRange (= x ?target) (= y ?grappler)]
  [Grapple
   (= target ?target)
   (= grappler ?grappler)
   (= false (:improved-grapple ?grappler))]
  
  => (insert! (->AttackOpportunity ?target ?grappler)))



(def p1 (->Player "Lorc" false 10))
(def p2 (->Player "Baron" true 5))

(map first
     (-> (mk-session)
         (insert 
          (->Grapple p1 p2))
         (insert p1)
         (insert p2)
         (fire-rules)
         (inspect/explain-activations)))

