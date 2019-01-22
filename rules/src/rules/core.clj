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
  (:import [rules.entities Player InRange Grapple AttackOpportunity]))



(defn explain! [metadata entity]
  (insert! (with-meta entity metadata)))


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

  => (explain! {:p1 ?p1 :p2 ?p2}
               (->InRange ?p1 ?p2)))



(defrule provoke-attack-of-opportunity
  [?range <- InRange (= x ?target) (= y ?grappler)]
  [?grapple <- Grapple
   (= target ?target)
   (= grappler ?grappler)
   (= false (:improved-grapple ?grappler))]

  => (explain!
      {:grapple ?grapple}
      (->AttackOpportunity ?target ?grappler)))


(def p1 (->Player "Lorc" false 10))
(def p2 (->Player "Baron" false 5))


(def session
  (-> (mk-session 'rules.core)
      (insert
       (->Grapple p1 p2))
      (insert p1)
      (insert p2)
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


(map meta
     (map first fact-explanations))


(def fact (ffirst fact-explanations))

(build-tree fact fact-explanations)

(defmulti build-explanation type)

(defmethod build-explanation AttackOpportunity [{:keys [target attacker] :as entity}]
  {:type :attack-of-opportunty
   :value
   (cond
     (contains? (meta entity) :grapple)
     (format
      "%s has an Attack of Opportunity on %s because %s is trying to grapple and doesn't have improved grapple."
      (:name attacker)
      (:name target)
      (:name target))
     :else
     (format "%s has an Attack of Opportunity on %s"
             (:name attacker)
             (:name target)))})

(defmethod build-explanation :default [e]
  nil)

(filter identity
        (map build-explanation (map first fact-explanations)))


