(ns rules.entities)

(defrecord Player [cmb cmd improved-grapple position reach]) 

(defrecord Grapple [grappler target]) 

(defrecord AttackOpportunity [attacker target]) 

(defrecord InRange [x y])

