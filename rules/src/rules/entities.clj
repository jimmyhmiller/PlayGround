(ns rules.entities)

(defrecord Player [name improved-grapple reach]) 

(defrecord Grapple [grappler target]) 

(defrecord AttackOpportunity [attacker target]) 

(defrecord InRange [x y])

