(ns pathfinder.core
  (:require [clojure.spec.alpha :as s]))


(s/def ::strength pos-int?)
(s/def ::dexterity pos-int?)
(s/def ::constitution pos-int?)
(s/def ::intelligence pos-int?)
(s/def ::wisdom pos-int?)
(s/def ::charisma pos-int?)

(s/def ::ability #{::strength
                   ::dexterity
                   ::constitution
                   ::intelligence
                   ::wisdom
                   ::charism})

(s/def ::ability-scores
  (s/keys :req [::strength
                ::dexterity
                ::constitution
                ::intelligence
                ::wisdom
                ::charisma]))

(s/def ::score-command #{:increment :decrement})

(s/def ::ability-modification (s/tuple ::score-command ::ability))


(defn cost-for-point-change [ability-scores [command ability]]
  (let [modifier {:increment  (fn [lookup score] (get lookup score 0))
                  :decrement (fn [lookup score] (- (get lookup (dec score) 0)))}]
    ((modifier command)
     {7 2
      8 1
      9 1
      10 1
      11 1
      12 1
      13 2
      14 2
      15 2
      16 3
      17 4} 
     (ability ability-scores))))



(defn apply-change [ability-scores [command ability]]
  (let [cost (cost-for-point-change ability-scores [command ability])
        modifiers {:increment inc
                   :decrement dec}]
    (if (not (zero? cost))
      (update ability-scores ability (modifiers command))
      ability-scores)))

(reduce (fn [[total stats] modifier] 
          [(+ (cost-for-point-change stats modifier) total)
           (apply-change stats modifier)]) 
        [0 initial-stats]
        (map first (s/exercise ::ability-modification 200)))

(apply-change {::intelligence 18}
              [:decrement ::intelligence])

(def initial-stats 
  {::strength 10
   ::dexterity 10
   ::constitution 10
   ::intelligence 10
   ::wisdom 10
   ::charism 10})





