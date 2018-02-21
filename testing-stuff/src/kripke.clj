(require '[clojure.set :as set])


(defn negate [belief]
  (if (= (first belief) :not)
    [(second belief)]
    [:not (first belief)]))


(def world :w1)
(def k1 (-> kripke :agents :k1))
(def worlds (:worlds kripke))
(def beliefs-in-world (k1 world))


(set/intersection (world beliefs-in-world) 
                  (into #{} (map negate (worlds world))))


(set/intersection (:w1 k1) ())

(defn possible-worlds-by-agent [worlds world agent]
  (let [beliefs-in-world (agent world)]
    (map first (filter (fn [[w props]] (empty? 
                                        (set/intersection 
                                         (w beliefs-in-world) 
                                         (into #{} (map negate props))))) 
                       worlds))))




(possible-worlds-by-agent (:worlds kripke) :w1 (-> kripke :agents :k1))

(def kripke
  {:worlds {:w1 #{[:raining]}
            :w2 #{[:not :raining]}}
   :pi (constantly true)
   :agents {:k1 {:w1 {:w1 #{[:raining]}
                      :w2 #{[:raining]}}}}})
