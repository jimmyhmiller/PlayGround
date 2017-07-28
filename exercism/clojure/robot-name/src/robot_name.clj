(ns robot-name)

(def upper-case-letters 
  (map char (concat (range 65 91))))

(defn generate-random-name []
  (str (rand-nth upper-case-letters)
       (rand-nth upper-case-letters)
       (rand-int 10)
       (rand-int 10)
       (rand-int 10)))

(defn robot []
  (atom {:name (generate-random-name)}))

(defn robot-name [robot]
  (:name @robot))

(defn reset-name [robot]
  (swap! robot assoc :name (generate-random-name)))
