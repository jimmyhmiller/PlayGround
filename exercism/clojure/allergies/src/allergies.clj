(ns allergies)

(def allergens
  {:cats 128
   :pollen 64
   :chocolate 32
   :tomatoes 16
   :strawberries 8
   :shellfish 4
   :peanuts 2
   :eggs 1})

(defn add-allergen [[n allergens] [allergen value]]
  (if (<= value n)
    [(- n value) (conj allergens allergen)]
    [n allergens]))

(defn to-size [n]
  (if (>= n 256)
    (- n 256)
    n))

(defn allergies [n]
  (->> allergens
       (reduce add-allergen [(to-size n) []])
       second
       reverse
       (into [])))

(defn allergic-to? [n allergen]
  (contains? (set (allergies n)) allergen))

