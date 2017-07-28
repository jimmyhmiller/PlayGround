(ns grains)

(defn power-2 [n]
  (reduce *' (repeat n 2)))

(defn square [num]
   (power-2 (dec num)))

(defn total []
  (->> (range 1 65)
       (map square)
       (reduce +)))
