(ns grade-school)

(defn add [db student grade]
  (update db grade (fnil conj []) student))

(defn grade [db grade-filter]
  (get db grade-filter []))

(defn sorted [db]
  (->> db
       (map (fn [[k v]] [k (sort v)]))
       (into (sorted-map))))
