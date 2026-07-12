//! `clojure.set` — pure library code over our sets. Bundled + loaded like core.

pub const CLOJURE_SET: &str = r##"
(ns clojure.set)
(defn union [& sets] (reduce (fn [a s] (into a s)) #{} sets))
(defn intersection [s1 & sets]
  (into #{} (filter (fn [x] (every? (fn [s] (contains? s x)) sets)) s1)))
(defn difference [s1 & sets]
  (into #{} (filter (fn [x] (not (some (fn [s] (contains? s x)) sets))) s1)))
(defn subset? [a b] (every? (fn [x] (contains? b x)) a))
(defn superset? [a b] (every? (fn [x] (contains? a x)) b))
(defn select [pred xset] (into #{} (filter pred xset)))
(defn project [rel ks] (into #{} (map (fn [m] (select-keys m ks)) rel)))
(defn rename-keys [m kmap]
  (reduce (fn [acc e] (let [k (first e) v (second e)]
                        (if (contains? m k) (assoc (dissoc acc k) v (get m k)) acc)))
          m kmap))
"##;
