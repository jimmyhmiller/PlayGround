(ns strain)

(defn retain [pred? coll]
  (keep #(if (pred? %) %) coll))

(defn discard [pred? coll]
  (keep #(if (not (pred? %)) %) coll))
