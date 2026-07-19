;; clojure.walk — generic tree walking, written in the language (subagent-authored,
;; matches the real clojure.walk).

(ns clojure.walk)
(defn walk [inner outer form]
  (cond
    (list? form) (outer (apply list (map inner form)))
    (map-entry? form) (outer [(inner (key form)) (inner (val form))])
    (seq? form) (outer (doall (map inner form)))
    (record? form) (outer (reduce (fn [r x] (conj r (inner x))) form form))
    (coll? form) (outer (into (empty form) (map inner form)))
    :else (outer form)))
(defn postwalk [f form] (walk (partial postwalk f) f form))
(defn prewalk [f form] (walk (partial prewalk f) f (f form)))
(defn keywordize-keys [m]
  (let [f (fn [e] (let [k (first e) v (second e)] (if (string? k) [(keyword k) v] [k v])))
        wf (fn [x] (if (map? x) (into {} (map f x)) x))]
    (postwalk wf m)))
(defn stringify-keys [m]
  (let [f (fn [e] (let [k (first e) v (second e)] (if (keyword? k) [(name k) v] [k v])))
        wf (fn [x] (if (map? x) (into {} (map f x)) x))]
    (postwalk wf m)))
(defn prewalk-replace [smap form] (prewalk (fn [x] (if (contains? smap x) (get smap x) x)) form))
(defn postwalk-replace [smap form] (postwalk (fn [x] (if (contains? smap x) (get smap x) x)) form))
