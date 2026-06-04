;; Minimal clojure.string shim — the common functions backed by registered
;; String instance methods (.toUpperCase/.toLowerCase/.trim/.startsWith/.replace)
;; and the clojure.string.Native/reverse host helper. Bare symbols resolve to
;; clojure.core (every ns auto-refers core). `split` (regex) is intentionally
;; omitted until we model java.util.regex.
(ns clojure.string)

(defn upper-case [s] (.toUpperCase (str s)))
(defn lower-case [s] (.toLowerCase (str s)))
(defn trim [s] (.trim (str s)))
(defn reverse [s] (clojure.string.Native/reverse (str s)))
(defn starts-with? [s substr] (.startsWith (str s) substr))
(defn ends-with? [s substr] (let [t (str s)] (.startsWith (clojure.string.Native/reverse t)
                                                          (clojure.string.Native/reverse substr))))
(defn includes? [s substr] (>= (.indexOf (str s) substr) 0))
(defn replace [s match replacement] (.replace (str s) match replacement))
(defn blank? [s] (or (nil? s) (= "" (.trim (str s)))))

(defn join
  ([coll] (apply str coll))
  ([separator coll]
   (let [s (seq coll)]
     (if s
       (reduce (fn [acc x] (str acc separator x)) (str (first s)) (rest s))
       ""))))
