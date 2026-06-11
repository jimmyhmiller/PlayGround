;; Minimal clojure.string shim — the common functions backed by registered
;; String instance methods (.toUpperCase/.toLowerCase/.trim/.startsWith/.replace)
;; and the clojure.string.Native/reverse and /split host helpers. Bare symbols
;; resolve to clojure.core (every ns auto-refers core). The reader models
;; `#"…"` regex literals as plain Strings of the pattern source, so `split`
;; passes the pattern through to the Rust-regex-backed native.
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

(defn split [s re] (clojure.string.Native/split (str s) re))

(defn split-lines [s] (split s "\r?\n"))

(defn triml [s] (clojure.string.Native/triml (str s)))
(defn trimr [s] (clojure.string.Native/trimr (str s)))

(defn capitalize [s]
  (let [t (str s)]
    (if (< (count t) 2)
      (upper-case t)
      (str (upper-case (subs t 0 1)) (lower-case (subs t 1))))))

(defn index-of
  ([s value] (let [i (.indexOf (str s) value)] (when (<= 0 i) i))))

(defn last-index-of
  ([s value] (let [i (.lastIndexOf (str s) value)] (when (<= 0 i) i))))

(defn join
  ([coll] (apply str coll))
  ([separator coll]
   (let [s (seq coll)]
     (if s
       (reduce (fn [acc x] (str acc separator x)) (str (first s)) (rest s))
       ""))))
