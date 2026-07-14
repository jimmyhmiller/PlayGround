;; clojure.zip — functional zippers, written in the language (subagent-authored,
;; faithful to the real clojure.zip).

(ns clojure.zip)
(defn zipper [branch? children make-node root]
  (with-meta [root nil]
    {:zip/branch? branch? :zip/children children :zip/make-node make-node}))
(defn seq-zip [root]
  (zipper seq? identity (fn [node children] (with-meta children (meta node))) root))
(defn vector-zip [root]
  (zipper vector? seq (fn [node children] (with-meta (vec children) (meta node))) root))
(defn node [loc] (first loc))
(defn path [loc] (:pnodes (second loc)))
(defn branch? [loc] ((:zip/branch? (meta loc)) (node loc)))
(defn children [loc]
  (if (branch? loc) ((:zip/children (meta loc)) (node loc)) (throw "called children on a leaf node")))
(defn make-node [loc n children] ((:zip/make-node (meta loc)) n children))
(defn lefts [loc] (seq (:l (second loc))))
(defn rights [loc] (:r (second loc)))
(defn down [loc]
  (when (branch? loc)
    (let [nd (first loc) p (second loc) kids (children loc)]
      (when kids
        (with-meta [(first kids)
                    {:l [] :pnodes (if p (conj (:pnodes p) nd) [nd]) :ppath p :r (next kids)}]
          (meta loc))))))
(defn up [loc]
  (let [nd (first loc) p (second loc)]
    (when p
      (let [pnodes (:pnodes p) ppath (:ppath p) l (:l p) r (:r p) changed? (:changed? p)]
        (if changed?
          (with-meta [(make-node loc (peek pnodes) (concat l (cons nd r)))
                      (and ppath (assoc ppath :changed? true))] (meta loc))
          (with-meta [(peek pnodes) ppath] (meta loc)))))))
(defn root [loc]
  (if (= :end (second loc)) (node loc)
    (let [p (up loc)] (if p (recur p) (node loc)))))
(defn right [loc]
  (let [nd (first loc) p (second loc)]
    (when (and p (seq (:r p)))
      (with-meta [(first (:r p)) (assoc p :l (conj (:l p) nd) :r (next (:r p)))] (meta loc)))))
(defn rightmost [loc]
  (let [nd (first loc) p (second loc)]
    (if (and p (seq (:r p)))
      (with-meta [(last (:r p)) (assoc p :l (apply conj (:l p) nd (butlast (:r p))) :r nil)] (meta loc))
      loc)))
(defn left [loc]
  (let [nd (first loc) p (second loc)]
    (when (and p (seq (:l p)))
      (with-meta [(peek (:l p)) (assoc p :l (pop (:l p)) :r (cons nd (:r p)))] (meta loc)))))
(defn leftmost [loc]
  (let [nd (first loc) p (second loc)]
    (if (and p (seq (:l p)))
      (with-meta [(first (:l p)) (assoc p :l [] :r (concat (rest (:l p)) [nd] (:r p)))] (meta loc))
      loc)))
(defn insert-left [loc item]
  (let [p (second loc)]
    (if (nil? p) (throw "Insert at top")
      (with-meta [(first loc) (assoc p :l (conj (:l p) item) :changed? true)] (meta loc)))))
(defn insert-right [loc item]
  (let [p (second loc)]
    (if (nil? p) (throw "Insert at top")
      (with-meta [(first loc) (assoc p :r (cons item (:r p)) :changed? true)] (meta loc)))))
(defn replace [loc n]
  (let [p (second loc)] (with-meta [n (assoc p :changed? true)] (meta loc))))
(defn edit [loc f & args] (replace loc (apply f (node loc) args)))
(defn insert-child [loc item] (replace loc (make-node loc (node loc) (cons item (children loc)))))
(defn append-child [loc item] (replace loc (make-node loc (node loc) (concat (children loc) [item]))))
(defn next [loc]
  (if (= :end (second loc)) loc
    (or (and (branch? loc) (down loc))
        (right loc)
        (loop [p loc]
          (if (up p) (or (right (up p)) (recur (up p))) [(node p) :end])))))
(defn prev [loc]
  (let [lloc (left loc)]
    (if lloc
      (loop [l lloc]
        (let [child (and (branch? l) (down l))]
          (if child (recur (rightmost child)) l)))
      (up loc))))
(defn end? [loc] (= :end (second loc)))
(defn remove [loc]
  (let [nd (first loc) p (second loc)]
    (if (nil? p) (throw "Remove at top")
      (let [l (:l p) ppath (:ppath p) pnodes (:pnodes p) rs (:r p)]
        (if (pos? (count l))
          (loop [l2 (with-meta [(peek l) (assoc p :l (pop l) :changed? true)] (meta loc))]
            (let [child (and (branch? l2) (down l2))]
              (if child (recur (rightmost child)) l2)))
          (with-meta [(make-node loc (peek pnodes) rs) (and ppath (assoc ppath :changed? true))] (meta loc)))))))
