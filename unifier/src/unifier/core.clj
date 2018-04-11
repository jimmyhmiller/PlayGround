(ns unifier.core
  (:require [clojure.string :as string]))

(defn logic-variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn walk-var-binding [var var-map]
  (if-let [val (get var-map var)]
    (if (logic-variable? val)
      (walk-var-binding val var-map)
      val)
    var))

(defn add-equivalence [var val var-map]
  (when var-map
    (assoc var-map var val)))

(def unify)
(def unify-terms)

(defn unify [x y var-map]
  (let [x' (walk-var-binding x var-map)
        y' (walk-var-binding y var-map)]
    (cond
      (= x' y') var-map
      (logic-variable? x') (add-equivalence x' y' var-map)
      (logic-variable? y') (add-equivalence y' x' var-map)
      :else (unify-terms x' y' var-map))))

(defmulti unify-terms (fn [x y map] [(type x) (type y)]))

(defmethod unify-terms [clojure.lang.Sequential clojure.lang.Sequential] [x y var-map]
  (reduce (fn [map [x' y']] (unify x' y' map)) var-map (map vector x y)))

(defmethod unify-terms [clojure.lang.IPersistentMap clojure.lang.IPersistentMap] [x y var-map]
  (unify-terms (seq x) (seq y) var-map))

(defmethod unify-terms [java.lang.Number java.lang.Number] [x y var-map]
  (if (= x y)
    var-map
    nil))

(defmethod unify-terms :default [x y var-map]
  (if (= x y)
    var-map
    nil))



(defn unify-many 
  ([] {})
  ([x y & xys]
   (loop [x x
          y y
          xys xys
          var-map {}]
     (if x
       (recur (first xys) (second xys) (drop 2 xys) (unify x y var-map))
       var-map))))

(defmulti substitute (fn [x var-map] (type x)))

(defmethod substitute clojure.lang.Symbol [x var-map]
  (if (logic-variable? x)
    (let [var (walk-var-binding x var-map)]
      (if (logic-variable? var)
        ::failed
        (substitute var var-map)))
    x))

(defmethod substitute clojure.lang.Sequential [x var-map]
  (let [subbed (map (fn [var] (substitute var var-map)) x)]
    (if (some #(= % ::failed) subbed)
      ::failed
      subbed)))

(defmethod substitute clojure.lang.IPersistentMap [x var-map]
  (let [subbed (map (fn [var] (substitute var var-map)) x)]
    (if (some #(= % ::failed) (flatten subbed))
      ::failed
      (into {} (map (partial into []) subbed)))))

(defmethod substitute :default [x var-map]
  x)

(defn merge-maps [& maps]
  (apply unify-many (flatten (map (partial into []) maps))))

(defn match-clause [facts q var-map]
  (filter identity (map (fn [fact] (unify q fact var-map)) facts)))

(defn process-query
  ([qs facts]
   (process-query (rest qs) facts (match-clause facts (first qs) {})))
  ([qs facts var-maps]
   (if (empty? qs)
     var-maps
     (process-query (rest qs) 
            facts 
            (mapcat (fn [var-map] (match-clause facts (first qs) var-map)) var-maps)))))

(defn query [facts select qs]
  (set (map #(substitute select %) (process-query qs facts))))

(defn match-1 [coll m v]
  (if-let [map (unify m coll {})]
    (substitute v map)
    ::failed))

(defn match* [coll m v & mvs]
  (if (empty? mvs)
    (match-1 coll m v)
    (let [potential-match (match-1 coll m v)]
      (if (= potential-match ::failed)
        (apply match* (cons coll mvs))
        potential-match))))

(defmacro match [coll m v & mvs]
  (let [quoted-mvs (map (fn [x] `(quote ~x)) mvs)]
    `(match* ~coll (quote ~m) (quote ~v) ~@quoted-mvs)))

(defn gen-var []
  (symbol (str "?" (gensym))))

(substitute '?z
            (unify-many 
             '?x 'a
             '?b 'b
             '?z '[?x -> ?b]))



(def facts
  [[1 :age 26]
   [2 :age 26]
   [2 :name "thing"]
   [1 :name "jimmy"]
   [3 :age 27]
   [3 :name "person"]])

(def query1 
  '[[?e1 :age ?age]
    [?e1 :name ?name]])

(def query2
  '[[?e1 :age ?age]
    [?e2 :age ?age]
    [?e1 :name ?name1]
    [?e2 :name ?name2]])

(def query3
  '[[?e1 :age 27]
    [?e1 :name ?name]])


(query facts 
       '[?name1 ?name2]
       query2)



