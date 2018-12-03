(ns testing-stuff.foundation
  (:require [clojure.spec.alpha :as s]
            [testing-stuff.unifier :as unifier])
  (:import [com.apple.foundationdb Database]
           [com.apple.foundationdb FDB]
           [com.apple.foundationdb.tuple Tuple]
           [com.apple.foundationdb Range]
           [com.apple.foundationdb.subspace Subspace]
           [java.util.function Function]))

(defn as-java-fn [f]
  (reify Function
    (apply [this arg]
      (f arg))))

(def fdb (FDB/selectAPIVersion 520))
(def db (.open fdb))

(defn tuple [& args]
  (Tuple/from (object-array args)))

(defn tuple-packed [& args]
  (.pack (Tuple/from (object-array args))))

(defn tuple-sub [subspace & args]
  (.pack subspace (apply tuple args)))

(Tuple/fromBytes (tuple-sub eav-index 1 2))

(defn to-vec [key-value]
  (rest
   (concat (.getItems (Tuple/fromBytes (.getKey key-value)))
           (.getItems (Tuple/fromBytes (.getValue key-value))))))

(defn serialize [datum]
  (map (fn [x] (if (keyword? x) (name x) x)) datum))

(def eav-index
  (Subspace. (tuple "eav")))

(def aev-index
  (Subspace. (tuple "aev")))

(defn from-vec-eav [[e a v]]
  [(tuple-sub eav-index e a) (tuple-packed v)])

(defn from-vec-aev [[e a v]]
  [(tuple-sub aev-index a e) (tuple-packed v)])

(defn clear-subspace! [db subspace]
  (.run db
        (as-java-fn 
         (fn [tr]
           (.clear tr
                   (Range/startsWith
                    (.pack subspace)))))))

(defn transact! [db datums]
  (.run db
        (as-java-fn
         (fn [tr]
           (doseq [datum datums]
             (let [datum (serialize datum)]
               (let [[key1 val1] (from-vec-eav datum)
                     [key2 val2] (from-vec-aev datum)]
                 (.set tr key1 val1)
                 (.set tr key2 val2))))))))


(defn datums->map [datums]
  (let [id (ffirst datums)]
    (->> datums
         (map (fn [[_ attr val]]
                [(keyword attr) val]))
         (into {:id id}))))

(defn range-query [db subspace & prefix]
  (map to-vec
       @(.run db
              (as-java-fn 
               (fn [tr]
                 (.asList 
                  (.getRange tr
                             (Range/startsWith
                              (.pack subspace (apply tuple prefix))))))))))

(defn entity-by-id [db id]
  (datums->map
   (range-query db eav-index id)))

(defn tuples-for-attr [db attr]
  (map (fn [[a e v]]
         [e a v])
       (range-query db aev-index (name attr))))

(defn namify [s]
  (if (keyword? s)
    (name s)
    s))

(defn query-plan-clause [[e a v]]
  (cond
    (not (unifier/variable? e))
    (cond-> {:plan [:eav e]}
      (not (unifier/variable? a)) (update :plan conj (namify a))
      (not (unifier/variable? v)) (assoc :value v))
    (not (unifier/variable? a)) 
    (cond-> {:plan [:aev (namify a)]}
       (not (unifier/variable? v)) (assoc :value v))))


(defn query-plan [clauses]
  (set (map query-plan-clause clauses)))

(defmulti fetch-index (fn [db {:keys [plan]}] (first plan)))

(defmethod fetch-index :eav [db {:keys [plan value]}]
  (filter (fn [[_ _ v]] (or (not value) (= v value))) 
          (map (fn [[e a v]]
                 [e (keyword a) v])
               (apply range-query db eav-index (rest plan)))))

(defmethod fetch-index :aev [db {:keys [plan value]}]
  (filter (fn [[_ _ v]] (or (not value) (= v value))) 
          (map (fn [[a e v]]
                 [e (keyword a) v])
               (apply range-query db aev-index (rest plan)))))

(defn fetch [db query]
  (let [plan (query-plan query)]
    (mapcat (partial fetch-index db) plan)))

(defn q* [query db]
  (unifier/q* query (fetch db (:where query))))

(defmacro q [query db]
  `(q* (quote ~query) db))


(transact! db
           [[1 :name "jimmy"]
            [1 :age 26]
            [2 :name "Falcon"]
            [2 :age 74]
            [4 :name "stuff"]
            [10 :name "thing"]
            [10 :age 74]])


(entity-by-id db 1)
(entity-by-id db 2)

[(tuples-for-attr db :name)
 (tuples-for-attr db :age)]



(doall
 (q {:find {:name1 ?name1 
            :name ?name2 
            :age ?age}
     :where [[?e1 :age ?age]
             [?e1 :name ?name1]
             [?e2 :name ?name2]
             [?e2 :age ?age]]} db))
