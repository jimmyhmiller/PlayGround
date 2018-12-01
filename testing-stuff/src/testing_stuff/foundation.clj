(ns testing-stuff.foundation
  (:require [clojure.spec.alpha :as s])
  (:import [com.apple.foundationdb Database]
           [com.apple.foundationdb FDB]
           [com.apple.foundationdb.tuple Tuple]
           [com.apple.foundationdb Range]
           [java.util.function Function]))

(defn as-java-fn [f]
  (reify Function
    (apply [this arg]
      (f arg))))

(def fdb (FDB/selectAPIVersion 520))
(def db (.open fdb))

(defn tuple [& args]
  (.pack (Tuple/from (object-array args))))

(defn to-vec [key-value]
  (concat (.getItems (Tuple/fromBytes (.getKey key-value)))
          (.getItems (Tuple/fromBytes (.getValue key-value)))))

(defn serialize [datum]
  (map (fn [x] (if (keyword? x) (name x) x)) datum))

(defn from-vec [datum]
  (let [ser-datum (serialize datum)]
    [(apply tuple (butlast ser-datum)) (tuple (last ser-datum))]))

(defn transact! [db datums]
  (.run db
        (as-java-fn
         (fn [tr]
           (doseq [datum datums]
             (let [[key val] (from-vec datum)]
               (.set tr key val)))))))

(defn datums->map [datums]
  (->> datums
       (map (fn [[_ attr val]]
              [(keyword attr) val]))
       (into {})))

(defn entity-by-id [db id]
  (datums->map
   (map to-vec
        @(.run db
               (as-java-fn 
                (fn [tr]
                  (.asList (.getRange tr (Range/startsWith (tuple id))))))))))

(transact! db
           [[1 :name "jimmy"]
            [1 :age 26]
            [2 :name "Falcon"]
            [2 :age 74]])

(entity-by-id db 1)
(entity-by-id db 2)






