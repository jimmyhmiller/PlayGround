(ns testing-stuff.graphql
  (:require [graphql-clj.parser :as parser])
  (:require [graphql-clj.type :as type])
  (:require [clojure.string :as string]))

(defn object [name fields]
  {:type-system-definitions  
   [{:section :type-system-definitions
     :fields (map->field fields)
     :node-type :type-definition
     :kind :OBJECT
     :type-name name}]})

(defn convert-field [[k v]]
  {:field-name (name k)
   :node-type :type-field
   :type-name (string/capitalize (name v))})

(defn map->field [coll]
  (->> coll
       (map convert-field)
       (into [])))

(def user-type
  (object "User" 
          {:name :string
           :age :int}))
