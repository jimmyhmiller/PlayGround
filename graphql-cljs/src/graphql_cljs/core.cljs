(ns graphql-cljs.core
  (:require [cljs.nodejs :as nodejs]
            [clojure.walk :as walk]
            [clojure.string :as string]))

(nodejs/enable-util-print!)

(defn replace-value-recursive [value replacement coll]
  (walk/postwalk (fn [v] (if (= value v) replacement v)) coll))

(defn map-key-recursive [key f coll]
  (walk/postwalk
   (fn [elem]
     (if (and (vector? elem)
              (= (count elem) 2)
              (= (get elem 0) key))
       (update elem 1 f)
       elem))
   coll))

(def express (nodejs/require "express"))
(def graphql (nodejs/require "graphql"))
(def graphql-http (nodejs/require "express-graphql"))


(def graphql-object-type (.-GraphQLObjectType graphql))
(def graphql-object-schema (.-GraphQLSchema graphql))
(def graphql-string (.-GraphQLString graphql))

(def type-map
  {:string (constantly graphql-string)})

(defn convert-types [f]
  (fn [type-value]
    ((get type-map type-value f) type-value)))


(defn graphql-object
  ([type-name fields]
   (graphql-object-type.
    (clj->js
     {:name (string/capitalize (name type-name))
      :fields (map-key-recursive :type (convert-types graphql-object) fields)})))
  ([{:keys [name fields] :as whole}]
   (if (nil? name)
     whole
     (graphql-object name fields))))


(defn graphql-query [schema]
  (graphql-object-schema.
   (clj->js {:query (graphql-object schema)})))


(defn graphql-server [{:keys [query & options]}]
  (graphql-http (clj->js (merge {:schema (graphql-query schema)} options))))



(def data
  (clj->js
   {:1 {:id "1", :name "Dan"},
    :2 {:id "2", :name "Marie"},
    :3 {:id "3", :name "Jessie"}}))

(def user-type
  {:name :user
   :fields
   {:id {:type :string}
    :name {:type :string}}})

(def schema
  {:name :query
   :fields
   {:user
    {:type user-type
     :args {:id {:type :string}}
     :resolve (fn [_ args]
                (aget data (.-id args)))}}})

(defn -main []
  (..
   (express)
   (use "/graphql" (graphql-server {:schema schema :pretty true}))
   (listen 3000))
  (println "GraphQL server running on http://localhost:3000/graphql"))


(set! *main-cli-fn* -main)
