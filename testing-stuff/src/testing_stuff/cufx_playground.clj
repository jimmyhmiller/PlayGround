(ns testing-stuff.cufx-playground
  (:require [clojure.edn :as edn]
            [clojure.string :refer [lower-case]]))

(def schema (-> "/Users/jimmymiller/Desktop/schemas-pp.edn"
                slurp
                edn/read-string))

(defn remove-nils
  "remove pairs of key-value that has nil value from a (possibly nested) map. also transform map to nil if all of its value are nil" 
  [nm]
  (clojure.walk/postwalk 
   (fn [el]
     (if (map? el)
       (let [m (into {} (remove (comp nil? second) el))]
         (when (seq m)
           m))
       el))
   nm))


(defn recursive-filter [f m]
  (filter #(and (map? %) (f %)) (tree-seq coll? seq m)))
 
(defn distinct-by [f coll]
  (let [groups (group-by f coll)]
    (map #(first (groups %)) (distinct (map f coll)))))

(defn set-typeName [obj]
  (assoc obj :typeName (or (:typeName obj) (:name obj))))

(defn extract-enums [schema] 
  (->> schema 
       (recursive-filter #(= (:type %) :enum))
       (map set-typeName)
       (distinct-by :typeName)
       (map #(select-keys % [:typeName :documentation :enumvals]))))


(defn typeName->keyword [typeName]
  (-> typeName lower-case keyword))

(defn ->graphql-enum [{:keys [typeName documentation enumvals]}]
  {(typeName->keyword typeName)
   {:description documentation
    :values (into [] (map keyword enumvals))}})


(defn to-graphql-enums [schema]
  {:enums 
   (->> schema
        extract-enums
        (map ->graphql-enum)
        (apply merge)
        remove-nils)})


(def complex {:name "coordinate",
            :type :complex,
            :elements
            [{:name nil,
              :type :complex,
              :elements
              [{:name "latitude", :type :decimal, :multiplicity [1 1]}
               {:name "longitude",
                :type :decimal,
                :multiplicity [1 1]}],
              :multiplicity [1 1]}],
            :multiplicity [0 1]})

(extract-complex-element complex)







(defn extract-complex-element [{:keys [elements] :as type}]
  (if (not-nil? elements)
    (update type :elements 
            #(flatten 
              (map (fn [element] 
                     (if (nil? (:name element))
                       (extract-complex-element (:elements element))
                       (extract-complex-element element))) %)))
    type))

(defn one-level-elements [type]
  (-> type
      extract-complex-element
      (update-in [:elements] #(map (fn [x] (dissoc x :elements)) %))))



(one-level-elements complex)

(defn multiplicity-annotation [multi]
  (cond 
    (= multi [1 1]) 'non-null
    (= (get multi 1) :unbounded) 'list
    :else nil))

(def not-nil? (complement nil?))

(defn apply-annotation [annotation form]
  (if (not-nil? annotation)
    `(~annotation ~form)
    form))


(defn extract-complex-types [schema]
  (->> schema
       (recursive-filter #(= (:type %) :complex))
       (map set-typeName)
       (distinct-by :typeName)
       (map one-level-elements)
       (map #(update-in % [:elements] (partial map set-typeName)))
       (filter #(= (:type %) :complex))))

(defn get-graphql-type [field]
  (let [type (:type field)
        typeName (typeName->keyword (:typeName field))]
    (case type
      :string 'String
      :integer 'Int
      :boolean 'Boolean
      :decimal 'Float
      :dateTime 'String
      :enum typeName
      typeName)))

(defn ->graphql-field [{:keys [name typeName multiplicity documentation type] :as field}]
  (when (not-nil? typeName)
    (let [annotation (multiplicity-annotation multiplicity)
          type (get-graphql-type field)]
      {(typeName->keyword name)
       {:type (apply-annotation annotation type)
        :description documentation}})))

(->graphql-field {:name "test",
    :type :string,
    :multiplicity [0 1],
    :typeName "language"})





(defn ->graphql-object [{:keys [typeName documentation multiplicity elements type]}]
  (when (not-nil? typeName)
    (let [fields (remove-nils (apply merge (filter not-nil? (map ->graphql-field elements))))]
      {(typeName->keyword typeName)
       {:fields fields
        :description documentation}})))

(distinct (map :type (flatten (map :elements (extract-complex-types schema)))))


(extract-complex-types schema)

(defn to-graphql-objects [schema]
  {:objects 
   (->> (extract-complex-types schema)
        (map ->graphql-object)
        (filter not-nil?)
        (remove-nils)
        (apply merge))})




(defn to-graphql [schema]
  (merge 
   (to-graphql-objects schema)
   (to-graphql-enums schema)))

(defn pprint-to-file [coll file-name]
  (clojure.pprint/pprint coll (clojure.java.io/writer file-name)))

(to-graphql schema)

(-> schema
    to-graphql
    (pprint-to-file "/Users/jimmymiller/Desktop/graphql-schema.edn"))
