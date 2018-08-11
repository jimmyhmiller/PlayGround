(ns testing-stuff.cufx
  (:require [clojure.spec.alpha :as s]
            [clojure.xml :as xml]
            [clojure.walk :as walk])
  (:import [java.io StringReader StringBufferInputStream])
  (:use [clojure.java.shell :only [sh]]))



(defn parse-xml [file]
  (xml/parse (StringBufferInputStream. (slurp file))))

(def validation-status (xml/parse
                         (StringBufferInputStream.
                          (slurp "schemas/ValidationStatus.xsd"))))


(defn attr [key value]
 (fn [xml] (= (key xml) value)))



(s/def :xsd/tag 
  #{:xs:schema 
    :xs:element 
    :xs:simpleType 
    :xs:annotation 
    :xs:enumeration 
    :xs:documentation
    :xs:restriction})
(s/def :xsd/attrs (s/nilable map?))
(s/def :xsd/content (s/or :string (s/tuple string?) :children (s/coll-of :xsd/element)))
(s/def :xsd/element (s/keys :req-un [:xsd/tag :xsd/attrs :xsd/content]))


(defn filter-by-attr-value [key value xml]
  (->> xml
       xml-seq
       (filter (attr key value))))

(def get-by-tag (partial filter-by-attr-value :tag))
(def get-simple-types (partial get-by-tag :xs:simpleType))
(def get-enum-values (partial get-by-tag :xs:enumeration))

(defn log-then [f]
  (fn [& args]
    (println args)
    (apply f args)))




(defn get-set-enums [simple-type]
  (->> simple-type
       get-enum-values
       (map #(-> % :attrs :value))
       set))

(->> (clojure.java.io/file "./schemas")
    file-seq
    (filter #(clojure.string/ends-with? (.getName %) ".xsd"))
    (map #(str "schemas/" (.getName %)))
    (map (fn [x]  {:name x :contents (slurp x :encoding "ISO-8859-1")}))
    (map (fn [{:keys [name contents]}] (spit name contents))))


(->> (clojure.java.io/file "./schemas")
    file-seq
    (filter #(clojure.string/ends-with? (.getName %) ".xsd"))
    (map #(str "schemas/" (.getName %)))
    (map parse-xml)
    (map get-simple-types)
    flatten
    (map get-set-enums)
    (filter (complement empty?))
    (map #(s/spec %))
    (map (fn [spec] (map first (s/exercise spec)))))



(parse-xml "schemas/AccessProfileFilt.xsd")

(defn make-enum-specs [simple-type]
  (->> simple-type
       get-set-enums
       s/spec))

(->> validation-status
     get-simple-types
     (map get-set-enums)
     (first)
     (s/spec)
     (s/exercise)
     (map first))

(s/explain :xsd/element validation-status)

validation-status
