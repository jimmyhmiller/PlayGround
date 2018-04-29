(ns s3-instant-rest.core
  (:import (com.amazonaws.services.s3 AmazonS3Client)
           (com.amazonaws.services.s3.model SelectObjectContentRequest JSONOutput))
  (:require [amazonica.aws.s3 :as s3]
            [cheshire.core :as json]
            [clojure.java.data :refer [to-java from-java]]
            [clojure.string :as string]
            [honeysql.core :as sql]))


(def bucket "jimmyhmiller-bucket")


(defn put-jsonish-file [obj file-name]
  (let [json (string/join "\n" (map json/generate-string obj))
        bytes (.getBytes json "UTF-8")
        input-stream (java.io.ByteArrayInputStream. bytes)]
    (s3/put-object
     {:bucket-name bucket
      :key file-name
      :input-stream input-stream
      :metadata {:content-length (count bytes)}})))

(put-jsonish-file  [{:a 2 :b 2}
                    {:a 3}
                    {:a {:b 3}}] 
                   "test.json")

(defn build-select-criteria [file-name expression]
  {:bucketName bucket
   :expressionType "SQL"
   :key file-name
   :inputSerialization {:json {:type "DOCUMENT"}}
   :outputSerialization {:json {:recordDelimiter "\n"}}
   :expression expression})

(defmethod to-java [JSONOutput clojure.lang.APersistentMap] [clazz props]
  (doto (JSONOutput.)
    (.setRecordDelimiter (:recordDelimiter props))))

(def c (AmazonS3Client.))

(defn select-object [select-criteria]
  (let [records (->> select-criteria
                     (to-java SelectObjectContentRequest)
                     (.selectObjectContent c)
                     .getPayload
                     .getRecordsInputStream 
                     slurp)] 
    (map json/parse-string (string/split records #"\n"))))

(select-object
 (build-select-criteria
  "test.json"
  (first (sql/format {:select [:*]
                      :from [[:S3Object :s]]
                      :where [:= :s.a 3]}
                     :parameterizer :none))))




