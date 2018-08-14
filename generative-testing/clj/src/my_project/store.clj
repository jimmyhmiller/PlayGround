(ns my-project.store
  (:require [clojure.java.io :as io]
            [amazonica.aws.s3 :as s3])
  (:refer-clojure :exclude [read]))

(defprotocol Store
  (read [this path])
  (write [this path content])
  (exists? [this path]))

(defn in-memory-store []
  (let [data (atom {})]
    (reify
      Store 
      (read [_ path]
        (get @data path)) 
      (write [_ path content] 
        (swap! data assoc path content)) 
      (exists? [_ path] 
        (contains? @data path)))))

(defn file-store []
  (reify
    Store
    (read [_ path] 
      (slurp path)) 
    (write [_ path content] 
      (spit path content)) 
    (exists? [_ path] 
      (.exists (io/file path)))))

(defn s3-store [bucket]
  (reify
    Store
    (read [_ path] (-> (s3/get-object bucket path) :input-stream slurp))
    (write [_ path content] (s3/put-object bucket path content))
    (exists? [_ path] (s3/does-object-exist bucket path))))



(comment
  (s3/list-buckets)



  (def my-store (in-memory-store))
  (def my-file-store (file-store))
  (def my-s3-store (s3-store "a-bucket-that-i-dont-care-about"))

  (write my-file-store "/tmp/thing.txt" "stuff")
  (read my-file-store "/tmp/thing.txt")
  (exists? my-file-store "/tmp/thing.txt")

  (write my-store :a :b)
  (exists? my-store :a)
  (read my-store :a)

  (write my-s3-store "/tmp/thing.txt" "stuff")
  (read my-s3-store "/tmp/thing.txt")
  (exists? my-s3-store "/tmp/thing.txt"))

