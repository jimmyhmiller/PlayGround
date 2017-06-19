(ns testing-stuff.faunadb
  (:require [clojure.reflect :as r]
            [clojure.pprint :refer [print-table]]
            [environ.core :refer [env]])
  (:import [com.faunadb.client FaunaClient]
           [com.faunadb.client.query Language]
           [com.faunadb.client.types Field Codec]
           [java.util.concurrent TimeUnit]))


(defn inspect [x]
  (print-table (:members (r/reflect x))))

(def admin-client 
  (-> (FaunaClient/builder)
      (.withSecret (:fauna-db env))
      (.build)))


(.get (.query admin-client (Language/CreateDatabase (Language/Obj "name" (Language/Value "test-db3")))))





(def key (.get 
          (.query admin-client 
                  (Language/CreateKey 
                   (Language/Obj 
                    "database" (Language/Database (Language/Value "test-db3")) 
                    "role" (Language/Value "server")))) 
          1 TimeUnit/SECONDS))

(def client-secret 
  (.get key (.to (Field/at (into-array String ["secret"]))
                 Codec/STRING)))


(def client 
  (-> (FaunaClient/builder)
      (.withSecret client-secret)
      (.build)))

(.query client (Language/CreateClass (Language/Obj "name" (Language/Value "posts"))))


{:language/create [{:language/class "posts"}
                   {:data 
                    {:title "test"}}]}

(.get (.query client 
              (Language/Create (Language/Class (Language/Value "posts")) 
                               (Language/Obj {"data"
                                              (Language/Obj {"title" "Test"})}))))



{:create
 {:class "posts"
  :data {:title "Test"}}}




(.get (.query client (Language/Get (Language/Ref "classes/posts/169428247593878021"))))
