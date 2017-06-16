(ns testing-stuff.datomic
  (:require  [datomic.api :as d]))


(def db-uri "datomic:mem://hello")

(d/create-database db-uri)

(def db-uri "datomic:dev://0.0.0.0:4334/hello")
(d/create-database db-uri)


(def conn (d/connect db-uri))

@(d/transact conn [{:db/doc "Hello world"}])
