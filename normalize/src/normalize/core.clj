(ns normalize.core
  (:require [clojure.spec.alpha :as s]))



(s/def ::attr string?)

(s/def ::my-entity (s/keys :req-un [::attr]))

(s/registry)

(s/form ::my-entity)


{:agent
 {:ident :id
  :relationships
  {:author_id :author}}}
