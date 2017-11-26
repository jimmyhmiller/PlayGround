(ns phone-number.spec
  (:require [phone-number.data :as data]
            [phone-number.handlers :as handlers]
            [clojure.spec.alpha :as s]
            [orchestra.spec.test :as st]
            [ring.core.spec :refer :all]))

(s/def ::number string?)
(s/def ::name string?)
(s/def ::context string?)

(s/def ::entry (s/keys :req-un [::number ::name ::context]))

(s/fdef data/csv-data->maps
        :args (s/cat :csv-data coll?)
        :ret (s/coll-of ::entry))

(s/fdef data/read-csv-file
        :args (s/cat :file #(instance? java.net.URL %))
        :ret (s/coll-of ::entry))

(s/fdef data/->e164
        :args (s/cat :number ::number)
        :ret (s/nilable ::number))

(s/fdef data/valid-e164?
        :args (s/cat :e164 (s/nilable ::number))
        :ret boolean?)

(s/def ::phone-number-map (s/map-of ::number (s/coll-of ::entry)))

(s/fdef data/get-cleaned-data
        :args (s/cat :file #(instance? java.net.URL %))
        :ret ::phone-number-map)

(s/fdef data/valid-number-format? 
        :args (s/cat :entry ::entry)
        :ret boolean?)

(s/fdef data/duplicate-context?
        :args (s/cat :phone-numbers ::phone-number-map
                     :entry ::entry)
        :ret boolean?)

(s/fdef data/update-if-not-duplicate!
        :args (s/cat :phone-numbers ::phone-number-map
                     :entry ::entry)
        :ret ::phone-number-map)

(s/fdef data/find-entry
        :args (s/cat :phone-numbers ::phone-number-map
                     :entry ::entry)
        :ret ::entry)

(s/def ::status pos-int?)
(s/def ::body any?)
(s/def ::response (s/keys :req-un [::status ::body]))

(s/fdef handlers/error
        :args (s/cat :status-code pos-int?
                     :error-message string?)
        :ret ::response)

(s/fdef handlers/not-nil?
        :args (s/cat :val any?)
        :ret boolean?)

(s/def ::query-error-codes 
  #{:invalid-format
    :valid
    :not-found})

(s/def ::query-number-reponse
  (s/cat :error-code ::query-error-codes
         :value any?))

(s/fdef handlers/get-phone-number
        :args (s/cat :phone-numbers ::phone-number-map
                     :phone-number ::number)
        :ret ::query-number-reponse)


(s/fdef handlers/query-number-response
        :args (s/cat :query (s/spec ::query-number-reponse))
        :ret ::response)

(s/fdef handlers/query-number
        :args (s/cat :phone-numbers ::phone-number-map
                     :number ::number)
        :ret ::response)


(s/def ::add-number-error-codes 
  #{:duplicate-context :invalid-entry :updated :invalid-number})

(s/def ::add-number-reponse
  (s/cat :error-code ::add-number-error-codes
         :value any?))

(s/fdef handlers/attempt-update!
        :args (s/cat :phone-numbers #(instance? clojure.lang.Atom %)
                     :entry ::entry)
        :ret ::add-number-reponse)

(s/fdef handlers/add-phone-number!
        :args (s/cat :phone-numbers #(instance? clojure.lang.Atom %)
                     :entry ::entry)
        :ret ::add-number-reponse)

(s/fdef handlers/add-number-response
        :args (s/cat :add-response (s/spec ::add-number-reponse))
        :ret ::response)


(s/fdef handlers/add-number!
        :args (s/cat :phone-numbers #(instance? clojure.lang.Atom %)
                     :entry ::entry)
        :ret ::response)


