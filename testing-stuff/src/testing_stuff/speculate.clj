(ns testing-stuff.speculate
  (:require [clojure.spec :as s]
            [speculate.json-schema :as js]
            [speculate.ast :as ast]
            [speculate.swagger :as swagger]
            [speculate.spec :as u]))

(s/def ::color #{"red" "green" "yellow"})

(s/def ::diameter
  (u/spec
   :description "Diameter of an apple in millimetres."
   :spec pos-int?
   :maximum 300))

(s/def ::description string?)

(s/def ::apple
  (u/spec
   :description "The fruit of the apple tree."
   :spec (s/keys :req-un [::color ::diameter] :opt-un [::description])))


(s/def :success/status 200)
(s/def :success/body ::apple)

(s/def :route-params/id uuid?)
(s/def ::route-params
  (s/keys :req-un [:route-params/id]))

(s/def :query-params/other-id uuid?)
(s/def ::query-params
  (s/keys :req-un [:query-params/other-id]))

(s/def ::request
  (s/keys :req-un [::route-params ::query-params]))

(s/def ::response
  (s/or :200
        (s/keys :req-un [:success/status :success/body])))



(s/def ::request-handler
  (s/fspec
   :args (s/cat :request ::request)
   :ret ::response))

(-> (ast/parse ::request-handler)
    :form)


(js/schema (ast/parse ::apple))

(swagger/derive [["hello" {:methods {:post {:handler ::request-handler}}}]]
                {:swagger "test" :base-path nil})
