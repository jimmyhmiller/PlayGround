(ns testing-stuff.stripe
  (:require [clj-http.client :as client]
            [clojure.java.data :refer [from-java]])
  (:import [com.stripe.model Customer]
           [com.stripe Stripe]))


(set! (. Stripe apiKey) "")
(:data (from-java (Customer/list {"limit" 3})))

(:data (:body (client/get "https://api.stripe.com/v1/customers?limit=3" 
                          {:basic-auth ["" ""] :as :json})))
