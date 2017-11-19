(ns phone-number.routes
  (:require [compojure.core :refer :all]
            [compojure.route :as route]
            [ring.middleware.defaults :refer [wrap-defaults api-defaults]]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [ring.util.response :refer [response not-found status]]
            [ring.util.io :refer [string-input-stream]]
            [clojure.java.io :as io]
            [phone-number.handlers :as handlers]
            [clojure.data.json :as json]))

(def file (io/resource "interview-callerid-data.csv"))

(def phone-numbers (atom (data/get-cleaned-data file)))

(defroutes app-routes
  (GET "/" [] "Hello World")
  (GET "/query" [number] 
       (handlers/query-number phone-numbers number))
  (POST "/number" {body :body}
        (handlers/add-number! phone-numbers body))
  (route/not-found "Not Found"))

(def app
  (-> app-routes
      wrap-json-response
      (wrap-json-body {:keywords? true})
      (wrap-defaults api-defaults)))


(comment
  (app {:uri "/query"
        :request-method :get
        :params {:number "+1319681881"}})

  (app {:uri "/number"
        :request-method :post
        :headers {"content-type" "application/json" "accept" "application/json"} 
        :body (string-input-stream 
               (json/write-str {:number "+13196891881", 
                                :context "blah2", 
                                :name "Bast Fazio"}))}))
