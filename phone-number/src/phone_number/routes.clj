(ns phone-number.routes
  (:require [compojure.core :refer :all]
            [compojure.route :as route]
            [ring.middleware.defaults :refer [wrap-defaults api-defaults]]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [ring.util.response :refer [response not-found status]]
            [ring.util.io :refer [string-input-stream]]
            [clojure.java.io :as io]
            [phone-number.data :as data]
            [phone-number.handlers :as handlers]
            [phone-number.spec :refer :all]
            [clojure.data.json :as json]))

(def file (io/resource "interview-callerid-data.csv"))

(defonce phone-numbers (atom (data/get-cleaned-data file)))

(defn app-routes [phone-numbers]
  (compojure.core/routes
   (GET "/query" [number] 
        (handlers/query-number @phone-numbers number))
   (POST "/number" {body :body}
         (handlers/add-number! phone-numbers body))
   (route/not-found {:body {:error "Route not found"}})))

(defn create-app [phone-numbers]
  (-> (app-routes phone-numbers)
      wrap-json-response
      (wrap-json-body {:keywords? true})
      (wrap-defaults api-defaults)))

(def app (create-app phone-numbers))
