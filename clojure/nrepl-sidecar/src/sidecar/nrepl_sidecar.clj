(ns sidecar.nrepl-sidecar
  (:require
   [clojure.data.json :as json]
   [clj-http.client :as http]
   [nrepl.middleware :refer [set-descriptor!]]
   [nrepl.misc :refer [response-for]]
   [nrepl.transport :as transport]))

(def ^:dynamic *sidecar-url* "http://localhost:8888")

(defn send-to-sidecar
  "Send data to sidecar server if it's available"
  [data]
  (try
    (http/post *sidecar-url*
               {:body (json/write-str data)
                :headers {"Content-Type" "application/json"}
                :timeout 1000
                :throw-exceptions false})
    (catch Exception _
      nil)))

(defn wrap-sidecar
  "Middleware that sends all nREPL requests and responses to a sidecar server"
  [handler]
  (fn [msg]
    (let [start-time (System/currentTimeMillis)]
      (send-to-sidecar {:type :request
                        :timestamp start-time
                        :session (:session msg)
                        :id (:id msg)
                        :op (:op msg)
                        :message msg})
      
      (let [responses (atom [])
            original-transport (:transport msg)
            capturing-transport (reify transport/Transport
                                  (recv [_] (.recv original-transport))
                                  (send [_ response]
                                    (swap! responses conj response)
                                    (send-to-sidecar {:type :response
                                                     :timestamp (System/currentTimeMillis)
                                                     :session (:session response)
                                                     :id (:id response)
                                                     :op (:op msg)
                                                     :response response})
                                    (.send original-transport response)))]
        
        (handler (assoc msg :transport capturing-transport))))))

(set-descriptor! #'wrap-sidecar
  {:requires #{}
   :expects #{}
   :handles {}})
