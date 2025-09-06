(ns ui-agent.server
  (:require
   [ring.adapter.jetty :as jetty]
   [ring.util.response :as response]
   [cheshire.core :as json]
   [clojure.string :as str]
   [ui-agent.agent :as agent]
   [ui-agent.core :as core]))

(def *server (atom nil))

(defn parse-json-body [request]
  (when-let [body (:body request)]
    (try
      (json/parse-string (slurp body) true)
      (catch Exception e
        (println "Error parsing JSON:" (.getMessage e))
        nil))))

(defn handler [request]
  (let [{:keys [request-method uri]} request]
    (cond
      ;; Health check
      (and (= request-method :get) (= uri "/health"))
      {:status 200
       :headers {"content-type" "application/json"}
       :body (json/generate-string {:status "ok" :timestamp (System/currentTimeMillis)})}
      
      ;; Message endpoint
      (and (= request-method :post) (= uri "/message"))
      (if-let [json-body (parse-json-body request)]
        (try
          (let [message (:message json-body)
                metadata (dissoc json-body :message)]
            ;; Check for debug/bypass commands
            (cond
              (= message "DEBUG_ERRORS")
              (do
                (println "=== DEBUG ERRORS COMMAND ===")
                (future (core/debug-draw-errors!))
                {:status 200
                 :headers {"content-type" "application/json"}
                 :body (json/generate-string {:status "debug"
                                              :message "Draw errors printed to console"})})
              
              (= message "SHOW_ERRORS")
              (let [errors (core/get-and-clear-draw-errors!)]
                (println "=== SHOW ERRORS COMMAND ===")
                (println "Errors:" errors)
                {:status 200
                 :headers {"content-type" "application/json"}
                 :body (json/generate-string {:status "debug"
                                              :errors errors
                                              :error-count (count errors)})})
              
              (= message "TEST_DRAW")
              (do
                (println "=== TEST DRAW COMMAND ===")
                (future 
                  (core/on-ui
                    (core/add-draw-fn! 
                      (fn [canvas]
                        (let [paint (doto (io.github.humbleui.skija.Paint.) 
                                      (.setColor (core/color 0xFFFF0000)))]
                          (.drawCircle canvas 100 100 50 paint))))))
                {:status 200
                 :headers {"content-type" "application/json"}
                 :body (json/generate-string {:status "debug"
                                              :message "Test circle drawn"})})
              
              :else
              ;; Normal message processing
              (do
                (future 
                  (try
                    (println (str "Processing message: " message))
                    (agent/process-message message metadata)
                    (println "Message processed successfully!")
                    (catch Exception e
                      (println "Error processing message:" (.getMessage e))
                      (.printStackTrace e))))
                ;; Return immediate response
                {:status 202
                 :headers {"content-type" "application/json"}
                 :body (json/generate-string {:status "accepted"
                                              :message "Message received and being processed"
                                              :received-message message
                                              :timestamp (System/currentTimeMillis)})})))
          (catch Exception e
            (println "Error parsing message:" (.getMessage e))
            {:status 500
             :headers {"content-type" "application/json"}
             :body (json/generate-string {:error "Internal server error" 
                                          :details (.getMessage e)})}))
        {:status 400
         :headers {"content-type" "application/json"}
         :body (json/generate-string {:error "Invalid JSON body"})})
      
      ;; Not found
      :else
      {:status 404
       :headers {"content-type" "application/json"}
       :body (json/generate-string {:error "Not found"})})))

(defn start-server! 
  ([] (start-server! 8080))
  ([port]
   (when @*server
     (println "Stopping existing server...")
     (.stop @*server))
   (println (str "Starting HTTP server on port " port "..."))
   (reset! *server (jetty/run-jetty handler {:port port :join? false}))
   (println (str "Server started on http://localhost:" port))
   @*server))

(defn stop-server! []
  (when @*server
    (println "Stopping HTTP server...")
    (.stop @*server)
    (reset! *server nil)
    (println "Server stopped.")))

(comment
  (start-server! 8080)
  (stop-server!))