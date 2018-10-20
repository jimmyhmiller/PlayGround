(ns micro-clj.core
  (:require [org.httpkit.server :as server]
            [ring.middleware.json :as json-middleware]))

(defn interpret-response [handler]
  (fn [request]
    (let [response (handler request)]
      (if (and (map? response)
               (or (contains? response :http/body)
                   (contains? response :http/headers)
                   (contains? response :http/status)))
        {:body (:http/body response)
         :headers (:http/headers response)
         :status (:http/status response)}
        {:body response}))))



(defn middleware [handler]
  (json-middleware/wrap-json-response
   (json-middleware/wrap-json-body 
    (#'interpret-response handler)
    {:keywords? true :bigdecimals? true})))

(defn -main [main]
  (let [port (or (System/getenv "PORT") "8080")
        host (or (System/getenv "HOST") "127.0.0.1")
        handler
        (ns-resolve
         (doto (symbol (namespace (symbol main)))
           require)
         (symbol (name (symbol main))))]
    (println (str "Server started on " host ":" port))
    (server/run-server (#'middleware handler)
                       {:port (Integer/parseInt port)
                        :host host})))

(comment
  (def disconnect
    (-main "micro-clj.server/-main"))
  (disconnect))
