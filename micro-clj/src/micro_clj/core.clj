(ns micro-clj.core
  (:require [org.httpkit.server :as server]
            [ring.middleware.format :as format-middleware]))

(defn middleware [handler]
  (format-middleware/wrap-restful-format 
   handler 
   :formats
   [:json-kw
    :edn
    :transit-json]))

(defn -main [main-ns]
  (let [handler
        (ns-resolve 
         (doto (symbol main-ns)
           require)
         '-main)
        disconnect (server/run-server (middleware handler) {:port 8080})]
    disconnect))

