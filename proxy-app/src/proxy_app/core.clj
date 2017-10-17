(ns proxy-app.core
  (:import [org.littleshoot.proxy.impl DefaultHttpProxyServer]
           [org.littleshoot.proxy HttpFiltersSourceAdapter HttpFiltersAdapter])
  (:require [net.http.server :as net])
  (:gen-class))



(def filter 
  (proxy [HttpFiltersSourceAdapter] []
    (filterRequest [request ctx]
      (proxy [HttpFiltersAdapter] [request]
        (clientToProxyRequest [http]
          (clojure.pprint/pprint (net/->request request))
          (println "\n\n\n\n"))
        (serverToProxyResponse [http]
          (clojure.pprint/pprint (bean http))
          (println "\n\n\n\n")
          http)))))


(def server
  (.. (DefaultHttpProxyServer/bootstrap)
      (withFiltersSource filter)
      (withPort 3333)
      (start)))


(.stop server)
