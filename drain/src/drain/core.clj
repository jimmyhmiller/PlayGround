(ns drain.core
  (:require [org.httpkit.server :as server]))



(def t #(println %))

(tap> {:test [1 2 3]})

(def channels (atom #{}))

(defn socket-tap []
  (let [connect! (fn [channel]
                  (swap! channels conj channel))
        handler (fn [request]
                  (server/with-channel request channel
                    (connect! channel)
                    (server/on-close channel (fn [status] 
                                               (println "channel closed: " status)
                                               (swap! channels #(remove #{channel} %))))
                    (server/on-receive channel (fn [data] ))))]
    (server/run-server handler {:port 9091})
    (fn [data]
      (doseq [channel @channels]
        (server/send! channel (str data))))))


(add-tap (socket-tap))
