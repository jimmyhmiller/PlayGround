(ns ping-experiment.core
  (:require [clj-time.core :as time]
            [clj-time.coerce :as coerce]
            [codax.core :as codax])
  (:gen-class))

(def db (codax/open-database "ping-times"))

; Must be started with sudo
(defn ping 
  ([domain] (ping domain 1000))
  ([domain timeout]
   (let [start (coerce/to-long (time/now))
         result (.isReachable (java.net.InetAddress/getByName domain) timeout)
         end (coerce/to-long (time/now))]
     {:start start
      :end end
      :diff (- end start)
      :result result})))

(defn -main  [& args]
  (loop []
    (codax/update-at! db [:ping-times] conj (ping "google.com"))
    (recur)))
  
