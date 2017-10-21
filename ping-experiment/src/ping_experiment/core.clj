(ns ping-experiment.core
  (:require [clj-time.core :as time]
            [clj-time.coerce :as coerce]
            [codax.core :as codax])
  (:gen-class))

(def db (codax/open-database "ping-times"))

(defn dropped? [packet]
  (false? (:result packet)))


; Must be started with sudo
(defn ping
  ([domain] (ping domain 1000))
  ([domain timeout]
   (let [start (coerce/to-long (time/now))
         result (.isReachable (java.net.InetAddress/getByName domain) timeout)
         end (coerce/to-long (time/now))]
     {:start start
      :diff (- end start)
      :result result})))

(defn -main  [& args]
  (loop []
    (Thread/sleep 1000)
    (codax/update-at! db [:ping-times] conj (ping "google.com"))
    (recur)))
 
