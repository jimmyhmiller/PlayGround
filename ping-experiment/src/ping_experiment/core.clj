(ns ping-experiment.core
  (:require [clj-time.core :as time]
            [clj-time.format :as format]
            [clj-time.predicates :as pred]
            [clj-time.coerce :as coerce]
            [codax.core :as codax]
            [incanter.core :as inc-core]
            [incanter.charts :as charts])
  (:gen-class))


(def db (codax/open-database "ping-times"))

(defn dropped? [packet]
  (false? (:result packet)))

(def small-data (take 100000 (codax/get-at! db [:ping-times])))
(def big-data (filter #(< (:diff %) 2000) (codax/get-at! db [:ping-times])))


(defn dropped-window [data window]
  (->> data
       (partition window)
       (map (fn [x] (* 100 (/ (count (filter dropped? x)) window))))))

(defn dropped-window-time [data window]
  (->> data
       (partition window)
       (map (fn [x] 
              (let [start (first x)] 
                {:start (:start start) 
                 :diff (apply max (map :diff x))
                 :count (* 100 (/ (count (filter dropped? x)) window))})))))



(def bad-data (take 20000 (drop 196000 big-data)))


(def friday 
  (->> big-data
       (filter #(pred/friday? (coerce/from-long (:start %))))))



(->> bad-data
     first
     :start
     (coerce/from-long)
     (format/unparse (format/formatter "MM dd YYYY h:mm")))


(def total (count big-data))
(def dropped-packets (filter dropped? big-data))
(def not-dropped-packets (filter (complement dropped?) big-data))




(def by-hourish (dropped-window bad-data (* 60 10)))
(def by-hourish-time (dropped-window-time big-data (* 60 5)))

(inc-core/view (charts/scatter-plot (range (count by-hourish)) by-hourish))

(inc-core/view (-> (charts/histogram by-hourish :nbins 10)))


(inc-core/view (-> (charts/histogram (map :count by-hourish-time) :nbins 100)))

(inc-core/view (charts/time-series-plot (map :start bad-data) (map :diff bad-data)))
(inc-core/view (charts/time-series-plot 
                (map :start by-hourish-time) 
                (map :count by-hourish-time) :x-label "Time" :y-label "Percent pack loss (5 minute windows)"))




(codax/close-database db)




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
