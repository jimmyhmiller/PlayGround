(ns jmx-clojure.core
  (:require [clojure.java.jmx :as jmx]
            [live-chart :as live]))


(def things (atom []))

(defn get-used-heap []
  (-> (jmx/mbean "java.lang:type=Memory")
      :HeapMemoryUsage 
      :used))

(->> "*:*"
     jmx/mbean-names
     seq
     (map #(.toString %))
     (map (fn [name] [name (jmx/mbean name)])))

(dotimes [n 1000000]
  (swap! things conj {:x (rand)}))

(live/show (live/time-chart [get-used-heap]))

(reset! things [])

(jmx/invoke "java.lang:type=Memory" :gc)


