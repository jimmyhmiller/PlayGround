(ns blocking-queue.core
  (:require [clj-time.core :refer [from-now minutes now seconds]]
            [clj-time.coerce :refer [to-long]])
  (:import [java.util.concurrent Delayed TimeUnit DelayQueue]
           [java.lang Comparable]
           [java.util Date]))

(def delays 
  {0 30
   30 60
   60 120
   120 240
   240 480})


(def service (atom {:up? true}))

(defrecord Delay [data expiration-date delay]
  Comparable
  (compareTo [this other-delay]
    (- expiration-date (.expiration-date other-delay)))
  Delayed
  (getDelay [this unit]
    (let [diff (- expiration-date (System/currentTimeMillis))]
      (.convert unit diff TimeUnit/MILLISECONDS))))

(defn no-delay [data]
  (->Delay data (to-long (now)) 0))

(defn delay-by [data n unit]
  (->Delay data (to-long (-> n unit from-now)) n))

(defn delay-more [data]
  (delay-by (.data data) (get delays (.delay data)) seconds))

(def successful-messages (atom []))
(add-watch successful-messages :success (fn [k r os ns] (println (last ns))))

(def fail-messages (atom []))
(defn add-fail-message [message]
  (swap! fail-messages conj message))

(defn add-success-message [message]
  (swap! successful-messages conj message))

(defn accept-message [q]
  (let [message (.take q)
        can-delay (get delays (.delay message) false)]
    (Thread/sleep 1000)
    (cond 
      (:up? @service) (add-success-message (:n (.data message)))
      can-delay (.put q (delay-more message))
      :else (add-fail-message (:n (.data message))))
    (recur q)))



(defn take-service-down []
  (swap! service assoc :up? false))

(defn hot-fix []
  (swap! service assoc :up? true))

(comment
  (hot-fix)
  (take-service-down)
)

(def q (DelayQueue.))
(def worker1 (future (accept-message q)))
(def worker2 (future (accept-message q)))

(future-cancel worker1)


(defn message [n] (no-delay {:n n}))

(doseq [n (range 100)]
  (.put q (message n)))

