(ns blocking-queue.core
  (:require [clj-time.core :refer [from-now minutes now seconds]]
            [clj-time.coerce :refer [to-long]]
            [clojure.pprint :as pprint])
  (:import [java.util.concurrent Delayed TimeUnit DelayQueue]
           [java.lang Comparable]
           [java.util Date]))

(def delays 
  {0 30
   30 60
   60 120
   120 240
   240 480})


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



(def service (atom {:up? true}))

(defn take-service-down []
  (swap! service assoc :up? false))

(defn hot-fix []
  (swap! service assoc :up? true))

(def successful-messages (atom []))

(add-watch successful-messages :success 
           (fn [k r os ns] (locking *out*
                             (pprint/pprint (last ns)))))

(defn add-success-message [message]
  (swap! successful-messages conj message))


(def fail-messages (atom []))

(defn add-fail-message [message]
  (swap! fail-messages conj message))


(defn accept-message [q]
  (let [message (.take q)
        can-delay (get delays (.delay message) false)]
    (Thread/sleep 1000)
    (cond
      (:up? @service) (add-success-message message)
      can-delay (.put q (delay-more message))
      :else (add-fail-message (:n (.data message))))
    (recur q)))

(comment
  (hot-fix)
  (take-service-down)
)

(def q (DelayQueue.))
(def worker1 (future (accept-message q)))
(def worker2 (future (accept-message q)))

(comment
  (future-cancel worker1)
  (future-cancel worker2))


(defn message [n] (no-delay {:n n}))

(doseq [n (range 10)]
  (.put q (message n)))

