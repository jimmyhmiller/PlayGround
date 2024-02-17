(ns jimmyhmiller.infinite-craft
  (:require [clj-http.client :as http]
            [clojure.edn :as edn]
            [clojure.set :as set]
            [clojure.java.io :as io]))




(def log-file "craft.log")

(defn clear-log-file []
  (io/delete-file log-file))


(defn craft* [first second]
  (:body
   (http/get "https://neal.fun/api/infinite-craft/pair"
             {:headers {:authority "neal.fun"
                        :referer "https://neal.fun/infinite-craft/"}
              :accept :json
              :socket-timeout 10000
              :connection-timeout 10000
              :query-params {:first first :second  second}
              :as :json})))


(defn log-result [first second result]
  (prn  {:first first :second second :result result})
  (spit log-file (prn-str {:first first :second second :result result}) :append true))


(defn craft [first second]
  (prn {:first first :second second})
  (let [result (craft* first second)]
    (log-result first second result)
    result))



(def state (atom {:elements #{"Water" "Fire" "Earth" "Wind"}
                  :pairs #{}
                  :results []}))


(defn rand-element [coll]
  (rand-nth (seq coll)))

(defn step-once []
  (let [current-state @state
        first (rand-element (:elements current-state))
        second (rand-element (:elements current-state))
        pair (set [first second])]
    (if (or (= (count pair) 1) (contains? (:pairs current-state) pair))
      (recur)
      (let [result (craft first second)]
        (swap! state (fn [state]
                       (-> state
                           (update :elements set/union pair)
                           (update :elements conj  (:result result))
                           (update :pairs conj pair)
                           (update :results conj {:first first :second second :result result}))))
        result))))



(count (:pairs @state))
(count (:elements @state))

(count
 (set
  (filter :isNew
          (map :result (:results @state)))))


(def stop-stats (atom false))

(def stats
  (future
    (loop [total (count (:elements @state))]
      (when (not @stop-stats)
        (Thread/sleep 60000)
        (let [new-total (count (:elements @state))]
          (println "====================================================")
          (println (- new-total total) "new items in the last minute")
          (println "====================================================")
          (recur new-total))))))


(comment
  (def stop (atom false))


  (reset! stop true)
  (reset! stop false)

  (def process
    (future
      (loop []
        (when (not @stop)
          (Thread/sleep 3000)
          (try
            (step-once)
            (catch Exception e
              (println "Timed out continuing to next one")))
          (recur)))))


  
  @process

  (future-cancel process)

  (step-once)
  )





(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (greet {:name (first args)}))
