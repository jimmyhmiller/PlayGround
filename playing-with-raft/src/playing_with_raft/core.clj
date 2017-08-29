(ns playing-with-raft.core
  (:require [clojure.core.async :as async :refer :all]))

;https://raft.github.io/raft.pdf


(defmulti process-input (fn [me servers state [type message]] type))

(defmethod process-input :elect-me [me servers state [_ candidate]]
  (when (nil? (:candidate @state))
    (swap! state assoc :candidate candidate)
    (go (>! (candidate servers) [:vote me]))))

(defmethod process-input :vote [me servers state [_ voter]]
  (swap! state update-in [:votes] conj voter))

(defn server [me input servers]
  (let [election-timeout (rand-int 30000)
        state (atom {:term 0 :votes #{} :leader nil :candidate nil})]
    (go (while true
          (alt!
            [input] ([value ch] (process-input me servers state value))
            (timeout election-timeout) (println "timeout"))
          (println me @state)))))


(def server-1 (chan))
(def server-2 (chan))


(def s1 (server :server-1 server-1 {:server-1 server-1 :server-2 server-2}))
(def s2 (server :server-2 server-2 {:server-1 server-1 :server-2 server-2}))

(close! s1)
(close! s2)

(go (>! server-1 [:elect-me :server-2]))

(<!! server-2)
