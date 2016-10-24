(ns async-testing.core
  (:gen-class)
  (:require [clojure.core.async :as async :refer :all]
            [org.httpkit.client :as http]
            [clojure.walk :as walk]
            [clojure.data.json :as json]))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))



(defn seconds [n]
  (* n 1000))

(defn minutes [n]
  (seconds (* n 60)))



(defn wait-send-repeat [network chan name]
  (go-loop
   [last-message :alive]
   (let [peer (rand-nth network)]
     (<! (timeout (seconds 10)))
     (println name last-message)
     (alt!
      chan ([message] (if message (recur message)))
      [[peer last-message]] (recur last-message)))))

(defn run-process []
  (let [b (chan)
        c (chan)]
    (wait-send-repeat [c] b "first")
    (wait-send-repeat [b] c "second")
    [b c]))

(def chans (run-process))
(def b (first chans))
(def c (second chans))

(go (println "message" (<! q)))
(put! b "Other message")


(defn network [n]
  (into [] (map #(chan) (range n))))


(network 10)

(defn get-http
  ([url]
   (get-http url {}))
  ([url options]
   (let [c (chan)]
     (http/get url options
               (fn [response] ;; asynchronous response handling
                 (put! c (update response :body #(json/read-str % :key-fn keyword)))))
     c)))

(go (println (<!
              (get-http
               "https://api.github.com/repos/nomicness/a-whole-new-world/issues/6/reactions"
               {:headers {"accept" "application/vnd.github.squirrel-girl-preview"}}))))




(defn check-on-pr [pr-url]
  (go-loop
   [retries 0]
   (println "retrying")
   (when (> retries 0)
     (<! (timeout (* (minutes 5) retries))))
   (let [{:keys [error body]} (<! (get-http pr-url))]
     (cond
      error (recur (inc retries))
      (not= (:state body) "closed") (recur 1)
      :else (println body)))))


(check-on-pr "https://api.github.com/repos/nomicness/a-whole-new-world/issues/6" )

(def c (chan))

(go-loop
 []
 (println (<! c))
 (recur))

(>!! c "hello world")

