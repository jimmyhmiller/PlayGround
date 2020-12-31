(ns live-view-server.core
  (:require [ring.adapter.jetty9 :as jetty]
            [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit])
  (:import [java.io ByteArrayInputStream ByteArrayOutputStream]))

(def state (atom {:name "No One"
                  :items []}))

(swap! state assoc :name "Jimmy!")
(swap! state assoc :items ["test"])



(def process 
  (future
    (let [items (map (fn [x]
                       (str "item" x))
                     (range 0 5000))]
      (loop [offset 0]
        (if (>= offset 5000)
          (recur 0)
          (do
            (Thread/sleep 16)
            (swap! state assoc :items (take 50 (drop offset items)))
            (recur (inc offset))))))))



(future-cancel process)





(comment
  
  (loop []
    (doseq [i (concat (range 0 500 1)  (range 500 0 -1))]
      #_(Thread/sleep 16)
      (swap! state assoc :left (str i "px")))
    (recur)))


(defn view [{:keys [name left top items]}]
  [:body [:h1 {:style {:color "green"
                       #_#_:position "absolute"
                       #_#_:left left
                       #_#_:top top}} "Hello " name]
   [:ul
    (for [item items]
      [:li item])]])


(defn send-transit! [ws payload]
  (let [out (ByteArrayOutputStream. 4096)
        writer (transit/writer out :json)]
    (transit/write writer payload)
    (jetty/send! ws (.toString out))))

(def clients (atom []))

(add-watch state :send-websocket
           (fn [_ _ old-value new-value]
             (let [patch (edit/get-edits (editscript/diff (view old-value) (view new-value)))
                   out (ByteArrayOutputStream. 4096)
                   writer (transit/writer out :json)]
               (transit/write writer {:type :patch
                                      :value patch})
               (doseq [ws @clients]
                 (jetty/send! ws (.toString out))))))





(def ws-handler
  {:on-connect (fn [ws]
                 (println "connect")
                 (swap! clients conj ws))
   :on-error (fn [ws e] (println "error" e))
   :on-close (fn [ws status-code reason]
               (swap! clients #(filterv (fn [x] (not= ws x)) %))
               (println "close"))
   :on-text (fn [ws text-message] (when (= text-message "init")
                                    (send-transit! ws {:type :init
                                                       :value (view @state)})))
   :on-bytes (fn [ws bytes offset len] (println "bytes" bytes) )
   :on-ping (fn [ws bytebuffer] (println "ping") )
   :on-pong (fn [ws bytebuffer] (println "pong"))} )




(defn app [req]
  {:body "It works!"})

(def server (jetty/run-jetty app {:websockets {"/loc" ws-handler}
                                  :port 50505
                                  :join? false}))
(.stop server)
