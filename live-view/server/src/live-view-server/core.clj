(ns live-view-server.core
  (:require [ring.adapter.jetty9 :as jetty]
            [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit])
  (:import [java.io ByteArrayInputStream ByteArrayOutputStream]))

(def state (atom {:name "No One"
                  :input-value ""
                  :items []}))

(comment

  (swap! state assoc :name "Jimmy!")
  (swap! state assoc :items [])



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
)




(comment
  
  (loop []
    (doseq [i (concat (range 0 500 1)  (range 500 0 -1))]
      #_(Thread/sleep 16)
      (swap! state assoc :left (str i "px")))
    (recur)))


(defn view [{:keys [name input-value items]}]
  [:body [:h1 {:style {:color "green"}} "Hello " name]
   [:form {:onsubmit [:test]}
    [:input {:value input-value
             :onchange [:input-change {}]}]
    [:button {:type "submit"} "test"]]
   [:ul
    (for [item items]
      [:li item])]])


(defn send-transit! [ws payload]
  (let [out (ByteArrayOutputStream. 4096)
        writer (transit/writer out :json)]
    (transit/write writer payload)
    ;; Super important to pass this empty map
    ;; it is the callback handlers. Otherwise
    ;; we are calling send synchronously.
    (jetty/send! ws (.toString out) {})))


(defn string->stream
  ([s] (string->stream s "UTF-8"))
  ([^String s encoding]
   (-> s
       (.getBytes encoding)
       (java.io.ByteArrayInputStream.))))

(defn read-transit [data]
  (let [in (string->stream data)
        reader (transit/reader in :json)]
    (transit/read reader)))

(def clients (atom []))

(add-watch state :send-websocket
           (fn [_ _ old-value new-value]
             (let [patch (edit/get-edits (editscript/diff (view old-value) (view new-value)))
                   out (ByteArrayOutputStream. 4096)
                   writer (transit/writer out :json)]
               (transit/write writer {:type :patch
                                      :value patch})
               (doseq [ws @clients]
                 ;; Super important to pass this empty map
                 ;; it is the callback handlers. Otherwise
                 ;; we are calling send synchronously.
                 (jetty/send! ws (.toString out) {})))))





(defn ws-handler [on-event]
  {:on-connect (fn [ws]
                 (println "connect")
                 (swap! clients conj ws))
   :on-error (fn [ws e] (println "error" e))
   :on-close (fn [ws status-code reason]
               (swap! clients #(filterv (fn [x] (not= ws x)) %))
               (println "close"))
   :on-text (fn [ws text-message] 
              
              (if (= text-message "init")
                (send-transit! ws {:type :init
                                   :value (view @state)})
                (on-event (read-transit text-message))))
   :on-bytes (fn [ws bytes offset len] (println "bytes" bytes) )
   :on-ping (fn [ws bytebuffer] (println "ping") )
   :on-pong (fn [ws bytebuffer] (println "pong"))} )



(defn handle-event [[action payload]]
  #_(println [action payload])
  (case action
    :test (when (not (empty? (:input-value @state))) 
            (dosync (swap! state update :items conj (:input-value @state))
                    (swap! state assoc :input-value "")))
    :input-change (swap! state assoc :input-value (:value payload))
    (prn [action payload])))

(defn app [req]
  {:body "It works!"})

(def server (jetty/run-jetty app {:websockets {"/loc" (ws-handler handle-event)}
                                  :port 50505
                                  :join? false}))

(comment
  (.stop server))
