(ns live-view-server.core
  (:require [ring.adapter.jetty9 :as jetty]
            [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit])
  (:import [java.io ByteArrayInputStream ByteArrayOutputStream]))


;; Metadata per connection
;; Filtering Broadcast
;; LifeCycle Handlers
;; Routing



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


(defn broadcast [clients patch]
  (doseq [[ws _] clients]
    ;; Super important to pass this empty map
    ;; it is the callback handlers. Otherwise
    ;; we are calling send synchronously.
    (jetty/send! ws patch)))

(defn update-view-and-send-patch [view state internal-state-atom broadcast]
  (let [new-view-state (view state)
        patch (edit/get-edits (editscript/diff (:view-state @internal-state-atom) new-view-state))
        _ (swap! internal-state-atom assoc :view-state new-view-state)
        out (ByteArrayOutputStream. 4096)
        writer (transit/writer out :json)]
    (transit/write writer {:type :patch
                           :value patch})
    (broadcast (:clients @internal-state-atom) (.toString out))))



(defn make-ws-handler [internal-state-atom on-event]
  {:on-connect (fn [ws]
                 (println "connect")
                 (swap! internal-state-atom update :clients assoc ws {:metadata {}}))
   :on-error (fn [ws e] (println "error" e))
   :on-close (fn [ws status-code reason]
               (swap! internal-state-atom update :clients dissoc ws)
               (println "close"))
   :on-text (fn [ws text-message]
              (if (= text-message "init")
                (send-transit! ws {:type :init
                                   :value (:view-state @internal-state-atom)})
                (on-event {:ws ws
                           :action (read-transit text-message)
                           :current-state @internal-state-atom
                           :state-atom internal-state-atom
                           :internal-state-atom internal-state-atom})))
   :on-bytes (fn [ws bytes offset len] (println "bytes" bytes) )
   :on-ping (fn [ws bytebuffer] (println "ping") )
   :on-pong (fn [ws bytebuffer] (println "pong"))} )


;; TODO: Need to break things apart so you can run this on your own server
;; TODO: Need to have options like port
;; TODO: Need to make broadcast overridable, but it needs more context
(defn start-live-view-server [{:keys [state view event-handler]}]
  (let [internal-state (atom {:clients {}
                              :view-state nil})]
    (swap! internal-state assoc :view-state (view @state))
    (when (var? view)
      (add-watch view :view-updated
                  (fn [_ _ _ new-view-fn]
                    (update-view-and-send-patch new-view-fn @state internal-state #'broadcast))))
    (add-watch state :send-websocket
               (fn [_ _ _ state]
                 (update-view-and-send-patch view state internal-state broadcast)))


    ;; Is there a performance penalty for the way I am doing the dynamic things here?
    ;; TODO: Need to actualy serve the client side page here.
    (jetty/run-jetty (fn [req] {:body "It Works"})
                     {:websockets {"/loc" (fn [_req]
                                            (#'make-ws-handler internal-state event-handler))}
                      :port 50505
                      :join? false}))
  )




(def state (atom {:name "No One"
                  :input-value ""
                  :items []
                  :actions []}))


;; Updating view can cause some state issues
;; TODO: FIX
(defn view [{:keys [name input-value items actions]}]
  [:body
   [:h1 {:style {:color "green"}} "Hello " name]
   [:form {:onsubmit [:submit {}]}
    [:input {:value input-value
             :onchange [:input-change {}]}]
    [:button {:type "submit"} "Submit"]
    [:button {:type "button" :onclick [:clear {}]} "Clear"]]
   [:ul
    (for [item actions]
      [:li (prn-str item)])]
   [:ul
    (for [item items]
      [:li item])]])


(defn handle-event [{:keys [action internal-state]}]
  (let [[type payload] action]
    #_(swap! state update :actions conj action)
    (case type
      :submit (when (not (empty? (:input-value @state)))
                (dosync (swap! state update :items conj (:input-value @state))
                        (swap! state assoc :input-value "")))
      :input-change (swap! state assoc :input-value (:value payload))
      :clear (swap! state assoc :items [] :actions [])
      (prn action))))


(def server (start-live-view-server {:state state 
                                     :view #'view 
                                     :event-handler #'handle-event}))

(comment
  (.stop server))





;; Need client local state
;; That means I need to run the view function for each client.
;; But I should really only do this if I need to.
;; How can I signal that this is the case?


;; How about this?
;; First we store the view state on each connection.
;; Because of persistent data structures, if we are smart this shouldn't be too bad.
;; I would love to just do that and a memoized view function (fifo). But that isn't enough.
;; If we have any user information, that means the data will be different. :(
;; So even if we render the same view every time, the input would be different for each user.
;; So we need to memoize the function, but also have a system for declaring what data we depend on.
;; Have to think about how the defaults should work here. But should probably follow the principle of maximal
;; flexibility at the base level and building up a less flexible but more convenient higher-level interface.


;; Maybe something like?
;; :view-depends-on [:state :user-state]




(comment

  (swap! state assoc :name "Jimmy!")
  (swap! state assoc :items [])

  (swap! state assoc :actions [])


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
