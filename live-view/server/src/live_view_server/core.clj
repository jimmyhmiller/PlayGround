(ns live-view-server.core
  (:require [ring.adapter.jetty9 :as jetty]
            [ring.middleware.resource]
            [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit]
            [clojure.java.io :as io])
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


;; I should actually put these on a queue are grab only the latest
;; Or maybe I need to do that on client side?
;; regardless, if the server is sending things faster than we can render,
;; We can just skip states and only render the latest
(defn update-view-and-send-patch [view state internal-state-atom broadcast]
  (let [current-state @internal-state-atom
        new-view-state (view state)
        patch (edit/get-edits (editscript/diff (:view-state current-state) new-view-state))
        _ (swap! internal-state-atom assoc :view-state new-view-state)
        out (ByteArrayOutputStream. 4096)
        writer (transit/writer out :json)]
    (transit/write writer {:type :patch
                           :value patch})
    (broadcast (:clients current-state) (.toString out))))



(defn make-ws-handler [internal-state-atom on-event]
  {:on-connect (fn [ws]
                 (do (println "connecting")
                     (swap! internal-state-atom update :clients assoc ws {:metadata {}})
                     (println "done connecting")))
   :on-error (fn [ws e] (println "error" e))
   :on-close (fn [ws status-code reason]
               (swap! internal-state-atom update :clients dissoc ws)
               (println "close"))
   :on-text (fn [ws text-message]
              (if (= text-message "init")
                (do (println "init")
                    (send-transit! ws {:type :init
                                       :value (:view-state @internal-state-atom)})
                    (println "done init"))
                (do
                  (println "event")
                  (try
                    (on-event {:ws ws
                               :action (read-transit text-message)
                               :current-state @internal-state-atom
                               :state-atom internal-state-atom
                               :internal-state-atom internal-state-atom})
                    (catch Exception e
                      (.printStackTrace e)))
                  (println "done event"))))
   :on-bytes (fn [ws bytes offset len])
   :on-ping (fn [ws bytebuffer])
   :on-pong (fn [ws bytebuffer])} )



(defn web-handler [req]
  (let [uri (:uri req)]
    (case uri
      "/" {:body  (slurp (io/resource "index.html"))
           :headers {"Content-Type" "text/html"}}
      "/main.js" {:body (slurp (io/resource "main.js"))})))

;; TODO: Need to break things apart so you can run this on your own server
;; TODO: Need to make broadcast overridable, but it needs more context
(defn start-live-view-server [{:keys [state view event-handler port]}]
  (let [internal-state (atom {:clients {}
                              :view-state nil})]
    (swap! internal-state assoc :view-state (view @state))
    (when (var? view)
      (add-watch view :view-updated
                 (fn [_ _ _ new-view-fn]
                   ;; Should these be in futures?
                   (future
                     (update-view-and-send-patch new-view-fn @state internal-state #'broadcast)))))
    (add-watch state :send-websocket
               (fn [_ _ _ state]
                 ;; Should these be in futures?
                 (future
                   (update-view-and-send-patch view state internal-state broadcast))))

    (jetty/run-jetty (ring.middleware.resource/wrap-resource #'web-handler "/")
                     {:websockets {"/loc" (fn [_req]
                                            (#'make-ws-handler internal-state event-handler))}
                      :port (or port 50505)
                      :join? false})))





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
