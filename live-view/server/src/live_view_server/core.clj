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

(def ^:dynamic *live-view-context* nil)

(defn send-transit! [ws payload]
  (binding [*live-view-context* (or *live-view-context* {})]
    (let [out (ByteArrayOutputStream. 4096)
          writer (transit/writer out :json)]
      (transit/write writer payload)
      ;; Super important to pass this empty map
      ;; it is the callback handlers. Otherwise
      ;; we are calling send synchronously.
      (jetty/send! ws (.toString out) {}))))


(defn string->stream
  ([s] (string->stream s "UTF-8"))
  ([^String s encoding]
   (binding [*live-view-context* (or *live-view-context* {})]
     (-> s
         (.getBytes encoding)
         (java.io.ByteArrayInputStream.)))))

(defn read-transit [data]
  (binding [*live-view-context* (or *live-view-context* {})]
    (let [in (string->stream data)
          reader (transit/reader in :json)]
      (transit/read reader))))


(defn broadcast [clients patch]
  (binding [*live-view-context* (or *live-view-context* {})]
    (doseq [[ws _] clients]
      ;; Super important to pass this empty map
      ;; it is the callback handlers. Otherwise
      ;; we are calling send synchronously.
      (jetty/send! ws patch {}))))


;; I should actually put these on a queue are grab only the latest
;; Or maybe I need to do that on client side?
;; regardless, if the server is sending things faster than we can render,
;; We can just skip states and only render the latest
(defn update-view-and-send-patch [view state internal-state-atom broadcast]
  (binding [*live-view-context* (or *live-view-context* {})]
    (let [[old-view-state new-view-state]
          (swap-vals! internal-state-atom (fn [internal-state]
                                            #_(if (<= (:epoch internal-state) epoch))
                                            (assoc internal-state :view-state (view @state))
                                           #_ internal-state))
          patch (edit/get-edits (editscript/diff (:view-state old-view-state) (:view-state new-view-state)))
          out (ByteArrayOutputStream. 4096)
          writer (transit/writer out :json)]
      (when-not (empty? patch)
        (transit/write writer {:type :patch
                              #_#_ :epoch epoch
                               :value patch})
        (broadcast (:clients new-view-state) (.toString out))))))

(defn make-ws-handler [internal-state-atom on-event]
  (let [live-view-context (or *live-view-context* {})]
    {:on-connect (fn [ws]
                   (binding [*live-view-context* (or live-view-context {})]
                     (swap! internal-state-atom update :clients assoc ws {:metadata {}})))
     :on-error (fn [ws e] (println "error" e))
     :on-close (fn [ws status-code reason]
                 (binding [*live-view-context* (or live-view-context {})]
                   (swap! internal-state-atom update :clients dissoc ws)))
     :on-text (fn [ws text-message]
                (binding [*live-view-context* (or live-view-context {})]
                  (if (= text-message "init")
                    (send-transit! ws {:type :init
                                       :value (:view-state @internal-state-atom)})
                    (try
                      (on-event {:action (read-transit text-message)})
                      (catch Exception e
                        (.printStackTrace e))))))
     :on-bytes (fn [ws bytes offset len])
     :on-ping (fn [ws bytebuffer])
     :on-pong (fn [ws bytebuffer])}))



(defn web-handler [req]
  (binding [*live-view-context* (or *live-view-context* {})]
    (let [uri (:uri req)]
      (case uri
        "/" {:body  (slurp (io/resource "index.html"))
             :headers {"Content-Type" "text/html"}}
        "/main.js" {:body (slurp (io/resource "main.js"))}))))




;; TODO: Need to break things apart so you can run this on your own server
;; TODO: Need to make broadcast overridable, but it needs more context
(defn start-live-view-server [{:keys [state view event-handler port skip-frames-allowed?]}]
  (let [live-view-context (or *live-view-context* {})]
    (let [^java.util.concurrent.ArrayBlockingQueue message-queue (java.util.concurrent.ArrayBlockingQueue. (* 1024 10))
          queue-worker (future
                         (let [coll (java.util.ArrayList.)]
                           (loop []
                             ;; Should consider if we want to take all
                             ;; messages and just process the last.
                             ;; Could mean we skip too many frames
                             ;; But also probobly give us much better
                             ;; performance
                             (binding [*live-view-context* (or *live-view-context* {})]
                               (try
                                 (if skip-frames-allowed?
                                   (do
                                     (.drainTo message-queue coll)
                                     (when-not (zero?( .size coll))
                                       (let [message (.get coll (dec (.size coll)))]
                                         (.clear coll)
                                         (when message
                                           (apply update-view-and-send-patch message))))
                                     (Thread/sleep 5))
                                   (let [message (.take message-queue)]
                                     (when message
                                       (apply update-view-and-send-patch message))))
                                 (catch Exception e
                                   (.printStackTrace e))))
                             (recur))))]
      (let [internal-state (atom {:clients {}
                                  :view-state nil
                                  :epoch 0})]
        (swap! internal-state assoc :view-state (view @state))
        (when (var? view)
          (add-watch view :view-updated
                     (fn [_ _ _ new-view-fn]
                       ;; Probably should reconsider future?
                       (future
                         (binding [*live-view-context* (or live-view-context {})]
                           (update-view-and-send-patch new-view-fn
                                                       state
                                                       internal-state
                                                       #'broadcast))))))
        (add-watch state :send-websocket
                   (fn [_ _ _ state-value]
                     ;; A queue ensures we are processing messages in order
                     ;; This is definitely not enough to handle our distributed issue.
                     ;; It is possible to send messages in order, but
                     ;; to recieve them out of order.
                     (binding [*live-view-context* (or live-view-context {})]
                       (.put message-queue
                             [view
                              state
                              internal-state
                              broadcast]))))

        (let [live-view-context *live-view-context*
              server (jetty/run-jetty (ring.middleware.resource/wrap-resource #'web-handler "/")
                                      {:websockets {"/loc" (fn [_req]
                                                             (binding [*live-view-context* (or live-view-context {})]
                                                               (make-ws-handler internal-state event-handler)))}
                                       :port (or port 50505)
                                       :join? false})]
          server)))))





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
