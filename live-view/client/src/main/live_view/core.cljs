(ns live-view.core
  (:require [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit]
            [hipo.core :as hipo]
            [hipo.interceptor :as interceptor]
            [clojure.set]
            [goog.object]))


(def event-listeners (atom {}))


(def event-fns
  {"onkeydown"
   (fn [node _ _ [action payload]]
     (let [listener (fn [e]
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS
                               (transit/write
                                writer
                                [action (assoc (or payload {})
                                               :keycode
                                               (.. e -keyCode))]))))]
       (.removeEventListener node "keydown" (get-in event-listeners [node "onkeydown"]) false)
       (swap! event-listeners assoc-in [node "onkeydown"] listener)
       (.addEventListener node "keydown" listener)))
   "onchange"
   (fn [node _ _ [action payload]]
     (let [listener (fn [e]
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS
                               (transit/write
                                writer
                                [action (assoc (or payload {}) :value
                                               (if (= (.. e -target -type) "checkbox")
                                                 (.. e -target -checked)
                                                 (.. e -target -value)))]))))]
       (println "ONCHANGE---------------------------------")
       (println  (get-in @event-listeners [node "onchange"]) )
       (println (.removeEventListener node "input" (get-in @event-listeners [node "onchange"]) false))
       (println "--------------------------------------------------")
       (swap! event-listeners assoc-in [node "onchange"] listener)
       (.addEventListener node "input" listener)))
   "onblur"
   (fn [node _ _ val]
     (let [listener (fn [e]
                      (.preventDefault e)
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "blur" (get-in @event-listeners [node "onblur"]) false)
       (println  "Adding blur" listener)
       (swap! event-listeners assoc-in [node "onblur"] listener)
       (.addEventListener node "blur" listener)))
   "onsubmit"
   (fn [node _ _ val]
     (let [listener (fn [e]
                      (.preventDefault e)
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "submit" (get-in @event-listeners [node "onsubmit"]) false)
                  (swap! event-listeners assoc-in [node "onsubmit"] listener)
                  (.addEventListener node "submit" listener)))
   "onclick"
   (fn [node _ _ val]
     (let [listener  (fn [e]
                       (.preventDefault e)
                       (let [writer (transit/writer :json)]
                         (.send  js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "click" (get-in @event-listeners [node "onclick"]) false)
       (swap! event-listeners assoc-in [node "onclick"] listener)
       (.addEventListener node "click" listener)))
   "ondoubleclick"
   (fn [node _ _ val]
     (let [listener (fn [e]
                      (.preventDefault e)
                      (let [writer (transit/writer :json)]
                        (.send  js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "dblclick" (get-in @event-listeners [node "ondoubleclick"]) false)
       (swap! event-listeners assoc-in [node "ondoubleclick"] listener)
                                 (.addEventListener node "dblclick" listener)))})


;; Doesn't look like event handlers are being set if the view changes.
;; Need to look into that.
(defn hipo-options [ws]
  {:attribute-handlers [{:target {:attr "onkeydown"}
                         :fn (get event-fns "onkeydown")}
                        {:target {:attr "onchange"}
                         :fn  (get event-fns "onchange")}
                        {:target {:attr "onblur"}
                         :fn (get event-fns "onblur")}
                        {:target {:attr "onsubmit"}
                         :fn (get event-fns "onsubmit")}
                        {:target {:attr "onclick"}
                         :fn (get event-fns "onclick")}
                        {:target {:attr "ondoubleclick"}
                         :fn  (get event-fns "ondoubleclick")}
                        ;; The builtin style handler uses
                        ;; aset which doesn't work for objects now
                        {:target {:attr "style"}
                         :fn (fn [node _ _ styles]
                               (doseq [[k v] styles]
                                 (goog.object/set (.-style ^js/HTMLElement node)
                                                  (name k) (if (number? v)
                                                             (str v "px")
                                                             v))))}]})



(deftype StyleInterceptor []
   interceptor/Interceptor
   (-intercept [_ t m f]
     (cond
       (and (= (:old-value m) (:new-value m))  (= t :update-attribute))
       nil
       
       (and (= t :update-attribute) (= (:name m) :style))
       (do
         (let [new-value (:new-value m)
               new-keys (set (keys new-value))
               old-keys  (set (keys (:old-value m)))
               removed-attributes (clojure.set/difference old-keys new-keys )]
           (doseq [attr removed-attributes]
             (goog.object/set (.-style ^js/HTMLElement (:target m))
                              (name attr) ""))
           (doseq [[k v] new-value]
             (goog.object/set (.-style ^js/HTMLElement (:target m))
                              (name k) v))))
       (and (= t :update-attribute))
       (do
         #_(println "Attribute" m)
         (if-let [event-handler (get event-fns (name (:name m)))]
           (do
            #_ (println event-handler (:target m) nil nil (:new-value m))
             (event-handler (:target m) nil nil (:new-value m))))
           (f))
       

      :else (f))))

(defn apply-patch [node current-hiccup patch]
  (let [new-hiccup (editscript/patch current-hiccup (edit/edits->script patch ))]
    (hipo/reconciliate!
     node
     new-hiccup
     {:interceptors [(StyleInterceptor.)]})

    new-hiccup))

(defn create-renderer [dom-node ws]
  (let [dom-node (atom dom-node)
        virtual-dom (atom nil)]
    (fn [data]
      (let [current-vdom @virtual-dom]
        (case (:type data)
          :patch (if current-vdom
                   (reset! virtual-dom (apply-patch @dom-node current-vdom (:value data)))
                   {:type :error
                    :reason :no-state})
          :init (do
                  (reset! virtual-dom (:value data))
                  (let [node (hipo/create (:value data) (hipo-options ws))]
                    (.replaceWith @dom-node node)
                    (reset! dom-node node))))))))








(defn init []
  (println "init")
  (let [port (or js/window.LIVE_VIEW_PORT js/window.location.port)
        ws (js/WebSocket. (str "ws://localhost:" port "/loc/"))
        renderer (create-renderer js/document.body ws)]
    ;; ugly hack
    (set! (.-liveWS js/window) ws)
    (println "should connect" ws)
    (set! (.-onerror ws) (fn [e] (println "error" e)))
    (set! (.-onopen ws) (fn []
                          (println "sending init")
                          (.send ws "init")
                          (println "sent init")))
    (set! (.-onmessage ws) (fn [e]
                              (println "got message")
                             (let [reader (transit/reader :json)]
                               (let [payload (transit/read reader (.-data e))]
                                 (println "read message")
                                 (renderer payload)
                                  (println "rendered")))))))
