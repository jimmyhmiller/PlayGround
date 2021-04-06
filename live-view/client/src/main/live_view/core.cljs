(ns live-view.core
  (:require [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit]
            [hipo.core :as hipo]
            [hipo.interceptor :as interceptor]
            [clojure.set]
            [goog.object]))


;; Ported from https://github.com/egierach/2048

(def event-listeners (atom {}))




;; This is far from optimal. Really what we need is a virtual event
;; system like react so we can handle this more generically. There is
;; a ton of repetition here and it is very incomplete. If I made a
;; virtual event system, we could handle all of this much better and
;; not have to keep updating this each time I have a new use case.
(def event-fns
  {"onkeydown"
   (fn [node _ _ [action payload]]
     (let [listener (fn [e]
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS
                               (transit/write
                                writer
                                [action (assoc (or payload {})
                                               :key (.. e -key)
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
       (.removeEventListener node "input" (get-in @event-listeners [node "onchange"]) false)
       (swap! event-listeners assoc-in [node "onchange"] listener)
       (.addEventListener node "input" listener)))
   "onblur"
   (fn [node _ _ val]
     (let [listener (fn [e]
                      (.preventDefault e)
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "blur" (get-in @event-listeners [node "onblur"]) false)
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
       (.addEventListener node "dblclick" listener)))
   "onmouseover"
   (fn [node _ _ val]
     (let [listener (fn [e]
                      (.preventDefault e)
                      (let [writer (transit/writer :json)]
                        (.send  js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "mouseover" (get-in @event-listeners [node "onmouseover"]) false)
       (swap! event-listeners assoc-in [node "onmouseover"] listener)
       (.addEventListener node "mouseover" listener)))
   "onmouseout"
   (fn [node _ _ val]
     (let [listener (fn [e]
                      (.preventDefault e)
                      (let [writer (transit/writer :json)]
                        (.send js/window.liveWS (transit/write writer val))))]
       (.removeEventListener node "mouseover" (get-in @event-listeners [node "onmouseout"]) false)
       (swap! event-listeners assoc-in [node "onmouseout"] listener)
       (.addEventListener node "mouseout" listener)))})


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
                         :fn (get event-fns "ondoubleclick")}
                        {:target {:attr "onmouseover"}
                         :fn (get event-fns "onmouseover")}
                        {:target {:attr "onmouseout"}
                         :fn (get event-fns "onmouseout")}
                        ;; The builtin style handler uses
                        ;; aset which doesn't work for objects now
                        ;; Duplicated down below in the interceptor
                        {:target {:attr "style"}
                         :fn (fn [node _ _ styles]
                               (doseq [[k v] styles]
                                 (goog.object/set (.-style ^js/HTMLElement node)
                                                  (name k) (if (number? v)
                                                             (str v "px")
                                                             v))))}]})



;; This exists to handle status in a better way but also to deal with
;; event handlers.
(deftype LiveViewInterceptor []
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
                              (name k) (if (number? v)
                                         (str v "px")
                                          v)))))
       (and (= t :update-attribute))
       (do
         (if-let [event-handler (get event-fns (name (:name m)))]
           (event-handler (:target m) nil nil (:new-value m)))
           (f))


      :else (f))))

(defn apply-patch [node current-hiccup patch]
  (let [new-hiccup (editscript/patch current-hiccup (edit/edits->script patch ))]
    (hipo/reconciliate!
     node
     new-hiccup
     {:interceptors [(LiveViewInterceptor.)]})

    new-hiccup))

(defn create-renderer [dom-node ws]
  (let [dom-node (atom dom-node)
        virtual-dom (atom nil)]
    (fn [data]
      (let [current-vdom @virtual-dom]
        (case (:type data)
          :patch (if current-vdom
                   (do
                     (reset! virtual-dom (apply-patch @dom-node current-vdom (:value data))))
                   {:type :error
                    :reason :no-state})
          :init (do
                  (reset! virtual-dom (:value data))
                  (let [node (hipo/create (:value data) (hipo-options ws))]
                    (.replaceWith @dom-node node)
                    (reset! dom-node node))))))))



;; This has no reconnection logic at all. Need to add that.
(defn init []

  (let [port (or js/window.LIVE_VIEW_PORT js/window.location.port)
        ws (js/WebSocket. (str "ws://localhost:" port "/loc/"))
        renderer (create-renderer js/document.body ws)]
    ;; ugly hack to help us handle stuff in events.
    (set! (.-liveWS js/window) ws)
    (set! (.-onerror ws) (fn [e] (println "error" e)))
    (set! (.-onopen ws) (fn []
                          (.send ws "init")))
    (set! (.-onmessage ws) (fn [e]
                             (let [reader (transit/reader :json)]
                               (let [payload (transit/read reader (.-data e))]
                                 (renderer payload)))))))
