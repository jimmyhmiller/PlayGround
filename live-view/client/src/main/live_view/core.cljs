(ns live-view.core
  (:require [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit]
            [hipo.core :as hipo]
            [hipo.interceptor :as interceptor]
            [clojure.set]
            [goog.object]))


;; Doesn't look like event handlers are being set if the view changes.
;; Need to look into that.
(defn hipo-options [ws]
  {:attribute-handlers [{:target {:attr "onchange"}
                         :fn (fn [node a b [action payload]]
                               (.addEventListener node
                                                  "input"
                                                  (fn [e]
                                                    (let [writer (transit/writer :json)]
                                                      (.send ws
                                                             (transit/write
                                                              writer
                                                              [action (assoc (or payload {}) :value (.-value (.-target e)))]))))))}
                        {:target {:attr "onsubmit"}
                         :fn (fn [node a b val]
                               (.addEventListener node
                                                  "submit"
                                                  (fn [e]
                                                    (.preventDefault e)
                                                    (let [writer (transit/writer :json)]
                                                      (.send ws (transit/write writer val))))))}
                        {:target {:attr "onclick"}
                         :fn (fn [node a b val]
                               (.addEventListener node
                                                  "click"
                                                  (fn [e]
                                                    (.preventDefault e)
                                                    (let [writer (transit/writer :json)]
                                                      (.send ws (transit/write writer val))))))}
                        ;; The builtin style handler uses
                        ;; aset which doesn't work for objects now
                        {:target {:attr "style"}
                         :fn (fn [node x y styles]
                               (doseq [[k v] styles]
                                 (goog.object/set (.-style ^js/HTMLElement node)
                                                  (name k) v)))}]})



(deftype StyleInterceptor []
   interceptor/Interceptor
   (-intercept [_ t m f]
     (cond
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
  (let [port (or js/window.LIVE_VIEW_PORT js/window.location.port)
        ws (js/WebSocket. (str "ws://localhost:" port "/loc/"))
        renderer (create-renderer js/document.body ws)]
  (set! (.-onopen ws) (fn [] (.send ws "init")))
  (set! (.-onmessage ws) (fn [e] (let [reader (transit/reader :json)]
                                   (let [payload (transit/read reader (.-data e))]
                                     #_(prn payload)
                                     (renderer payload)))))))
