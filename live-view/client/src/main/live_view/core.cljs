(ns live-view.core
  (:require [morphdom]
            [editscript.core :as editscript]
            [editscript.edit :as edit]
            [cognitect.transit :as transit]
            [hipo.core :as hipo]
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
                         :fn (fn [node _ _ styles] 
                               (doseq [[k v] styles]
                                 (goog.object/set (.-style ^js/HTMLElement node)
                                                  (name k) v)))}]})


(defn apply-patch [node current-hiccup patch]
  (let [new-hiccup (editscript/patch current-hiccup (edit/edits->script patch ))]
    (hipo/reconciliate!
     node
     new-hiccup)
   
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
  (let [ws (js/WebSocket. "ws://localhost:50505/loc/")
        renderer (create-renderer js/document.body ws)]
  (set! (.-onopen ws) (fn [] (.send ws "init")))
  (set! (.-onmessage ws) (fn [e] (let [reader (transit/reader :json)]
                                   (let [payload (transit/read reader (.-data e))]
                                     #_(prn payload)
                                     (renderer payload)))))))
