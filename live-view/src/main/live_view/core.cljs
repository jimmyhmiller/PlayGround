(ns live-view.core
  (:require [morphdom]
            [editscript.core :as editscript]
            [crate.core :as crate]))

(defn apply-morphdom-patch [node current-hiccup patch]
  (let [new-hiccup (editscript/patch current-hiccup patch)]
    (morphdom
     node
     (crate/html new-hiccup)

     #js {:onBeforeElUpdated (fn [from to]
                               (cond 
                                 (and
                                  (= (.-tagName to) "INPUT")
                                  (not (.-value (.-attributes to))))
                                 false

                                 :else true))})
   
    new-hiccup))

(defn create-renderer [dom-node]
  (let [dom-node (atom dom-node)
        virtual-dom (atom nil)]
    (fn [data]
      (let [current-vdom @virtual-dom]
        (case (:type data)
          :patch (if current-vdom
                   (reset! virtual-dom (apply-morphdom-patch @dom-node current-vdom (:value data)))
                   {:type :error
                    :reason :no-state})
          :init (do
                  (reset! virtual-dom (:value data))
                  (let [node (crate/html (:value data))]
                    (.replaceWith @dom-node node)
                    (reset! dom-node node))))))))


