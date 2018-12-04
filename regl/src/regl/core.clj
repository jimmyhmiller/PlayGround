(ns regl.core
  (:require [fn-fx.fx-dom :as dom]
            [fn-fx.controls :as ui]
            [fn-fx.diff :refer [component defui render should-update?]])
  (:import [javafx.scene.control TreeItem]))


(def tree
  (let [root (TreeItem. "asdf")]
    (.add (.getChildren root) (TreeItem. "asadfadf"))
    root))

(defui MainComponent 
  (render [this args]
          (ui/stage
           :title "Hello World!"
           :shown true
           :min-width 300
           :min-height 300
           :scene (ui/scene
                   :root (ui/stack-pane
                          :children [(ui/tree-view :root tree)
                                     (ui/button
                                      :text "Say 'Hello World'"
                                      :on-action {:say "Hello Word!"})])))))

(def state (atom {}))

(defn handler [evt]
  (println "Received Event: " evt))

(def ui (agent (dom/app (main-component {}) handler)))

(remove-watch #'main-component :ui)

(add-watch #'main-component :ui
           (fn [_ _ _ _]
             (swap! state update :force-bit not)))

(add-watch state :ui
           (fn [_ _ _ ns]
             (send ui
                   (fn [old-ui]
                     (dom/update-app old-ui (main-component ns))))))
