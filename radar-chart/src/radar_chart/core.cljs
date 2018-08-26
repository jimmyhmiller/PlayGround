(ns ^:figwheel-hooks radar-chart.core
    (:require [reagent.core :as reagent :refer [atom]]
              [react-d3-radar]))



(def Radar (.-default react-d3-radar))



(enable-console-print!)



(defonce app-state (atom {:text "Hello world!"}))


(defn hello-world []
  [:> Radar {:width 500
             :height 500
             :data (clj->js {:variables [{:key "x" :label "x"}]})
             :sets (clj->js [{:key :me :label :me :values {:x 5}}])}]
  [:div
   [:h1 (:text @app-state)]
   [:h3 "Edit this and watch it change!"]])

(reagent/render-component [hello-world]
                          (. js/document (getElementById "app")))

;; specify reload hook with ^;after-load metadata
(defn ^:after-load on-reload []
  ;; optionally touch your app-state to force rerendering depending on
  ;; your application
  ;; (swap! app-state update-in [:__figwheel_counter] inc)
)
