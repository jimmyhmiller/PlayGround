(ns radar-bug.core
  (:require [reagent.core :as reagent :refer [atom]]
            [react-d3-radar]))

(def Radar (.-default react-d3-radar))


(defn hello-world []
  [:> Radar {:width 500
             :height 500
             :data (clj->js {:variables [{:key "x" :label "x"}]})
             :sets (clj->js [{:key :me :label :me :values {:x 5}}])}])


(reagent/render-component [hello-world]
                          (. js/document (getElementById "app")))