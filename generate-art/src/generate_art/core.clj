(ns generate-art.core
  (:require [quil.core :as q])
  (:require [generate-art.dynamic :as dynamic])
  (:gen-class))

(q/defsketch example
  :title "Sketch"
  :setup dynamic/setup
  :draw dynamic/draw
  :settings (fn []
              (q/smooth 2)
              (q/pixel-density 2))
  :size [800 900])

(defn refresh []
  (use :reload 'generate-art.dynamic)
  (.loop example))


(add-watch #'dynamic/draw :refresh (fn [_ _ _ _] 
                                     (refresh)))
