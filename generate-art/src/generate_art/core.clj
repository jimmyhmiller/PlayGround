(ns generate-art.core
  (:require [quil.core :as q])
  (:require [generate-art.dynamic :as dynamic])
  (:gen-class))

(q/defsketch example
  :title "Sketch"
  :setup #'dynamic/setup
  :draw #'dynamic/draw
  :key-pressed #'dynamic/key-press
  :key-released #'dynamic/key-release
  :settings (fn []
              (q/smooth 2)
              (q/pixel-density 2))
  :size [800 900])

(defn refresh []
  #_(use :reload 'generate-art.dynamic)
  (.loop example))


(add-watch #'dynamic/draw :refresh (fn [_ _ _ _] 
                                     (refresh)))
