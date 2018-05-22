(ns generate-art.core
  (:require [quil.core :as q])
  (:require [generate-art.dynamic :as dynamic])
  (:gen-class))

(q/defsketch example
             :title "Sketch"
             :setup dynamic/setup
             :draw dynamic/draw
             :size [700 400])

(defn refresh []
  (use :reload 'generate-art.dynamic)
  (.loop example))


(add-watch #'dynamic/draw :refresh (fn [_ _ _ _] 
                                     (refresh)))
