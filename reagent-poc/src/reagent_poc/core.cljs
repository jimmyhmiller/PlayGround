(ns reagent-poc.core
    (:require [reagent.core :as reagent]))

(defn pair->css [[key value]]
  (str (name key) ":" value ";"))

(defn ->css [coll]
  (if (string? coll)
    coll
    (apply str (interpose "\n" (map pair->css coll)))))

(defn styled [tag css]
  (((.-default js/styled) (name tag)) #js [(->css css)]))

(def container 
  (styled :div 
          {:background-color "#333"
           :height "40px"
           :width "40px"}))

(def white-text 
  (styled :p "color: #fff;"))

(defn home-page []
  [:div 
   [:h2 "Welcome to Reagent"]
   [:> container [:> white-text "hello"]]])

(defn mount-root []
  (reagent/render [home-page] (.getElementById js/document "app")))

(defn init! []
  (mount-root))
