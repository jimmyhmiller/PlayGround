(ns clojurescript-talk.core
    (:require [reagent.core :as reagent :refer [atom]]
              [reagent.session :as session]
              [secretary.core :as secretary :include-macros true]
              [accountant.core :as accountant]
              [jimmyhmiller.spectacle]))

;; (def spectacle (reagent/adapt-react-class (.-Spectacle js/Spectacle)))
;; (def deck (reagent/adapt-react-class (.-Deck js/Spectacle)))
;; (def slide (reagent/adapt-react-class (.-Slide js/Spectacle)))
;; (def text (reagent/adapt-react-class (.-Text js/Spectacle)))


(def spectacle (.-Spectacle js/Spectacle))
(def deck (.-Deck js/Spectacle))
(def slide (.-Slide js/Spectacle))
(def text (.-Text js/Spectacle))




;; -------------------------
;; Views



(defn slides []
  [:> spectacle [:> deck [:> slide [:> text "hello"]]]])





;; -------------------------
;; Initialize app

(defn mount-root []
  (reagent/render slides (.getElementById js/document "app")))

(defn init! []
  (mount-root))
