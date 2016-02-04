(ns devcard.core
  (:require
   [om.core :as om :include-macros true]
   [sablono.core :as sab :include-macros true])
  (:require-macros
   [devcards.core :as dc :refer [defcard deftest]]))

(enable-console-print!)


(defn widget [data owner]
  (om/component
   (sab/html [:h2 "This is an om card, " (:text data)])))

(defcard omcard-ex
  (dc/om-root widget)
  {:text "yep"})

(defn main []
  ;; conditionally start the app based on whether the #main-app-area
  ;; node is on the page
  (if-let [node (.getElementById js/document "main-app-area")]
    (js/React.render (sab/html [:div "This is working"]) node)))

(main)

;; remember to run lein figwheel and then browse to
;; http://localhost:3449/cards.html

