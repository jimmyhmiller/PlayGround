(ns clojurescript-talk.core
    (:require [reagent.core :as r]
              [reagent.session :as session]
              [secretary.core :as secretary :include-macros true]
              [accountant.core :as accountant]))



;;(defonce state (reagent/atom 0))


(defonce click-count (r/atom 0))

(defn state-ful-with-atom []
  [:div {:on-click #(swap! click-count inc)}
   "I have been clicked " @click-count " times."])



(comment (defn app []
           [:button 
            {:style {:font-size 30}
             :on-click #(swap! state inc)} 
            "Count: " @state]))

(defn mount-root []
  (r/render state-ful-with-atom (.getElementById js/document "app")))

(defn init! []
  (mount-root))
