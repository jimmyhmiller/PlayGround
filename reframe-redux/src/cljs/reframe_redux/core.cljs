(ns reframe-redux.core
    (:require [reagent.core :as reagent]
              [re-frame.core :as re-frame]
              [re-frame.db :as reframe-db]
              [reframe-redux.events]
              [reframe-redux.subs]
              [reframe-redux.db :as db]
              [reframe-redux.views :as views]
              [reframe-redux.config :as config]))


(defn dev-setup []
  (when config/debug?
    (enable-console-print!)
    (println "dev mode")))

(defn mount-root []
  (re-frame/clear-subscription-cache!)
  (reagent/render [views/main-panel]
                  (.getElementById js/document "app")))



(re-frame/dispatch [:change-name "jimmy"])



(defn ^:export init []
  (re-frame/dispatch-sync [:initialize-db])
  (dev-setup)
  (mount-root))









