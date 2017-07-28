(ns reframe-redux.events
    (:require [re-frame.core :as re-frame]
              [reframe-redux.db :as db]))

(enable-console-print!)

(def connected-tools
  (.connect (.-__REDUX_DEVTOOLS_EXTENSION__ js/window)))


(defn construct-action [[type payload]]
  #js {:type (name type)
       :payload payload})

(def init-middleware
  (re-frame/after (fn [state _] (.init connected-tools (clj->js state)))))

(def send-middleware
  (re-frame/after 
   (fn [state action]
     (.send connected-tools (construct-action action) (clj->js state)))))



 (construct-action [:change-name "jimmy"]) 

(re-frame/reg-event-db
 :initialize-db
 [init-middleware]
 (fn  [_ _]
   db/default-db))


(re-frame/reg-event-db
 :change-name
 [send-middleware]
 (fn [state [_ name]]
   (assoc state :name name)))
