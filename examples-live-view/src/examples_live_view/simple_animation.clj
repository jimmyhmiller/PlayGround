(ns example-live-view.simple-animation
  (:require [live-view-server.core :as live-view]))


(defn view [{:keys [top left]}]
  [:div {:style {:width 200 :height 200 :background-color "black"
                 :transform (str "translate("  left "px," top "px)")}}])

(def state (atom {:top 20
                  :left 20}))

(comment
  (future-cancel my-future)
  )
  
(def my-future
  (future
    (loop [n 0
           velocity 5]
      (Thread/sleep 16)
      (swap! state assoc :top n :left n)
      (let [next (+ n velocity)]
        (recur next (cond (> next 500) -5
                          (< next 0) 5
                          true velocity))))))



(defn event-handler [])


(def live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 4321}))
