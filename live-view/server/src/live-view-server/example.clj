(ns live-view-server.example
  (:require [live-view-server.core :as live]))


(def state (atom {:name "No One"
                  :input-value ""
                  :items []
                  :actions []}))



;; Updating view can cause some state issues
;; TODO: FIX
#_(defn view [{:keys [name input-value items actions]}]
  [:body
   [:h1 {:style {:color input-value}} "Hello " name]
   [:form {:onsubmit [:submit {}]}
    [:input {:value input-value
             :onchange [:input-change {}]}]
    [:button {:type "submit"} "Submit"]
    [:button {:type "button" :onclick [:clear {}]} "Clear"]]
   [:ul
    (for [item actions]
      [:li (prn-str item)])]
   [:ul
    (for [item items]
      [:li item])]])



(defn bar-chart [{:keys [coll]}]
  [:div {:style {:display "flex"
                 :flex-direction "row"
                 :align-items "flex-end"}}
   (for [{:keys [color value title]} coll]
     [:div {:style {:background-color color
                    :height (str value "px")
                    :color "white"
                    :margin-right "20px"
                    :width "200px"}} title])])

(defn view [_]
  [:body
   [:h1 "Graph Things"]
   (bar-chart {:coll [{:color "red" :value 200 :title "requests"}
                      {:color "green" :value 100 :title "requests"}
                      {:color "blue" :value 50 :title "requests"}]})])


(defn handle-event [{:keys [action internal-state]}]
  (let [[type payload] action]
   #_ (swap! state update :actions conj action)
    (case type
      :submit (when (not (empty? (:input-value @state)))
                (dosync (swap! state update :items conj (:input-value @state))
                        (swap! state assoc :input-value "")))
      :input-change (swap! state assoc :input-value (:value payload))
      :clear (swap! state assoc :items [] :actions [])
      (prn action))))


(def server (live/start-live-view-server {:state state 
                                     :view #'view 
                                     :event-handler #'handle-event}))

(comment
  (.stop server))



(comment


  (def process
    (future
      (let [items (map (fn [x]
                         (str "item" x))
                       (range 0 5000))]
        (loop [offset 0]
          (if (>= offset 5000)
            (recur 0)
            (do
              (Thread/sleep 16)
              (swap! state assoc :items (take 50 (drop offset items)))
              (recur (inc offset))))))))

  (future-cancel process))
