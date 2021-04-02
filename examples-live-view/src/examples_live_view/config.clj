(ns examples-live-view.config
  (:require [live-view-server.core :as live-view]))


(def initial-config {:greeting "Hello"
                     :delayed-field "Delayed"
                     :turbo false})

(def my-config (atom initial-config))


(defn view [{:keys [greeting delayed-field turbo]}]
  [:body
   [:link
    {:href "https://unpkg.com/blocks.css/dist/blocks.min.css",
     :rel "stylesheet"}]
   [:style {:type "text/css"}
    ".block.warning, .block.warning::before { background: #ee0a0a; color: white } "]
   [:div {:style {:display "flex"
                  :min-height "100vh"
                  :justify-content "center"
                  :align-items "center"}}
    [:div.card.block.fixed {:style {:width 600 :height 330}}
     [:h2 {:style {:border-bottom "1px solid #222"}} "Configure the Application"]
     [:p "This controls the settings for this application."]
     [:div [:label "Greeting Message"]]
     [:div.wrapper.block.fixed
      [:input {:type "text" :value greeting :onchange [:change-greeting]}]]
     [:div [:label "Delayed Update Text Field"]]
     [:div {:style {:display "flex"}}
      [:div.wrapper.block.fixed
       [:input {:type "text" :value delayed-field :onchange [:change-delayed-field]}]]
      [:button.block.accent {:onclick [:save-delayed-field]} "Save"]]
     [:label "Turbo Mode"]
     [:input {:type "checkbox" :checked turbo :onchange [:change-turbo]}]
     [:button.block.warning
      {:style {:background-color "#ee0a0a"
               :position "absolute"
               :top 270
               :left 520}
       :onclick [:reset]}
      "Reset"]]]])


(def state (atom {}))

(add-watch my-config :state (fn [_ _ _ value]
                              (reset! state value)))


(defn event-handler [{:keys [action]}]
  (let [[action-type payload] action]
    (println action)
    (case action-type
      :change-greeting (swap! my-config assoc :greeting (:value payload))
      :change-delayed-field (swap! state assoc :delayed-field (:value payload))
      :save-delayed-field (swap! my-config assoc :delayed-field (:delayed-field @state))
      :change-turbo (swap! my-config assoc :turbo  (:value payload))
      :reset (reset! my-config initial-config)
      (println "unhandled action-type"))))



(def live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 5555}))


(comment
  (.stop live-view-server)
  )
