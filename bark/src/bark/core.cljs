(ns ^:figwheel-always bark.core
  (:require-macros [cljs.core.async.macros :refer [go]])
  (:require
   [reagent.core :as reagent :refer [atom]]
   [goog.events :as events]
   [goog.dom :as dom]
   [cljs.core.async :refer [put! chan <! >! timeout]]))

(enable-console-print!)

(println "Edits to this text should show up in your developer console.")



;; This was cool code, but not sure it makes sense for what I want to achieve


(defonce app-state 
  (atom
   {:title-text "Day one of your new job as a software developer."
    :buttons {}
    :points 0}))


(defn update-points [f]
  (swap! app-state update-in [:points] f))

(defn update-text [text]
  (swap! app-state assoc-in [:title-text] text))

(defn register-button [config]
  (let [events (chan)
        config (assoc config :events events)]
    (swap! app-state assoc-in [:buttons (:id config)] config)
    config))

(defn assoc-button [name value]
  (swap! app-state update-in [:buttons name]
         (fn [old-value]
           (merge old-value value))))

(defn update-button [name key f]
  (swap! app-state update-in [:buttons name key] f))

(defn get-button [name]
  (get-in @app-state [:buttons name]))

(defn action-button [{:keys [id text enabled events percent]}]
  (let [color (if enabled "#000" "#aaa")
        gradient (str "linear-gradient(to left, #fff " (- 100 percent) "%, #ccc " (- 100 percent) "%)")]
    [:div {:style
           {:border (str "1px solid " color)
            :margin-top 10
            :padding "7px 14px"
            :color color
            :background gradient
            :cursor (if enabled :pointer :arrow)}
           :id id
           :key id
           :on-click #(when enabled (put! events %))}
     text]))


(defn render [button fn]
  (assoc-button button {:render fn}))


(defn disable-button [button sleep]
  (let [finished (chan)
        change 1
        time (/ sleep (/ 100 change))]
    (assoc-button button {:percent 100})
    (let [{:keys [enabled]} (get-button button)]
      (when enabled
        (go
          (assoc-button button {:enabled false}) 
          (loop []
            (let [{:keys [percent]} (get-button button)] 
              (when (not (neg? percent))
                (do
                  (<! (timeout time))
                  (update-button button :percent (fn [percent] (- percent change)))
                  (recur)))))
          (assoc-button button {:enabled true})
          (>! finished :finished)))
      finished)))


(defn defbutton [config handler]
  (handler (register-button config)))

(defbutton
  {:id :code
   :text "Code"
   :enabled true
   :percent 0}
  (fn [{:keys [id events]}]
    (go
      (<! (timeout 3000))
      (render id action-button)
      (while true
        (<! events)
        (update-text "Coding Feels Good")
        (update-points inc)
        (<! (disable-button id 3000))))))


(defbutton
  {:id :meeting
   :text "Attend Meeting"
   :enabled true
   :percent 0}
  (fn [{:keys [id events]}]
    (go
      (<! (timeout 3000))
      (render id action-button)
      (while true
        (<! events)
        (update-text "Lost your concentration")
        (update-points inc)
        (disable-button :code 10000)
        (<! (disable-button id 6000))))))



(defn world []
  [:div#world
   {:style
    {:display :flex
     :flex-direction :column
     :height :100vh
     :width :100vw
     :margin 0
     :padding 0
     :align-items :center
     :justify-content :center}}
   [:div {:style
          {:position :absolute
           :top 0
           :left 0}}
    (str "Points: " (:points @app-state))]
   [:div {:style {:display :flex}}
    [:h3 (:title-text @app-state)]]
   [:div  (for [[k config] (filter (fn [[_ v]] (contains? v :render)) (:buttons @app-state))]
           ((:render config) config))]])



(reagent/render-component [world]
                          (. js/document (getElementById "app")))


(defn on-js-reload []
  ;; optionally touch your app-state to force rerendering depending on
  ;; your application
  ;; (swap! app-state update-in [:__figwheel_counter] inc)
  )

