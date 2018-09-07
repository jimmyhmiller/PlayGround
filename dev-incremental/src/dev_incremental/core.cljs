(ns ^:figwheel-hooks dev-incremental.core 
  (:require-macros  [cljs.core.async.macros :as m :refer [go alt!]])
  (:require
   [goog.dom :as gdom]
   [reagent.core :as reagent :refer [atom]]
   [clojure.spec.alpha :as s] 
   [cljs.core.async :refer [put! chan <! >! timeout]]))


(defn combine-reducers [reducer-map]
  (fn [state action] 
    (reduce (fn [st [key f]] (update st key f action)) state reducer-map)))

(defn make-dispatch [state-atom reducer]
  (fn [action] 
    (swap! state-atom reducer action)))



(defonce app-state 
  (atom {:buttons {1 {:percent 0
                      :text "Write Code"
                      :rate-of-change -1}
                   2 {:percent 0
                      :text "Attend Meeting"
                      :rate-of-change -0.2}}}))

(defmulti button-reducer (fn [state action] (first action)))

(defmethod button-reducer :modify-percent [state [_ {:keys [id change]}]]
  (if (>= (get-in state [id :percent]) 0)
    (update-in state [id :percent] + change)
    state))

(defmethod button-reducer :change-percent [state [_ {:keys [id percent]}]]
  (assoc-in state [id :percent] percent))

(def reducer
  (combine-reducers {:buttons button-reducer}) )

(def dispatch (make-dispatch app-state reducer))

(defn button-updater []
  (go
    (loop []
      (doseq [[id {:keys [rate-of-change]}] (:buttons @app-state)]
        (dispatch [:modify-percent {:id id :change rate-of-change}]))
      (<! (timeout 100))
      (recur))))

(defn build-gradient [percent color1 color2]
  (str "linear-gradient(to left, " 
       color1 " "
       (- 100 percent) "%, "
       color2 " "
       (- 100 percent) "%)"))

(defn button [{:keys [id text percent on-click]}]
  (let [disabled? (> percent 0)
        color (if disabled? "#aaa" "#000")
        gradient (build-gradient percent "#fff" "#aaa")]
    [:div {:style {:border (str "solid 1px " color)
                   :padding 5
                   :margin 10
                   :cursor :pointer
                   :background gradient
                   :color (if disabled? "#666" "#000") 
                   :text-align :center}
           :on-click (if disabled? (fn []) on-click)}
     text]))

(defn get-app-element []
  (gdom/getElement "app"))

(defn hello-world []
  [:div {:style {:width 150}}
   (for [[id button-info] (:buttons @app-state)]
     [button
      (merge
       button-info
       {:key id
        :id id
        :on-click #(dispatch [:change-percent {:id id :percent 100}])})])])

(defn mount [el]
  (reagent/render-component [hello-world] el))

;; conditionally start your application based on the presence of an "app" element
;; this is particularly helpful for testing this ns without launching the app
(when-let [el (get-app-element)]
  (mount el))

;; specify reload hook with ^;after-load metadata
(defn ^:after-load on-reload []
  (mount (get-app-element))
  ;; optionally touch your app-state to force rerendering depending on
  ;; your application
  ;; (swap! app-state update-in [:__figwheel_counter] inc)
)
