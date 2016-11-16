(ns app.ui
  (:require
   #_[om.core :as om :include-macros true]
   [cljs.spec :as s :include-macros true]
   [clojure.test.check.generators]
   [sablono.core :as sab :include-macros true])
  (:require-macros
   [devcards.core :as dc :refer [defcard deftest]]))

(enable-console-print!)

(s/def ::checked boolean?)
(s/def ::disabled boolean?)
(s/def ::label #{"Active" "Ordered" "Arrived"})
(s/def ::toggle-state (s/keys :req [::checked ::disabled ::label]))


(defn get-states [spec]
  (as-> spec sp
        (s/exercise sp 100)
        (map first sp)
        (distinct sp)))

#_(defn toggle [{:keys [app.ui/checked app.ui/disabled app.ui/label] :as state}]
  (defcard example
    (sab/html
     [:div
      [:label label]
      [:input {:type :checkbox
               :checked checked
               :disabled disabled}]])
    state {:inspect-data true}))



(s/def ::size #{:small :medium :large})
(s/def ::active boolean?)
(s/def ::shape #{:square :circle})
(s/def ::image (s/keys :req [::size ::active ::shape]))


(def image-size
  {:small 50
   :medium 100
   :large 150})

(defn photo [{:keys [app.ui/size app.ui/active app.ui/shape] :as state}]
  (defcard profile
    (sab/html
     [:img {:src "http://az616578.vo.msecnd.net/files/2016/02/22/635917076783020956-1943515409_dwight.jpeg"
            :style {:filter (if active "grayscale(100%)" "none")
                    :width (image-size size)
                    :height (image-size size)
                    :border-radius (if (= shape :circle) (image-size size) 0)}}])
    state {:inspect-data true}))


(doseq [state (get-states ::image)]
  (photo state))

#_(doseq [state (get-states ::toggle-state)]
  (toggle state))

(defn main []
  ;; conditionally start the app based on whether the #main-app-area
  ;; node is on the page
  (if-let [node (.getElementById js/document "main-app-area")]
    (.render js/ReactDOM (sab/html [:div "This is working"]) node)))

(main)

;; remember to run lein figwheel and then browse to
;; http://localhost:3449/cards.html

