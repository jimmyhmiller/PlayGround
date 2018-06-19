(ns charts-exercise.core 
(:require
   [reagent.core :as r]
   [recharts :refer [LineChart XAxis YAxis CartesianGrid Line]]))


(def data 
  (clj->js
   [{:name "A" :uv 4000 :pv 2400 :amt 2400}
    {:name "B" :uv 3000 :pv 1398 :amt 2210}
    {:name "C" :uv 2000 :pv 9800 :amt 2290}
    {:name "D" :uv 2780 :pv 3908 :amt 2000}
    {:name "E" :uv 1890 :pv 4800 :amt 2181}
    {:name "F" :uv 2390 :pv 3800 :amt 2500}
    {:name "G" :uv 3490 :pv 4300 :amt 2100}]))

(defn home-page []
  [:div 
   [:> LineChart {:width 500 :height 300 :data data}
    [:> XAxis {:dataKey "name"}]
    [:> YAxis]
    [:> CartesianGrid {:stroke "#eee" :strokeDashArray "5 5"}]
    [:> Line {:type "monotone" :dataKey "uv" :stroke "#8884d8"}]
    [:> Line {:type "monotone" :dataKey "pv" :stroke "#387908"}]]])


(defn mount-root []
  (r/render [home-page] (.getElementById js/document "app")))

(defn init! []
  (mount-root))
