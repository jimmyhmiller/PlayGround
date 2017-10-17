(ns jmx-clojure.core
  (:require [clojure.java.jmx :as jmx]
            [live-chart :as live]
            [incanter.core :as inc-core]
            [incanter.charts :as charts]
            [incanter.datasets :as datasets]
            [dorothy.core :as dot]
            [jmx-clojure.live-charts :refer [make-live-chart]])
  (:import [org.jfree.chart ChartPanel JFreeChart])
  (:import [javax.swing JComponent JLabel JPanel JFrame]))

(def current-state (atom :c))

(reset! current-state :d)



(defn graph []
  (-> (dot/digraph [[:a :b :c] [:b :d]
                    [@current-state {:style :filled :color :black :fontcolor :white}]])
      dot/dot))

(def frame (dot/show! (graph)))


(defn rerender-image [graph frame]
  (let [^bytes bytes (dot/render (graph) {:format :png})
        icon  (javax.swing.ImageIcon. bytes)
        w     (.getIconWidth icon)
        h     (.getIconHeight icon)
        lbl   (javax.swing.JLabel. icon)
        sp    (javax.swing.JScrollPane. lbl)]
    (doto frame
      (.setContentPane sp)
      (.setVisible true))))


(defn redraw [time graph frame]
  (while true
    (Thread/sleep time)
    (rerender-image graph frame)))

(def draw-task (future (redraw 250 graph frame)))

(future-cancel draw-task)

(defn random-data []
  (map (fn [x] {:x (rand-int 100) :y (rand-int 100)}) (range 100)))

(def dates (map #(-> (date-time (+ 1900 %))
                     .getMillis)
                (range 100)))

(def data (atom [30 50]))


(reset! data [50 30])

(swap! data (fn [[x y]] [(dec x) (inc y)]))


(defn make-chart [data]
  (-> (charts/bar-chart [:thing1 :thing2] @data)
      (charts/set-y-range 0 120)))

(def live (make-live-chart (partial make-chart data)))





(defn- createFrame [title]
  (doto (new JFrame title)
    (.setVisible true)
    (.pack)
    (.setDefaultCloseOperation (. JFrame DISPOSE_ON_CLOSE))))

(defn- display [^JComponent com title]
  (let [f (createFrame title)
        g (.getContentPane f)]
    (do (.add g com)
        (.pack f))
    f))



(inc-core/read-dataset {:x 1 :y 1})

(def things (atom []))

(defn get-used-heap []
  (-> (jmx/mbean "java.lang:type=Memory")
      :HeapMemoryUsage 
      :used))

(->> "*:*"
     jmx/mbean-names
     seq
     (map #(.toString %))
     (map (fn [name] [name (jmx/mbean name)])))


(live/show (live/time-chart [get-used-heap]))


(reset! things [])




