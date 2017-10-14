(ns jmx-clojure.live-charts
  (:import [org.jfree.chart ChartPanel JFreeChart]
           [javax.swing JComponent JLabel JPanel JFrame]))

(defn create-frame [title]
  (doto (new JFrame title)
    (.setVisible true)
    (.pack)
    (.setDefaultCloseOperation (. JFrame DISPOSE_ON_CLOSE))))


(defn display [^JComponent com title]
  (let [f (create-frame title)
        g (.getContentPane f)]
    (do (.add g com)
        (.pack f))
    f))

(defn redraw [chart panel time]
  (while true
    (Thread/sleep time)
    (.setChart panel ^JFreeChart (chart))))

(defn make-live-chart
  ([chart] (make-live-chart chart ""))
  ([chart chart-name]
   (let [panel (ChartPanel. ^JFreeChart (chart))
         _ (display panel chart-name)
         redrawer (future (redraw chart panel 250))]
     #(future-cancel redrawer))))













