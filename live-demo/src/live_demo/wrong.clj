(ns live-demo.core
  (:require [clojure.java.jmx :as jmx]
            [live-chart :as live]
            [incanter.core :as inc-core]
            [incanter.charts :as charts]
            [incanter.datasets :as datasets]
            [dorothy.core :as dot]))


(def state (atom {:last-worker 0
                  :workers []
                  :items-left 64
                  :items-finished 0}))


(defn make-workers [start workers finish]
  (let [worker-graph 
        (->> workers
             (map (fn [w] [start w finish]))
             (into []))]
    (concat [{:rankdir :LR} [start] [finish]] worker-graph)))

(def start "start")
(def finish "finish")
(def workers (:workers @state))

 (filter (complement empty?) (make-workers start workers finish))

(defn create-graph []
  (let [{:keys [last-worker workers items-left items-finished]} @state
        start (str "start(" items-left ")")
        finish (str "finished(" items-finished ")")]
    (filter (complement empty?) (make-workers start workers finish))))


(defn add-worker [state]
  (let [{:keys [last-worker workers items-left]} @state
        new-worker (inc last-worker)
        worker-name (str "worker-" new-worker)
        left (dec items-left)]
    (swap! state assoc 
           :workers (conj workers worker-name)
           :last-worker new-worker
           :items-left left)))

(add-worker state)

(create-graph)

(defn graph []
  (-> (dot/digraph (create-graph))
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
