(ns example-live-view.flappy-bird
  (:require [live-view-server.core :as live-view]))

;; https://github.com/bhauman/flappy-bird-demo


(defn floor [x] (Math/floor x))



(defn translate [start-pos vel time]
  (floor (+ start-pos (* time vel))))

(def horiz-vel -0.25)
(def gravity 0.05)
(def jump-vel 15)
(def start-y 312)
(def bottom-y 561)
(def flappy-x 212)
(def flappy-width 57)
(def flappy-height 31)
(def pillar-spacing 324)
(def pillar-gap 158)
(def pillar-width 86)

(def starting-state { :timer-running false
                      :jump-count 0
                      :initial-vel 0
                      :start-time 0
                      :flappy-start-time 0
                      :flappy-y   start-y
                      :pillar-list
                      [{ :start-time 0
                         :pos-x 900
                         :cur-x 900
                         :gap-top 200 }]})

(defn reset-state [_ cur-time]
  (-> starting-state
      (update-in [:pillar-list] (fn [pls] (map #(assoc % :start-time cur-time) pls)))
      (assoc
       :start-time cur-time
       :cur-time cur-time
       :flappy-start-time cur-time
       :timer-running true)))




(def flap-state (atom starting-state))


(defn curr-pillar-pos [cur-time {:keys [pos-x start-time] }]
  (translate pos-x horiz-vel (- cur-time start-time)))

(defn in-pillar? [{:keys [cur-x]}]
  (and (>= (+ flappy-x flappy-width)
           cur-x)
       (< flappy-x (+ cur-x pillar-width))))

(defn in-pillar-gap? [{:keys [flappy-y]} {:keys [gap-top]}]
  (and (< gap-top flappy-y)
       (> (+ gap-top pillar-gap)
          (+ flappy-y flappy-height))))

(defn bottom-collision? [{:keys [flappy-y]}]
  (>= flappy-y (- bottom-y flappy-height)))

(defn collision? [{:keys [pillar-list] :as st}]
  (if (some #(or (and (in-pillar? %)
                      (not (in-pillar-gap? st %)))
                 (bottom-collision? st)) pillar-list)
    (assoc st :timer-running false)
    st))

(defn new-pillar [cur-time pos-x]
  {:start-time cur-time
   :pos-x      pos-x
   :cur-x      pos-x
   :gap-top    (+ 60 (rand-int (- bottom-y 120 pillar-gap)))})

(defn update-pillars [{:keys [pillar-list cur-time] :as st}]
  (let [pillars-with-pos (map #(assoc % :cur-x (curr-pillar-pos cur-time %)) pillar-list)
        pillars-in-world (sort-by
                          :cur-x
                          (filter #(> (:cur-x %) (- pillar-width)) pillars-with-pos))]
    (assoc st
      :pillar-list
      (if (< (count pillars-in-world) 3)
        (conj pillars-in-world
              (new-pillar
               cur-time
               (+ pillar-spacing
                  (:cur-x (last pillars-in-world)))))
        pillars-in-world))))

(defn sine-wave [st]
  (assoc st
    :flappy-y
    (+ start-y (* 30 (Math/sin (/ (:time-delta st) 300))))))

(defn update-flappy [{:keys [time-delta initial-vel flappy-y jump-count] :as st}]
  (if (pos? jump-count)
    (let [cur-vel (- initial-vel (* time-delta gravity))
          new-y   (- flappy-y cur-vel)
          new-y   (if (> new-y (- bottom-y flappy-height))
                    (- bottom-y flappy-height)
                    new-y)]
      (assoc st
        :flappy-y new-y))
    (sine-wave st)))

(defn score [{:keys [cur-time start-time] :as st}]
  (let [score (- (Math/abs (floor (/ (- (* (- cur-time start-time) horiz-vel) 544)
                               pillar-spacing)))
                 4)]
  (assoc st :score (if (neg? score) 0 score))))

(defn time-update [timestamp state]
  (-> state
      (assoc
          :cur-time timestamp
          :time-delta (- timestamp (:flappy-start-time state)))
      update-flappy
      update-pillars
      collision?
      score))


(defn jump [{:keys [cur-time jump-count] :as state}]
  (-> state
      (assoc
          :jump-count (inc jump-count)
          :flappy-start-time cur-time
          :initial-vel jump-vel)))

;; derivatives

(defn border [{:keys [cur-time] :as state}]
  (-> state
      (assoc :border-pos (mod (translate 0 horiz-vel cur-time) 23))))

(defn pillar-offset [{:keys [gap-top] :as p}]
  (assoc p
    :upper-height gap-top
    :lower-height (- bottom-y gap-top pillar-gap)))

(defn pillar-offsets [state]
  (update-in state [:pillar-list]
             (fn [pillar-list]
               (map pillar-offset pillar-list))))

(defn world [state]
  (-> state
      border
      pillar-offsets))

(defn px [n] (str n "px"))

(defn pillar [{:keys [cur-x pos-x upper-height lower-height]}]
  (println upper-height)
  [:div.pillars
   [:div.pillar.pillar-upper {:style {:left (px cur-x)
                                       :height upper-height}}]
   [:div.pillar.pillar-lower {:style {:left (px cur-x)
                                       :height lower-height}}]])



(defn time-loop [time]
  (let [new-state (swap! flap-state (partial time-update time))]
    (when (:timer-running new-state)
      (Thread/sleep 30)
      (recur (inst-ms (java.time.Instant/now))))))

(def my-future (atom nil))



(defn start-game []
  (let [time  (inst-ms (java.time.Instant/now))]
    (reset! flap-state (reset-state @flap-state time))
    (reset! my-future (future (time-loop time)))))

(defn main-template [{:keys [score cur-time jump-count
                             timer-running border-pos
                             flappy-y pillar-list]}]
  [:body
   [:link {:rel "stylesheet" :href "flappy.css"}]
   [:div.board {:onclick [:flap]}
    [:h1.score score]
    (if-not timer-running
      [:a.start-button {:onclick [:start-game]}
       (if (< 1 jump-count) "RESTART" "START")]
      [:span])
    [:div (map pillar pillar-list)]
    [:div.flappy {:style {:top (px flappy-y)}}]
    [:div.scrolling-border {:style { :background-position-x (px border-pos)}}]]])



(defn view [state]
  (main-template (world state)))

(defn event-handler [{:keys [action]}]
  (println action)
  (let [[action-type payload] action]
    (case action-type
      :start-game (start-game)
      :flap (future (swap! flap-state jump))
      (println "unhandled" action-type))))


(def live-view-server
  (live-view/start-live-view-server
   {:state flap-state
    :view #'view
    :event-handler #'event-handler
    :port 4444}))

(comment
  (let [node (.getElementById js/document "board-area")]
    (defn renderer [full-state]
      (.render js/ReactDOM (main-template full-state) node)))

  (add-watch flap-state :renderer (fn [_ _ _ n]
                                    (renderer (world n))))

  (reset! flap-state @flap-state))
