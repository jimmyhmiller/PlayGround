(ns examples-live-view.ultimate
  (:require [live-view-server.core :as live-view]))


(defn generate-board [[tl tm tr ml mm mr bl bm br]]
  {:tl tl
   :tm tm
   :tr tr
   :ml ml
   :mm mm
   :mr mr
   :bl bl
   :bm bm
   :br br})


(def state
  (atom {:tl (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :tm (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :tr (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :ml (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :mm (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :mr (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :bl (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :bm (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :br (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :big (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
         :current-player :X
         :current-board :_}))



(def app-history (atom [@state]))
(def replaying (atom false))


(add-watch state :history
           (fn [_ _ _ n]
             (when-not (or
                        (= (last @app-history) n)
                        @replaying)
               (swap! app-history conj n))))


(defn undo []
  (when (> (count @app-history) 1)
    (swap! app-history pop)
    (reset! state (last @app-history))))



(defn replay
  ([] (replay @app-history))
  ([history]
   (if (empty? history)
     (reset! replaying false)
     (do
       (reset! replaying true)
       (reset! state (first history))
       (Thread/sleep 1000)
       (recur (rest history))))))


(defn replay-n [n]
  (let [history @app-history]
    (replay (drop (- (count history) (inc n)) history))))


(def winning-positions
  [[:tr :tm :tl]
   [:mr :mm :ml]
   [:br :bm :bl]
   [:tr :mr :br]
   [:tm :mm :bm]
   [:tl :ml :bl]
   [:tl :mm :br]
   [:bl :mm :tr]])

(defn positions-from-board [board pos]
  (vals (select-keys board pos)))

(defn winner [row]
  (when (and
         (nil? (some #{:_} row))
         (= (count (distinct row)) 1) )
    (first row)))


(defn board-full? [board]
  (not-any? #(= % :_) (vals board)))


(defn active-board? [app board]
  (or
   (= (app :current-board) board)
   (= (app :current-board) :_)
   (board-full? (app (app :current-board)))))


(def switch-players {:X :O
                     :O :X})


(defn place-piece [app board spot]
  (let [current-state @app]
    (when (and
           (= (-> current-state board spot) :_)
           (not= board :big)
           (active-board? current-state board))
      (let [current-player (current-state :current-player)]
        (swap! app (fn [app]
                     (-> app
                         (assoc-in [board spot] current-player)
                         (assoc :current-player (switch-players current-player))
                         (assoc :current-board spot))))))))


(defn combine-keywords [k1 k2]
  (keyword (str (name k1) (name k2))))


(defn mark-cell [mark size]
  (cond (= mark :_)
        {}
        (= mark :X)
        {:background-image (str "url(images/X.png)")
         :background-size (str (- size 20) "px " (- size 20) "px")
         :background-repeat "no-repeat"
         :background-position "60% 60%"}
        (= mark :O)
        {:background-image "url(images/O.png)"
         :background-size (str (- size 20) "px " (- size 20) "px")
         :background-repeat "no-repeat"
         :background-position "60% 60%"}))


(defn create-cell [app board spot mark size]
  [:span {:class (str (name spot) )
          :style
          (merge
           {:height size
            :width size}
           (mark-cell mark size))
          :onclick [:place-piece {:board board :spot spot}]}])


(defn create-row [app board row marks cell-size]
  [:div.row
   {:style
    {:height cell-size
     :width (* (+ cell-size 6) 3)}}
   (map
    (fn [spot]
      (let [spot (combine-keywords row spot)
            mark (marks spot)]
        (create-cell app board spot mark cell-size)))
    [:l :m :r])])


(defn create-board [app board {:keys [x y] :as position} cell-size]
  (let [active? (active-board? app board)]
    [:div {:style
           {:position "absolute"
            :top y
            :left x}
           :class (if active? "active" "")}
     (map
      (fn [row position]
        (create-row app board row (app board) cell-size))
      [:t :m :b]
      [:l :m :r])]))


(defn create-ultimate-boards [app {:keys [x y] :as position} cell-size]
  [:div
   [:link {:rel "stylesheet" :href "ultimate.css"} "test"]
   [:span.big (create-board app :big position cell-size)]
   (map
    (fn [board board-position]
      (create-board app board board-position (float (/ cell-size 4))))
    [:tl :ml :bl :tm :mm :bm :tr :mr :br]
    (for [x1 (range 0 (* cell-size 3) cell-size)
          y1 (range 0 (* cell-size 3) cell-size)]
      {:x (+ (+ x1 x) (* cell-size 0.125))
       :y (+ (+ y1 y) (* cell-size 0.125))}))])


(defn view [{:keys [x y cell-size] :as state}]
  (create-ultimate-boards state {:x 0 :y 0} 330))

(defn event-handler [{:keys [action]}]
  (let [{:keys [board spot]} (second action)]
    (place-piece state board spot)))

(def live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 54321}))


(comment

  (.stop live-view-server))
