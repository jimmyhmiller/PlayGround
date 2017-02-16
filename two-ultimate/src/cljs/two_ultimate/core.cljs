(ns two-ultimate.core
  (:require [om.core :as om :include-macros true]
            [om.dom :as dom :include-macros true]
            [sablono.core :as html :refer-macros [html]]))

(enable-console-print!)


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



(defonce app-state (atom {:tl (generate-board [:_ :_ :_ :_ :_ :_ :_ :_ :_])
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


(def app-history (atom [@app-state]))
(def replaying (atom false))


(add-watch app-state :history
           (fn [_ _ _ n]
             (when-not (or
                        (= (last @app-history) n)
                        @replaying)
               (swap! app-history conj n))))


(defn undo []
  (when (> (count @app-history) 1)
    (swap! app-history pop)
    (reset! app-state (last @app-history))))


(defn replay
  ([] (replay @app-history))
  ([history]
   (if (empty? history)
     (reset! replaying false)
     (do
       (reset! replaying true)
       (reset! app-state (first history))
       (js/setTimeout (partial replay (rest history)) 1000)))))


(defn replay-n [n]
  (let [history @app-history]
    (replay (drop (- (count history) (inc n)) history))))


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
  (when (and
         (= (-> @app board spot) :_)
         (not= board :big)
         (active-board? @app board))
    (let [current-player (@app :current-player)]
      (om/update! app (-> @app
                          (assoc-in [board spot] current-player)
                          (assoc :current-player (switch-players current-player))
                          (assoc :current-board spot))))))


(defn combine-keywords [k1 k2]
  (keyword (str (name k1) (name k2))))


(defn mark-cell [mark size]
  (cond (= mark :_)
        {}
        (= mark :X)
        {:background-image "url(images/X.png)"
         :background-size (str (- size 20) "px " (- size 20) "px")
         :background-repeat "no-repeat"
         :background-position "60% 60%"}
        (= mark :O)
        {:background-image "url(images/O.png)"
         :background-size (str (- size 20) "px " (- size 20) "px")
         :background-repeat "no-repeat"
         :background-position "60% 60%"}))


(defn create-cell [app board spot mark size]
  [:span {:class (name spot)
          :style
          (merge
           {:height size
            :width size}
           (mark-cell mark size))
          :on-click #(place-piece app board spot)}])


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
      [:t :m :b])]))


(defn create-ultimate-boards [app {:keys [x y] :as position} cell-size]
  [:div
   [:span.big (create-board app :big position cell-size)]
   (map
    (fn [board board-position]
      (create-board app board board-position (/ cell-size 4)))
    [:tl :ml :bl :tm :mm :bm :tr :mr :br]
    (for [x1 (range 0 (* cell-size 3) cell-size)
          y1 (range 0 (* cell-size 3) cell-size)]
      {:x (+ (+ x1 x) (* cell-size 0.125))
       :y (+ (+ y1 y) (* cell-size 0.125))}))])



  (om/root
   (fn [app owner]
     (reify
       om/IRender
       (render [_]
               (html (create-ultimate-boards app {:x 0 :y 0} 230)))))
   app-state
   {:target (. js/document (getElementById "app"))})
