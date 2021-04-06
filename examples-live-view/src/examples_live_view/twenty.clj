(ns examples-live-view.twenty
  (:require [live-view-server.core :as live-view]))


(def key-counter
  "An incrementing key for board elements."
  (atom 0))

(defn next-key
  "Incrents key-counter."
  []
  (swap! key-counter inc))

(defn board-element
  "Returns a map with an key and a value"
  [value]
  {:key (next-key) :value value})

(defn n-board-elements
  "Returns a vector of n board elements with value v."
  [n v]
  (vec (repeatedly n #(board-element v))))

(defn initial-board
  "Returns a 16 element vector of 0 elements with two randomly-placed 2's.
  Meant to be the initial board state of the game."
  []
  (let [board-positions (shuffle (range 0 16))
        [a b] (take 2 board-positions)]
    (-> (n-board-elements 16 0)
        vec
        (assoc a (board-element 2))
        (assoc b (board-element 2)))))

(defn left-justify
  "Returns a copy of row stripped of zero-valued elements and right-padded
  with zero-valued elements.

  Example: (left-justify [0 4 0 8]) => [4 8 0 0]"
  [row]
  (let [without-zeroes (vec (remove #(zero? (:value %)) row))]
    (vec (into without-zeroes
               (n-board-elements (- (count row) (count without-zeroes)) 0)))))

(defn merge-dups
  "Returns a copy of row with elements at index1 and index2 merged
  if they have equal values.

  If elements are merged, the merge-callback is called with the
  value of the merged cell.  Meant to aid in score-keeping.

  The element at index1 is overwritten with a double-valued version
  of the element at index2, and index2's spot is back-filled with a
  zero element.

  Example: (merge-dups [2 2 0 0] 0 1) => [4 0 0 0]"
  [row index1 index2 merge-callback]
  (let [item1 (nth row index1)
        item2 (nth row index2)]
    (if (and (< 0 (:value item1))
             (= (:value item1) (:value item2)))
      (do
        (merge-callback (* 2 (:value item2)))
        (assoc row
               index1 (update item2 :value * 2)
               index2 (board-element 0)))
      row)))

(defn all-neighbors
  "Returns a list of index pairs for all neighboring elements in row.

  Example: (all-neighbors [:a :b :c :d]) => ([0 1] [1 2] [2 3])"
  [row]
  (let [indices (range (count row))]
    (map vector indices (drop 1 indices))))

(defn march-left
  "Returns a copy of row with non-zero elements marched toward
  the left and duplicates merged.

  Compound merges are not allowed, but parallel merges are.
  Examples:
  (march-left [2 2 4 2]) => [4 4 2 0] ;; not [8 2 0 0]
  (march-left [2 2 2 2]) => [4 4 0 0] ;; not [8 0 0 0]"
  [row merge-callback]
  (let [row-as-vector (vec row)
        justified (left-justify row-as-vector)
        merged (reduce (fn [r [i1 i2]]
                         (merge-dups r i1 i2 merge-callback))
                       justified
                       (all-neighbors row))]
    (left-justify merged)))

(defn march-right
  "Returns a copy of row marched to the right (reverse of march-left)."
  [row merge-callback]
  (-> row
      (reverse)
      (march-left merge-callback)
      (reverse)))

(defn transpose-matrix
  "Takes a collection of collections representing a matrix as a series of rows
  and returns a series of vectors representing the matrix as columns.

  Example: (transpose-matrix [[1 2 3]     [[1 4 7]
                              [4 5 6]  =>  [2 5 8]
                              [7 8 9]])    [3 6 9]]"
  [matrix]
  (apply mapv vector matrix))


;; keycodes for arrow keys
(def left "ArrowLeft")
(def right "ArrowRight")
(def up "ArrowUp")
(def down "ArrowDown")

(def arrows #{left right up down})


(defn resolve-input
  "Resolves the input action on the current board to produce the next.

  Assumes a 4x4 board.
  Example: (resolve-input [2 2 2 2        [4 4 0 0
                           0 2 2 2         4 2 0 0
                           0 0 2 2    =>   4 0 0 0
                           0 0 0 2]        2 0 0 0]
                          left
                          callback)"
  [current-board action merge-callback]
  (let [rows (partition 4 current-board)]
    (cond
      (= action left)
      (flatten (map #(march-left % merge-callback) rows))

      (= action right)
      (flatten (map #(march-right % merge-callback) rows))

      (= action up)
      (->> rows
           (transpose-matrix)
           (map #(march-left % merge-callback))
           (transpose-matrix)
           (flatten))

      (= action down)
      (->> rows
           (transpose-matrix)
           (map #(march-right % merge-callback))
           (transpose-matrix)
           (flatten))

      :else
      current-board)))

(defn upgrade-random-zero
  "Returns a copy of new-board with a random zero-valued element's value increased
  to either 2 or 4.

  No upgrade will take place if previous-board and new-board are equal, or if
  there are no zero-valued elements left in new-board.

  The replacement values for the chosen zero element will be 2's roughly 80%
  of the time."
  [previous-board new-board]
  (let [values-index (map-indexed vector (map :value new-board))
        zeroes (filter #(zero? (second %)) values-index)
        replacement-value (rand-nth [2 2 2 2 4])]
    (if (or (= (map :value previous-board)
               (map :value new-board))
            (empty? zeroes))
      new-board
      (let [random-index (first (rand-nth zeroes))]
        (as-> random-index z
          (nth new-board z)
          (update z :value + replacement-value)
          (assoc (vec new-board) random-index z))))))

(defn evaluate-move
  "Return a new version of the board that has been altered according
  to the player input and the game's rules."
  [current-board key-pressed score-callback]
  (as-> current-board b
    (resolve-input b key-pressed score-callback)
    (upgrade-random-zero current-board b)))

(defn win?
  "Returns true if the board has any squares with value 2048, and false otherwise."
  [board]
  (some? (seq (filter #(= 2048 (:value %)) board))))

(defn game-over?
  "Returns true if there are no more possible moves, false otherwise."
  [board]
  (let [up-board (resolve-input board up (constantly nil))
        down-board (resolve-input board down (constantly nil))
        left-board (resolve-input board left (constantly nil))
        right-board (resolve-input board right (constantly nil))]
    (= up-board down-board left-board right-board)))






(def default-db
  {:board (n-board-elements 16 0)
   :score 0
   :playing false
   :game-over false
   :victory false})

(defn start-game [db]
  (assoc db
         :board (initial-board)
         :score 0
         :playing true
         :game-over false
         :victory false))

(defn end-game [db win?]
  (assoc db
         :playing false
         :game-over true
         :victory win?))



(defn score [{:keys [score]}]
  [:p.score "Score: " score])

(defn new-game-button [title]
  [:p.control
   [:button.button.is-danger.is-large
    {:onclick  [:app.events/game-started]}
    title]])

(defn control-panel [{:keys [playing] :as state}]
  [:div.control-panel.is-clearfix
   [:div.game-title
    "2048"]
   (if playing
     [:div.game-controls
      (score state)]
     [:div.game-controls
      (new-game-button "Play Game")])])

(defn square [segment]
  [:div {:key (:key segment)
         :class (str "square square-" (:value segment))}
   (when (and (number? (:value segment)) (< 0 (:value segment)))
     (:value segment))])

(defn instructions [{:keys [playing]}]
  (when (not playing)
    [:div.notification.instructions
     [:p.header "To Play:"]
     [:p "Click the \"Play Game\" button above."]
     [:p "Then, move the tiles with your arrow keys."]
     [:br]
     [:p "The object of the game is to make a single tile with the value '2048' in it."]]))

(defn board [{:keys [board] :as state}]
  [:div.board-surround
   [:div
    {:class "board board-flip"}
    (for [segment board]
      (square segment))]
   [:div.board.board-bg
    (for [segment (n-board-elements 16 "bg")]
      (square segment))]
   (instructions state)])

(defn game-over-overlay [{:keys [victory gameover score]}]
  (let [winner victory]
    [:div
     {:class (str "modal"
                  (if gameover
                    " is-active"
                    ""))}
     [:div.modal-background]
     [:div.modal-content.game-over-modal
      [:div.box
       [:p.header (if winner "You won!" "Game Over")]
       [:p (str "Your Score: " score)]
       [:p (if winner
             "Studies show that winning is good for you.  Cheers to your health!"
             "You'll get it eventually.  Keep trying!")]
       [:br]
       [:p
        [:button.button.is-success
         {:onclick [:app.events/game-over-acknowledged]}
         "Word"]]]]
     [:button.modal-close.is-large
      {:onclick  [:app.events/game-over-acknowledged]}]]))





(defn view [state]
  [:body {:onkeydown [:app.events/on-keydown]}
   [:link {:rel "stylesheet" :href "/twenty/main.css"}]
   [:div
    (control-panel state)
    (board state)
    (game-over-overlay state)]])
#_(view @state)



(def state (atom default-db))

;; Silly trick for mutually recursive things
(def event-handler)


(defn handle-points-added
  "Updates the score when the player causes tiles to be merged."
  [db score]
  (update db :score + score))

(defn handle-keypress
  "Updates the game state based on an arrow key being pressed."
  [db {:keys [key]} points-added-callback]
  (if (arrows key)
    (assoc db
           :board
           (evaluate-move (:board db)
                          key
                          points-added-callback))
    db))

(defn handle-endgame-monitor-tick
  "Checks for endgame conditions and ends the game if met."
  [state]
  (let [{:keys [board playing]} @state
        win? (win? board)
        game-over? (game-over? board)]
    (when (and playing (or win? game-over?))
      (swap! state end-game win?))))

(defn event-handler [{:keys [action]}]
  ;; points-added exists becasue the original code had side-effects on
  ;; dispatch, which if I represented naively would have been
  ;; side-effects in a swap!. The side-effect was to also swap!. That
  ;; meant that we were in an infinite loop situation.
  (let [points-added (atom [])]
    (let [[action-type payload] action]
      (case action-type
        :app.events/game-started (swap! state start-game)
        :app.events/on-keydown (do
                                 (swap! state handle-keypress payload #(swap! points-added conj %))
                                 (doseq [points @points-added]
                                   (event-handler {:action [:app.events/points-added points]}))
                                 (reset! points-added []))
        :app.events/points-added (swap! state handle-points-added payload)
        (println "Unhandled Action" action))))
  (handle-endgame-monitor-tick state))

(defonce live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 1117}))
