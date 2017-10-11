;; This buffer is for text that is not saved, and for Lisp evaluation.
;; To create a file, visit it with C-x C-f and enter text in its buffer.

(def num-players 5)
(def num-dice 7)


(defn roll []
  (+ (rand-int 6) 1))

(defn player [dice] 
  (set 
   (for [n (range dice)]
     (roll))))

(defn players [num-players dice]
  (for [i (range num-players)]
    (player dice)))

(defn run-scenario [n goal]
  (frequencies 
   (for [i (range n)]
     (goal (players num-players num-dice)))))

(defn first-goal [rolls]
  (some #(clojure.set/subset? #{5 6} %) rolls))

(defn second-goal [rolls]
  (every? #(or (contains? % 5) (contains? % 6)) rolls))


(run-scenario 100000 first-goal)
(run-scenario 100000 second-goal)
