(ns wander.core20)




;  Given a number "x" and a sorted array of coins "coinset", write a function
;  that returns the amounts for each coin in the coinset that sums up to X or
;  indicate an error if there is no way to make change for that x with the given
;  coinset. For example, with x=7 and a coinset of [1,5,10,25], a valid answer
;  would be {1: 7} or {1: 2, 5: 1}. With x = 3 and a coinset of [2,4] it should
;  indicate an error. Bonus points for optimality.

 ; Use the following examples to test it out

; A. x = 6 coinset = [1,5,10,25]
; B. x = 6, coinset = [3,4]
; C. x = 6, coinset = [1,3,4]
; D. x = 6, coinset = [5,7]




;; So I started this problem and got a bit flustered as an assumption
;; I was implicitly making didn't hold true.
;; I had done something similar to this but using us currency instead
;; of an arbitrary one. US currency has a nice property where the
;; "greedy" solution here works out well. Basically you take the
;; highest denomination coin and just keep subtracting.

;; This is not that problem and I confused it with that one. So, we
;; have to be more clever here. There is no simple linear time
;; algorithm to do this and I had kind of gone into it assuming there
;; was.

;; We need to backtrack at some point or other because we can make the
;; wrong choice and have to correct it.


(defn coin-count [coll]
  (if coll
    (reduce + (vals coll))
    ##Inf))

;; Ugly hack around lack of mutual recursion in clj
(def make-change-helper)

(defn make-change-helper* [amount-remaining coinset]
  (let [coin (first coinset)]
    (cond
      (nil? coin) nil

      (neg? amount-remaining)
      nil
      
      (zero? amount-remaining) {}

      ;; We are going to make two attempts here.
      ;; First we assume we can use the current coin.
      ;; We try that out recursively, if it worked, we update the count for that coin.
      ;; But we also try not using the current coin.
      ;; Finally we compare the coin count of the two approaches to find the minimal set.
      ;; Without this minmal thing, we could instead try the current coint, and if that doesn't
      ;; work, only then do we do the recursive call without the current count.
      :else (let [make-change-1 (make-change-helper (- amount-remaining (first coinset)) coinset)
                  make-change-1 (when make-change-1 (clojure.core/update make-change-1 coin (fnil inc 0)))
                  make-change-2 (make-change-helper amount-remaining (rest coinset))]
              (min-key coin-count make-change-1 make-change-2)))))

;; Memoize so we don't go down the same path multiple times
;; Trades space for speed.
(def make-change-helper (memoize make-change-helper*))

(defn make-change [x coinset]
  (if-let [result (make-change-helper x coinset)]
    result
    "error"))

(make-change 30 [4 9])
(make-change 6 [3 4])
(make-change 6 [1 3 4])
(make-change 6 [5 7])
(make-change 6249 [186, 419, 83, 408])

;; Quick test code to see that my numbers really add up
(defn check-answer [amount coinset]
  (let [answer (make-change amount coinset)]
    (if (not= answer "error")
      (reduce (fn [sum [k v]] (+ sum (* k v))) 0 answer)
      answer)))

(check-answer 30 [4 9])
(check-answer 6249 [186, 419, 83, 408])
(check-answer 6 [1 3 4])
