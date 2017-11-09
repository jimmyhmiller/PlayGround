; write a function that takes a number and returns an array of its prime factors.
; write a function that takes a string and returns an array of characters that appear more than once.
; write a function that checks if a string is a pallindrome.
; write a function that takes an array of numbers and adds one to each number.
; write a function that takes an array of numbers and returns the largest and smallest number and the index of each in the array

(defn prime-factors
  ([n]
   (prime-factors n 2))
  ([n factor]
   (cond
     (<= n 1) '()
     (zero? (mod n factor)) (cons factor (prime-factors (/ n factor) factor))
     :else (recur n (inc factor)))))

(defn get-duplicates [coll]
  (let [freqs (frequencies (seq coll))]
    (filter #(> (freqs %) 1) coll)))

(defn palindrome? [coll]
  (= (reverse coll) (seq coll)))

(defn add-one [coll]
  (map #(+ % 1) coll))

(defn min-index [coll]
  (->> coll
       (map-indexed vector)
       (apply min-key second)
       first))

(defn max-index [coll]
  (->> coll
       (map-indexed vector)
       (apply max-key second)
       first))
