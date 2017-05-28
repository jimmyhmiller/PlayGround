(ns beer-song
  (:require [clojure.string :refer [join]]))

(defn pluralize [word num]
  (if (= num 1)
    word
    (str word "s")))

(def verse-0 
  (str "No more bottles of beer on the wall, no more bottles of beer.\n"
       "Go to the store and buy some more, 99 bottles of beer on the wall.\n"))

(def verse-1
  (str "1 bottle of beer on the wall, 1 bottle of beer.\n"
       "Take it down and pass it around, no more bottles of beer on the wall.\n"))


(defn verse [num]
  (cond 
    (= num 0) verse-0
    (= num 1) verse-1
    :else (str num 
               " " (pluralize "bottle" num)
               " of beer on the wall, "
               num 
               " " (pluralize "bottle" num)
               " of beer.\nTake one down and pass it around, "
               (dec num)
               " " (pluralize "bottle" (dec num))
               " of beer on the wall.\n")))

(defn sing 
  ([start] (sing start 0))
  ([start end]
   (->> (range start (dec end) -1)
        (map verse)
        (join "\n"))))
