(ns atbash-cipher
  (:require [clojure.string :refer [join replace lower-case]]))

(def cipher
  {\a \z
   \b \y
   \c \x
   \d \w
   \e \v
   \f \u
   \g \t
   \h \s
   \i \r
   \j \q
   \k \p
   \l \o
   \m \n
   \n \m
   \o \l
   \p \k
   \q \j
   \r \i
   \s \h
   \t \g
   \u \f
   \v \e
   \w \d
   \x \c
   \y \b
   \z \a})

(defn encode [message]
  (let [cleaned-message (replace message #"[\p{Punct} ]" "")]
    (->> cleaned-message
         lower-case
         (map #(get cipher % %))
         (filter identity)
         (partition 5 5 [])
         (map (partial join ""))
         (join " "))))
