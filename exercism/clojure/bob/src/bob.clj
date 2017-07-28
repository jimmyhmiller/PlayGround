(ns bob
  (:require [clojure.string :as string]))


(defn every-not-empty? [pred coll]
  (if (empty? coll)
    false
    (every? pred coll)))

(defn yelling? [message]
  (->> message
       (filter #(Character/isLetter %))
       (every-not-empty? #(Character/isUpperCase %))))

(defn question? [message]
  (string/ends-with? message "?"))

(defn said-nothing? [message]
  (string/blank? message))

(defn response-for [message]
  (cond 
    (yelling? message) "Whoa, chill out!"
    (question? message) "Sure."
    (said-nothing? message) "Fine. Be that way!"
    :else "Whatever."))
