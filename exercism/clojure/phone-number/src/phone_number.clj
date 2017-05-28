(ns phone-number
  (:require [clojure.string :refer [replace join starts-with?]]))

(defn only-numbers [num]
  (replace num #"[^0-9]" ""))

(defn full-us? [num]
  (and (= (count num) 11)
           (starts-with? num "1")))

(defn valid? [num]
  (= (count num) 10))

(defn strip-extra-1 [num]
  (if (full-us? num)
    (subs num 1)
    num))

(defn clean-num [num]
  (-> num
      only-numbers
      strip-extra-1))

(defn number [num]
  (let [cleaned-num (clean-num num)]
    (if (valid? cleaned-num)
      cleaned-num
      "0000000000")))

(defn area-code [num]
  (->> num
       only-numbers
       strip-extra-1
       (take 3)
       (join "")))

(defn pretty-print [num]
  (let [cleaned-num (clean-num num)]
    (str "(" (subs cleaned-num 0 3) ") "
         (subs cleaned-num 3 6) "-" 
         (subs cleaned-num 6))))

