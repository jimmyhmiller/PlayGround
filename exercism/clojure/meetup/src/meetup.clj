(ns meetup
  (:require [clj-time.core :as t]
            [clj-time.predicates :as pr]))

(defn add-day [date]
  (t/plus date (t/days 1)))

(defn sub-day [date]
  (t/minus date (t/days 1)))

(def directions 
  {:asc add-day
   :desc sub-day})

(defn seek-to-date [pred direction date]
  (let [iterator (directions direction)]
    (->> date
         (iterate iterator)
         (drop-while (complement pred)))))

(defn find-first [pred date]
  (->> date
       (seek-to-date pred :asc)
       first))

(defn last-day-of-month [date]
  (-> date
      (t/plus (t/months 1))
      sub-day))

(defn find-last [pred date]
  (->> date
       last-day-of-month
       (seek-to-date pred :desc)
       first))

(defn find-nth [n pred date]
  (let [date-finder (comp add-day (partial find-first pred))]
    (->> date
         (iterate date-finder)
         (drop n)
         first
         sub-day)))

(defn teenth? [date]
  (< 12 (t/day date) 20))

(defn find-teenth [pred date]
  (->> date
       (find-first teenth?)
       (find-first pred)))

(def day->pred 
  {:sunday pr/sunday?
   :monday pr/monday?
   :tuesday pr/tuesday?
   :wednesday pr/wednesday?
   :thursday pr/thursday?
   :friday pr/friday?
   :saturday pr/saturday?})

(def description->finder
  {:first (partial find-nth 1)
   :second (partial find-nth 2)
   :third (partial find-nth 3)
   :fourth (partial find-nth 4)
   :last find-last
   :teenth find-teenth})

(defn date->tuple [date]
  [(t/year date) (t/month date) (t/day date)])

(defn meetup [month year day description]
  (let [date (t/date-time year month)
        pred (day->pred day)
        finder (description->finder description)]
    (->> date
         (finder pred)
         date->tuple)))


