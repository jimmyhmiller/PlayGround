(ns calories.core
  (:require [cheshire.core :as json])
  (:import (java.time LocalDate)
           (java.time.format DateTimeFormatter))
  (:gen-class))

(set! *warn-on-reflection* true)


(def ^:dynamic dry-run false)

(def file-path (str (System/getProperty "user.home") "/.calories.json"))

(def pound 3500)
(def base-rate 2340)
(def daily 1840)

(defn write-file [file content]
  (if dry-run
    (println content)
    (spit file (json/generate-string content))))

(defn read-file [file]
  (try
    (json/parse-string (slurp file) true)
    (catch Exception e
      (write-file file []))))

(defn format-date [^LocalDate local-date]
  (.format
   local-date
   DateTimeFormatter/ISO_LOCAL_DATE))

(defn today []
  (format-date (LocalDate/now)))

(defn yesterday []
  (format-date (.minusDays (LocalDate/now) 1)))

(defn make-record [date calories]
  {:date date
   :calories calories})

(defn add-record [file calories]
  (write-file file (conj (read-file file) (make-record (today) calories))))

(defn calories-for-date [date]
  (->> file-path
       read-file
       (filter (comp #{date} :date))
       (map :calories)))

(defn total [date]
  (reduce + 0 (calories-for-date date)))

(defn calories-left [date]
  (- daily (total date)))

(defmulti command (comp keyword first))

(defmethod command :add [[_ calories]]
  (add-record file-path (Integer/parseInt calories))
  (println (str "Added " calories " calories"))
  (println (str (calories-left (today)) " calories left")))

(defmethod command :today [_]
  (println (str (calories-left (today)) " calories left")))

(defmethod command :yesterday [_]
  (println (str (calories-left (yesterday)) " calories left")))

(defmethod command :date [[_ ^String date]]
  (println (str (calories-left date) " calories left")))

(defn number-of-days [contents]
  (count (group-by :date contents)))

(defn calories-overall [contents]
  (reduce + 0 (map :calories contents)))

(defn expected-calories [contents]
  (* daily (number-of-days contents)))

(defmethod command :pounds [_]
  (let [contents (read-file file-path)
        total-calories (calories-overall contents)
        base-calories (* base-rate (number-of-days contents))
        pounds (float (/ (- base-calories total-calories)
                  pound))]
    (println (str pounds " projected to have been lost so far"))))

(defn categorize [pred true-label false-label coll]
  (->> coll
       (group-by pred)
       (map (juxt (comp {true true-label false false-label} first) second))
       (into {})))

(defmethod command :breakdown [[_ date]]
  (let [calories (calories-for-date date)
        {:keys [exercise food]} (categorize pos? :food :exercise calories)
        ^Integer total-exercise (reduce + 0 exercise)
        total-food (reduce + 0 food)]
    (println (str total-food " calories eaten"))
    (println (str (Math/abs total-exercise) " calories burned"))
    (println (str  (+ total-exercise total-food) " net calories"))
    (println (str (- base-rate (+ total-exercise total-food)) " calorie deficit"))))

(defmethod command :extra [_]
  (let [contents (read-file file-path)
        diff (- (expected-calories contents)
                (calories-overall contents))]
    (println (str "You have " diff " extra calories"))))

(defmethod command :default [args]
  (if-let [calories (first args)]
    (command [:add calories])
    (command [:today])))

(defn -main [& args]
  (command args))


;; Notes
;; Consider passing in the file content
