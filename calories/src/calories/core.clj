(ns calories.core
  (:require [cheshire.core :as json])
  (:import (java.time LocalDate)
           (java.time.format DateTimeFormatter))
  (:gen-class))

(set! *warn-on-reflection* true)


(def ^:dynamic dry-run false)

(def file-path (str (System/getProperty "user.home") "/.calories.json"))

(def pound 3500)
(def base-rate 2500)
(def daily 1500)

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

(defn total [date]
  (->> file-path
       read-file
       (filter (comp #{date} :date))
       (map :calories)
       (reduce + 0)))

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

(defmethod command :default [args]
  (if-let [calories (first args)]
    (command [:add calories])
    (command [:today])))

(defn -main [& args]
  (command args))

