(ns calories.core
  (:require [cheshire.core :as json])
  (:import (java.text SimpleDateFormat))
  (:gen-class))

(def file-path "/Users/jimmy/.calories.json")

(defn write-file [file content]
  (spit file (json/generate-string content)))

(defn read-file [file]
  (try 
    (json/parse-string (slurp file) true)
    (catch Exception e
      (write-file file []))))

(defn today []
  (.format
   (SimpleDateFormat. "yyyy-MM-dd")
   (java.util.Date.)))

(defn make-record [date calories]
  {:date date
   :calories calories})

(defn add-record [file calories]
  (write-file file (conj (read-file file) (make-record (today) calories))))

(defn total-today []
  (->> file-path
       read-file
       (filter (comp #{(today)} :date))
       (map :calories)
       (reduce + 0)))

(defn -main [& args]
  (let [calories (first args)]
    (if calories
      (do
        (add-record file-path (Integer/parseInt calories))
        (println (str "Added " calories " calories")))
      (println (total-today)))))

