(ns account-number.main
  (:require [account-number.core :as core]
            [clojure.string :as string]
            [clojure.java.io :as io]))

(defn scenario-1 [content]
  (->> content
       core/split-into-rows
       (map core/rows->seven-segment)
       (map core/seven-segment->account-number)
       (map string/join)
       (string/join "\n")))

(defn scenario-2 [content]
  (->> content
       core/split-into-rows
       (map core/rows->seven-segment)
       (map core/seven-segment->account-number)
       (map core/valid-account-number?)
       (string/join "\n")))

(defn scenario-3 [content]
  (->> content
       core/split-into-rows
       (map core/rows->seven-segment)
       (map core/seven-segment->account-number)
       (map core/format-output)
       (string/join "\n")))

(def well-formed (io/resource "well-formed.txt"))
(def ill-formed (io/resource "ill-formed.txt"))

(def scenarios
  {"scenario-1" {:scenario scenario-1 :file well-formed}
   "scenario-2" {:scenario scenario-2 :file well-formed}
   "scenario-3" {:scenario scenario-3 :file ill-formed}})

(defn -main 
  ([command]
   (-main command (-> command scenarios :file)))
  ([command file]
   (if-let [{:keys [scenario]} (scenarios command)]
     (println (scenario (slurp file)))
     (println "Please enter a valid scenario [scenario-1 scenario-2 scenario-3]"))))

(-main "scenario-3")
