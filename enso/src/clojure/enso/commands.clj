(ns enso.commands
  (:require [clojure.string :as string]))

(def state (atom {:commands (set [])
                  :suggestors {}
                  :executors {}}))

(defn register-command [command]
  (swap! state update :commands conj command))

(defn register-suggestor [suggestor]
  (swap! state update-in 
         [:suggestors (:command suggestor)] 
         (fnil conj []) suggestor))

(defn register-executor [executor]
  (swap! state assoc-in
         [:executors (:command executor) (:type executor)] 
         executor))

(defn get-commands [text]
  (let [command (first (string/split text #" "))]
    (if (empty? text)
      '()
      (filter #(string/starts-with? (name (:name %)) command) (:commands @state)))))

(defn get-suggestions [command text]
  (let [suggestors (-> @state :suggestors command)]
    (mapcat (fn [{:keys [suggestor]}] (suggestor text)) suggestors)))

(defn get-executor [command {:keys [type]}]
  (->> @state :executors command type :executor))

(defn execute-command [command suggestion]
  ((get-executor command suggestion) suggestion))

(register-command 
 {:name :open
  :args [:target]})

(register-command 
 {:name :help})

(register-suggestor
 {:command :open
  :suggestor (fn [text] [{:type :application
                           :target (str text " thing")}])})

(register-suggestor
 {:command :open
  :suggestor (fn [text] [{:type :application
                           :target (str text " other thing")}])})

(register-executor
 {:command :open
  :type :application
  :executor (fn [suggestion] (println "Opening " suggestion))})
