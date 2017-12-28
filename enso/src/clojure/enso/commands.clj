(ns enso.commands
  (:require [clojure.string :as string]
            [clojure.java.shell :refer [sh]]))

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

(defn get-suggestions [{command-name :name :or {command-name nil}} text]
  (if (or (nil? command-name) (empty? text))
    '()
    (let [suggestors (-> @state :suggestors command-name)]
      (mapcat (fn [{:keys [suggestor]}] (suggestor text)) suggestors))))

(defn get-commands-with-suggestions [text]
  (let [commands (get-commands text)
        args (apply str (rest (string/split text #" ")))
        suggestions (map #(get-suggestions % args) commands)
        result (mapcat (fn [command suggestions] 
                  (map vector (repeat command) 
                       (if (empty? suggestions) '(nil) suggestions))) 
                       commands suggestions)]
    (if (empty? result)
      '([nil nil])
      result)))


(defn get-executor [{command-name :name} {:keys [type] :or {type :default}}]
  (->> @state :executors command-name type :executor))

(defn execute-command [command suggestion]
  (when (and (not (nil? command))
             (not (nil? suggestion)))
    ((get-executor command suggestion) suggestion)))

(register-command 
 {:name :open
  :args [:target]})

(register-command 
 {:name :help})

(def applications
  (.listFiles (clojure.java.io/file "/Applications")))


(def open-suggestions
  (->> applications
       (map #(.getName %))
       (filter #(string/ends-with? % ".app"))
       (map #(string/replace-first % #"\.[^.]+$" ""))
       (concat ["terminal"])
       (map string/lower-case)
       sort))


(defn build-suggestion-open [suggestion]
  {:type :application
   :target suggestion})

(defn open-suggestor [text] 
  (if (empty? text)
    []
    (->> open-suggestions
         (filter #(string/includes? % text))
         (sort-by #(string/index-of % text))
         (map build-suggestion-open)
         (take 10))))


(register-suggestor
 {:command :open
  :suggestor open-suggestor})

(register-executor
 {:command :open
  :type :application
  :executor (fn [{:keys [target]}] (sh "open" "-a" target))})

(register-executor
 {:command :help
  :type :default
  :executor (fn [suggestion] (println "No help to give"))})
