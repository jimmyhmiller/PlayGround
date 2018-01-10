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

(defn build-default-suggestion [command args]
  {:type :default (first (:args command)) args})

(defn get-commands-with-suggestions [text]
  (let [commands (get-commands text)
        args (apply str (rest (string/split text #" ")))
        suggestions (map #(get-suggestions % args) commands)
        result (mapcat (fn [command suggestions] 
                  (map vector (repeat command) 
                       (if (empty? suggestions) 
                         (list (build-default-suggestion command args)) 
                         suggestions))) 
                       commands suggestions)]
    (if (empty? result)
      '([nil nil])
      result)))


(defn get-executor [{command-name :name} {:keys [type] :or {type :default}}]
  (get-in @state [:executors command-name type :executor] identity))

(defn execute-command [command suggestion]
  (when (and (not (nil? command))
             (not (nil? suggestion)))
    ((get-executor command suggestion) suggestion)))

(get-executor {:name :quit} {:type :default})

(register-command 
 {:name :open
  :args [:target]})

(register-command
 {:name :google
  :args [:search]})

(register-command
 {:name :zoom
  :args [:person]})

(register-command 
 {:name :help})

(register-command
 {:name :quit})

(register-executor
 {:command :quit
  :type :default
  :executor (fn [_] (System/exit 0))})

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

(def zoom-users 
  (->
   (sh "/Users/jimmy/Documents/Code/dev-environment/bin/zoom" "shortlist")
   :out
   (string/split #" ")))

(defn get-zoom-users [text]
  (->> zoom-users
       (filter #(string/includes? % text))
       (sort-by #(string/index-of % text))
       (map (fn [person] {:type :default :person person}))))

(register-suggestor
 {:command :zoom
  :suggestor get-zoom-users})

(register-executor
 {:command :zoom
  :type :default
  :executor (fn [{:keys [person]}] 
              (sh "/Users/jimmy/Documents/Code/dev-environment/bin/zoom" "join" person))})

(register-executor
 {:command :google
  :type :default
  :executor (fn [{:keys [search]}] (sh "open" (str "https://www.google.com/search?q=" search)))})

(register-executor
 {:command :help
  :type :default
  :executor (fn [suggestion] (println "No help to give"))})
