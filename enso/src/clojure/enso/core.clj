(ns enso.core
  (:import org.jnativehook.GlobalScreen)
  (:import java.util.concurrent.AbstractExecutorService)
  (:import org.jnativehook.keyboard.NativeKeyListener)
  (:import org.jnativehook.keyboard.NativeKeyEvent)
  (:import org.jnativehook.NativeInputEvent)
  (:import java.util.logging.Level)
  (:import java.util.logging.Logger)
  (:import java.awt.Component))

(use '[clojure.java.shell :only [sh]])
(use 'seesaw.core)
(use 'seesaw.dev)
(use 'seesaw.color)
(use 'seesaw.border)
(use 'seesaw.util)
(use 'seesaw.font)
(require '[seesaw.bind :as bind])
(require '[seesaw.graphics :as g])
(require '[clojure.string :as string])

(def state (atom
            {:active false
             :command-text ""}))



(add-watch state :test
           (fn [key atom old-state new-state]
             (when (not= old-state new-state)
               (println "changed" new-state))))

(def l (label
        :text "Welcome to Enso! Enter a command, or type \"help\" for assistance."
        :background "#ABC26B"
        :minimum-size [275 :by 100]
        :foreground :white
        :font "Gentium-32"
        :border (empty-border :thickness 5)))

(def input-label (label
                  :background "#000"
                  :foreground "#fff"
                  :font "Gentium-32"
                  :border (empty-border :thickness 5)
                  :visible? false))


(let [logger (Logger/getLogger (.getName (.getPackage GlobalScreen)))]
  (.setLevel logger Level/WARNING)
  (.setUseParentHandlers logger false))


(bind/bind state (bind/transform :command-text) (bind/property input-label :text))
(bind/bind state (bind/transform #(not (zero? (count (% :command-text))))) (bind/property input-label :visible?))


(defn change-text [key-text command-text]
  (println key-text command-text)
  (string/lower-case (str command-text key-text)))


(defn colorize-text [text match]
  (str "<html><font color=white>" text "</font>" (subs match (count text)) "</html>"))


(def applications
  (.listFiles(clojure.java.io/file "/Applications")))

(def open-suggestions
  (->> applications
       (map #(.getName %))
       (filter #(string/ends-with? % ".app"))
       (map #(string/replace-first % #"\.[^.]+$" ""))
       (map #(str "open " %))
       (concat ["open terminal"])
       sort))


(defn match-suggestion [text suggestion]
  (let [words (string/split text #" ")]
    (and (string/starts-with? suggestion (first words))
         (every? #(string/includes? (string/lower-case suggestion) %) (rest words)))))


(defn split-word [word suggestion]
  (let [word-location (clojure.string/index-of suggestion word)]
    (filter seq
            [(subs suggestion 0 word-location)
             [:match (subs suggestion word-location (+ (count word) word-location))]
             (subs suggestion (+ word-location (count word)))])))

(defn split-words [words suggestion]
  (if (empty? words)
    [suggestion]
    (let [new-suggestion (split-word (first words) suggestion)]
      (concat (butlast new-suggestion) (split-words (rest words) (last new-suggestion))))))


(defn generic-label [text color]
  (label
   :text text
   :background "#000"
   :foreground color
   :font "Gentium-32"))

(defn match-label [match]
  (if (string? match)
    (generic-label match "#ABC26B")
    (generic-label (last match) :white)))


(defn suggestions [text]
  (let [suggs open-suggestions]
    (if (empty? text)
      []
      (take 15 (filter (partial match-suggestion text) suggs)))))


(defn suggestion->label [text suggestion]
  (let [words (string/split text #" ")
        matches (split-words words (string/lower-case suggestion))]
    (horizontal-panel
     :background "#000"
     :border (empty-border :thickness 5)
     :items (map match-label matches))))


(def f
  (doto (frame :undecorated? true)
    (.setOpacity (float 0.8))
    (.setLocation 0 20)
    (.setAlwaysOnTop true)
    (.setBackground (color 0 0 0 0))))


(defn update-font! [container font-face]
  (doseq [label (select container [:JLabel])]
    (.setFont label (font font-face)))
  container)


(defn left-align [c]
  (doto c
    (.setAlignmentX Component/LEFT_ALIGNMENT)))

(defn default-labels [state]
  (let [labels [l]
        new-suggestions (suggestions (:command-text state))
        suggestion-labels (map (partial suggestion->label (:command-text state)) new-suggestions)
        input (if (zero? (count suggestion-labels)) [input-label] [])
        new-labels (into [] (map left-align (concat labels input suggestion-labels)))]
    (vertical-panel :background (color 0 0 0 0) :items (update-in new-labels [1] update-font! "Gentium-64"))))

(bind/bind state (bind/transform default-labels) (bind/property f :content))


(defn get-windows []
  (->> (sh "osascript" "-e" "set allWindows to \"[\"

           tell application \"System Events\"
           repeat with p in (processes whose visible is true)
           set wins to \"[\"
           repeat with w in windows of p
           set wins to wins & \"\\\"\" & name of w & \"\\\" \"
           end repeat
           set wins to wins & \"]\"
           set groupings to \"{:window \" & wins & \" :process \\\"\" & name of p & \"\\\"}\"
           set allWindows to allWindows & {groupings}
           end repeat
           end tell

           allWindows & \"]\"
           ")
       :out
       (read-string)
       (filter #(not (empty? (:window %))))))

(defn showit []
  (-> f
      pack!
      show!))

(defn hide [e]
  (let [k (.getRawCode e)]
    (when (= k 53)
      (reset! state {:active false :command-text ""}))))


(add-watch state :show
           (fn [key atom old-state new-state]
             (when (not= old-state new-state)
               (if (:active new-state)
                 (showit)
                 (hide! f)))))

(add-watch state :command
           (fn [key atom old-state new-state]
             (when (not (:active new-state))
               (println "command!!" old-state new-state)
               (let [command (first (suggestions (:command-text old-state)))]
                 (when (and command (string/starts-with? command "open"))
                   (println "opening")
                   (sh "open" "-a" (subs command 5)))))))

(defn log-key [event-type e]
  (when (:active @state)
    (let [f (.getDeclaredField NativeInputEvent "reserved")]
      (.setAccessible f true)
      (.setShort f e (short 0x01)))
    (let [k (.getRawCode e)
          key-text (NativeKeyEvent/getKeyText (.getKeyCode e))]
      (when (not (= k 53))
        (println "log-key" k))
      (when (= k 49)
        (swap! state update :command-text (partial change-text " ")))
      (when (= k 51)
        (swap! state update :command-text #(apply str (butlast %))))
      (when (and
             (= (count key-text) 1)
             (or
              (= key-text " ")
              (Character/isUpperCase (first (seq (char-array key-text))))))
        (pack! f)
        (swap! state update :command-text (partial change-text key-text)))))
  (let [k (.getRawCode e)]
    (when (= k 53) (swap! state assoc :active true))))


(defn voidDispatchService []
  (let [running (atom true)]
    (proxy
      [AbstractExecutorService] []
      (shutdown [] (reset! running false))
      (shutdownNow []
                   (reset! running false)
                   (new java.util.ArrayList 0))
      (isShutdown [] (not @running))
      (isTerminated [] (not @running))
      (awaitTermination [timeout unit] true)
      (execute [r] (.run r)))))


(defn myGlobalKeyListener []
  (reify
    NativeKeyListener
    (nativeKeyTyped [this event] #(%))
    (nativeKeyPressed [this event] (log-key "pressed" event))
    (nativeKeyReleased [this event]
                       (hide event))))


(defn -main [& args]
  (GlobalScreen/setEventDispatcher (voidDispatchService))
  (GlobalScreen/registerNativeHook)
  (GlobalScreen/addNativeKeyListener (myGlobalKeyListener)))

;(-main)
