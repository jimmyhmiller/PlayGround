(ns enso.refactor
  (:import [org.jnativehook GlobalScreen]
           [java.util.concurrent AbstractExecutorService]
           [org.jnativehook.keyboard NativeKeyListener NativeKeyEvent]
           [org.jnativehook NativeInputEvent]
           [java.util.logging Level Logger]
           [java.awt Component GraphicsEnvironment])
  (:require [seesaw.bind :as bind]
            [seesaw.graphics :as g]
            [clojure.string :as string]
            [clojure.java.shell :refer [sh]]
            [seesaw.core :refer 
             [label
              frame 
              show! 
              pack! 
              hide! 
              vertical-panel 
              horizontal-panel 
              repaint!]]
            [seesaw.border :refer [empty-border]]
            [seesaw.font :refer [font]]
            [seesaw.color :refer [color]]
            [enso.commands :as commands]
            [enso.parse :as parse]))

(def ^:const green  "#ABC26B")


(defn create-help-label [text]
  (label
   :text text
   :background green
   :minimum-size [275 :by 100]
   :foreground :white
   :font "Gentium Plus-32"
   :border (empty-border :thickness 5)))

(def standard-help-text 
  "Welcome to Enso! Enter a command, or type \"help\" for assistance.")

(defn create-input-label []
  (label
   :background "#000"
   :foreground "#fff"
   :font "Gentium Plus-64"
   :border (empty-border :thickness 5)
   :visible? false))

(defn generic-label
  ([text color font-size font-style]
   (label
    :text text
    :background "#000"
    :foreground color
    :font (font :nme "Gentium Plus" :style font-style :size font-size)))
  ([text color font-size]
   (generic-label text color font-size #{})))

(defn match-label [font-size match]
  (if (string? match)
    (generic-label match "#ABC26B" font-size #{:italic})
    (generic-label (last match) :white font-size)))

(defn draw-window []
  (doto (frame :undecorated? true)
    (.setOpacity (float 0.85))
    (.setLocation 0 20)
    (.setAlwaysOnTop true)
    (.setBackground (color 0 0 0 0))))


(defn capture-keys [event]
  (doto (.getDeclaredField NativeInputEvent "reserved")
  (.setAccessible true)
  (.setShort event (short 0x01))))

(defn handle-key-press [event callback]
  (callback (.getRawCode event)
            (NativeKeyEvent/getKeyText (.getKeyCode event)) event))

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

(defn start-key-logger [listener]
  (GlobalScreen/setEventDispatcher (voidDispatchService))
  (GlobalScreen/registerNativeHook)
  (GlobalScreen/removeNativeKeyListener listener)
  (GlobalScreen/addNativeKeyListener listener))


;;;;;;;;;;;;;;;;; MAIN



(def state (atom {:active false
                  :command-text ""}))
(def input-label (create-input-label))
(def help-label (create-help-label standard-help-text))
(def window (draw-window))

(bind/bind state
           (bind/transform :command-text) 
           (bind/property input-label :text))

(bind/bind state 
           (bind/transform #(not (empty? (% :command-text)))) 
           (bind/property input-label :visible?))

(add-watch state :show
           (fn [key atom old-state new-state]
             (cond 
               (and (:active new-state) 
                    (not (:active old-state))) (-> window pack! show!)
               (and (not (:active new-state))
                    (:active old-state)) (-> window hide!))))

(remove-watch state :update)

(add-watch state :commands
           (fn [key atom old-state new-state]
             (let [text (:command-text new-state)
                   command (first (commands/get-commands text))
                   suggestion (first (commands/get-suggestions command text))]
               (println text)
               (println (parse/parse text command suggestion)))))



(defmulti render-parsed first)

(defmethod render-parsed :line [[_ & children]]
  (horizontal-panel
   :background :black
   :border (empty-border :thickness 5)
   :items (map render-parsed children)))

(defmethod render-parsed :match [[_ value]]
  (generic-label value :white 64))

(defmethod render-parsed :unmatch [[_ value]]
  (generic-label value green 64))

(defmethod render-parsed :arg [[_ value]]
  (generic-label value :grey 64))


(defn command->label [{command-name :name
                       args :args}]
  (horizontal-panel 
   :background :black
   :border (empty-border :thickness 5)
   :items (cons (generic-label (name command-name) green 64)
                (map #(generic-label (str " " (name %)) :gray 64) args))))

(defn left-align [c]
  (doto c
    (.setAlignmentX Component/LEFT_ALIGNMENT)))

(defn command-labels [text]
  (let [commands (commands/get-commands-with-suggestions text)]
    (vertical-panel
     :background :black
     :items (->> commands
                 (map (fn [[command suggestion]] 
                        (parse/parse text command suggestion)))
                 (map render-parsed)
                 (cons help-label)
                 (map left-align)))))

(bind/bind state
           (bind/transform #(command-labels (:command-text %)))
           (bind/property window :content))

(defn change-text [command-text key-text]
  (string/lower-case (str command-text key-text)))

(defn valid-character? [text]
  (Character/isUpperCase (first (seq (char-array text)))))

(defn determine-command-update [command code text]
  (cond
    (= code 49) (change-text command " ")
    (= code 51) (apply str (butlast command))
    (valid-character? text) (change-text command text)
    :else command))

(defn update-command [code text]
  (swap! state update :command-text determine-command-update code text))

(defn clear-command []
  (swap! state assoc :command-text ""))

(defn set-active [bool]
  (swap! state assoc :active bool))



(defn show [code text event]
  (when (:active @state)
    (capture-keys event)
    (update-command code text)
    (-> window pack! repaint!))
  (when (= code 53)
    (capture-keys event)
    (set-active true)))

(defn hide [code text event] 
  (when (= code 53)
    (set-active false)
    (apply commands/execute-command 
                        (first (commands/get-commands-with-suggestions 
                                (:command-text @state))))
    (clear-command)))

(def myGlobalKeyListener
  (reify
    NativeKeyListener
    (nativeKeyTyped [this event] #(%))
    (nativeKeyPressed [this event] (handle-key-press event show))
    (nativeKeyReleased [this event]
                       (handle-key-press event hide))))



(defn -main []
  (start-key-logger myGlobalKeyListener))


(comment
  (-main))

