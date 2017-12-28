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
              repaint!
              replace!
              invoke-now
              invoke-soon
              select]]
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
   :h-text-position :left
   :halign :left
   :v-text-position :top
   :valign :top
   :font "Gentium Plus-32"
   :border (empty-border :thickness 5)))

(def standard-help-text 
  "Welcome to Enso! Enter a command, or type \"help\" for assistance.")



(defn create-input-label []
  (label
   :halign :left
   :h-text-position :left
   :background "#000"
   :foreground "#fff"
   :font "Gentium Plus-64"
   :border (empty-border :thickness 5)))

(defn generic-label
  ([text color font-size font-style]
   (label
    :halign :left
    :h-text-position :left
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

(defn draw-window [content]
  (doto (frame :undecorated? true :size [1000 :by 1000] :content content)
    (.setOpacity (float 0.0))
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




(def help-label (create-help-label standard-help-text))
(def input-container (left-align (horizontal-panel)))
(def command-containers 
  (doall (for [i (range 10)]
           (left-align (horizontal-panel)))))

(def container 
  (->> command-containers
       (concat [help-label input-container])
       (into [])
       (vertical-panel :items)
       (left-align)))

(def window (draw-window container))
(show! window)


(.putClientProperty (.getRootPane window) "Window.shadow" Boolean/FALSE)


(add-watch state :show
           (fn [key atom old-state new-state]
             (cond 
               (and (:active new-state) 
                    (not (:active old-state))) (-> window (.setOpacity (float 0.85)))
               (and (not (:active new-state))
                    (:active old-state)) (-> window (.setOpacity (float 0.0))))))

(add-watch state :commands
           (fn [key atom old-state new-state]
             (let [text (:command-text new-state)
                   command (first (commands/get-commands text))
                   suggestion (first (commands/get-suggestions command text))]
               (println text)
               (println (parse/parse text command suggestion)))))



(def type->style 
  {:selected {:highlight :white
              :base enso.refactor/green
              :arg :gray
              :size 64}
   :not-selected {:highlight "#c8d79e"
                  :base enso.refactor/green
                  :arg :gray
                  :size 32}})

(defn generic-label-for-type [text type prop]
  (let [styles (type->style type)]
    (generic-label text (prop styles) (:size styles))))



(defmulti render-parsed-command (fn [parsed type] (first parsed)))

(defmethod render-parsed-command :line [[_ & children] type]
  (map #(render-parsed-command % type) children))

(defmethod render-parsed-command :match [[_ value] type]
  (generic-label-for-type value type :highlight))

(defmethod render-parsed-command :unmatch [[_ value] type]
  (generic-label-for-type value type :base))

(defmethod render-parsed-command :arg [[_ value] type]
  (generic-label-for-type value type :arg))

(defmethod render-parsed-command :default [_ _] nil)


(defn left-align [c]
  (doto c
    (.setAlignmentX Component/LEFT_ALIGNMENT)))

(defn top-match-label [text]
  (println "top" text)
  (let [commands (commands/get-commands-with-suggestions text)
        [command suggestion] (first commands)]
    (render-parsed-command (parse/parse text command suggestion) :selected)))

(defn other-match-labels [text]
  (let [commands (rest (commands/get-commands-with-suggestions text))]
    (->> commands
         (map (fn [[command suggestion]]
            (parse/parse text command suggestion)))
         (map #(render-parsed-command % :not-selected))
         (into []))))




(def bound 
  (bind/bind state
             (bind/transform #(top-match-label (:command-text %)))
             (bind/property input-container :items)
             (bind/notify-now)))


(def bindings 
  (doall (for [i (range 10)]
           (bind/bind state
                      (bind/transform #(get (other-match-labels (:command-text %)) i []))
                      (bind/property (nth command-containers i) :items)))))



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
  (invoke-soon (swap! state update :command-text determine-command-update code text)))

(defn clear-command []
  (invoke-now (swap! state assoc :command-text "")))

(defn set-active [bool]
  (invoke-soon (swap! state assoc :active bool)))

(defn show [code text event]
  (when (:active @state)
    (capture-keys event)
    (update-command code text))
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
    (nativeKeyPressed [this event] (handle-key-press event show))))



(defn -main []
  (start-key-logger myGlobalKeyListener))


(comment
  (-main))

