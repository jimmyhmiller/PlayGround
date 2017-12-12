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
            [seesaw.core :refer [label frame]]
            [seesaw.border :refer [empty-border]]
            [seesaw.font :refer [font]]
            [seesaw.color :refer [color]]))

(defn create-help-label [text]
  (label
   :text text
   :background "#ABC26B"
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
            (NativeKeyEvent/getKeyText (.getKeyCode event))))

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

(defn myGlobalKeyListener [show hide]
  (reify
    NativeKeyListener
    (nativeKeyTyped [this event] #(%))
    (nativeKeyPressed [this event] (handle-key-press event show))
    (nativeKeyReleased [this event]
                       (handle-key-press event hide))))

(defn start-key-logger [show hide]
  (GlobalScreen/setEventDispatcher (voidDispatchService))
  (GlobalScreen/registerNativeHook)
  (GlobalScreen/addNativeKeyListener (myGlobalKeyListener show hide)))

