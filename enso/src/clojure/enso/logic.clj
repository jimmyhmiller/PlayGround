(ns enso.logic
  (:require [clojure.string :as string]
            [seesaw.core :refer [invoke-now]])
  (:import java.util.concurrent.AbstractExecutorService
           [org.jnativehook GlobalScreen NativeInputEvent]
           org.jnativehook.keyboard.NativeKeyEvent))

(defn capture-keys [event]
  (doto (.getDeclaredField NativeInputEvent "reserved")
  (.setAccessible true)
  (.setShort event (short 0x01))))

(defn handle-key-press [state event callback]
  (callback state 
            (.getRawCode event)
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

(defn update-command [state code text]
  (invoke-now (swap! state update :command-text determine-command-update code text)))

(defn clear-command [state]
  (invoke-now (swap! state assoc :command-text "")))

(defn set-active [state bool]
  (invoke-now (swap! state assoc :active bool)))

(defn on-press [state code text event]
  (when (:active @state)
    (capture-keys event)
    (update-command state code text))
  (when (= code 53)
    (capture-keys event)
    (set-active state true)))

(defn on-release [state code text event]
  (when (= code 53)
    (set-active state false)
    (clear-command state)))
