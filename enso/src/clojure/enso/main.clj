(ns enso.main
  (:require [enso.logic :as logic]
            [enso.ui :as ui])
  (:import org.jnativehook.keyboard.NativeKeyListener))

(def state (atom {:active false
                  :command-text ""}))

(add-watch state :debug
           (fn [key _ old-state new-state]
             (println new-state)))

; Keeping these top level like this makes live editing easier
; But if I want the ui to be pluggable, I will definitely need
; to do something else.

(def help-label (ui/create-help-label ui/standard-help-text))
(def input-container (ui/left-align (ui/make-input-container)))
(def command-containers (ui/create-horizontal-panels 10))
(def container (ui/make-container command-containers 
                                  [help-label input-container]))
(def window (ui/draw-window container))

(def input-binding (ui/bind-input state input-container))
(def auto-complete-binding (ui/bind-auto-complete state command-containers))
(def window-binding-visibility (ui/bind-visibility state window))

(def myGlobalKeyListener
  (reify
    NativeKeyListener
    (nativeKeyTyped [this event] #(%))
    (nativeKeyPressed [this event]
      (logic/handle-key-press state event logic/on-press))
    (nativeKeyReleased [this event] 
      (logic/handle-key-press state event logic/on-release))))







(defn -main []
  (logic/start-key-logger myGlobalKeyListener))

(comment
  (-main))
