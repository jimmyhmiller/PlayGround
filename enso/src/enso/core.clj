(ns enso.core
  (:gen-class)
  (:import org.jnativehook.GlobalScreen)
  (:import org.jnativehook.keyboard.NativeKeyListener))


(use 'seesaw.core)
(use 'seesaw.dev)
(require '[seesaw.graphics :as g])


(def l (label
        :text "Enter command"
        :background "#ABC26B"
        :size [275 :by 50]
        :foreground :white
        :font "Gentium-32"))

(def f (doto (frame :undecorated? true
                    :content
                    (border-panel :center l))
         (.setOpacity (float 0.7))
         (.setLocation 0 20)
         (.setAlwaysOnTop true)))

(defn showit []
    (-> f
      pack!
      show!))

(defn hide [e]
  (let [k (.getRawCode e)]
    (when (= k 58) (hide! f))))

(defn log-key [e]
    (let [k (.getRawCode e)]
      (when (= k 58) (showit))))

(defn myGlobalKeyListener []
  (reify
    NativeKeyListener
    (nativeKeyPressed [this event] (log-key event))
    (nativeKeyReleased [this event] (hide event))))


(defn -main [& args]
  (GlobalScreen/registerNativeHook)
  (.addNativeKeyListener (GlobalScreen/getInstance) (myGlobalKeyListener)))

(-main)


