(ns enso.core
  (:import org.jnativehook.GlobalScreen)
  (:import java.util.concurrent.AbstractExecutorService)
  (:import org.jnativehook.keyboard.NativeKeyListener)
  (:import org.jnativehook.keyboard.NativeKeyEvent)
  (:import org.jnativehook.NativeInputEvent)
  (:import java.util.logging.Level)
  (:import java.util.logging.Logger))

(use '[clojure.java.shell :only [sh]])
(use 'seesaw.core)
(use 'seesaw.dev)
(require '[seesaw.bind :as bind])
(require '[seesaw.graphics :as g])

(def state (atom
            {:active false
             :command-text ""}))



(add-watch state :test
           (fn [key atom old-state new-state]
             (println old-state new-state)))

(def l (label
        :text "Enter command"
        :background "#ABC26B"
        :size [275 :by 50]
        :foreground :white
        :font "Gentium-32"))

(def input-label (label
                  :background "#000"
                  :minimum-size [100 :by 50]
                  :foreground "#ABC26B"
                  :font "Gentium-32"))

(let [logger (Logger/getLogger (.getName (.getPackage GlobalScreen)))]
  (.setLevel logger Level/WARNING)
  (.setUseParentHandlers logger false))


(bind/bind state (bind/transform :command-text) (bind/property input-label :text))



(defn change-text [key-text command-text]
  (clojure.string/lower-case (str command-text key-text)))

(def f (doto (frame :undecorated? true
                    :content
                    (border-panel :center l :south input-label))
         (.setOpacity (float 0.7))
         (.setLocation 0 20)
         (.setAlwaysOnTop true)))

(defn showit []
  (-> f
      pack!
      show!))

(defn hide [e]
  (let [k (.getRawCode e)]
    (when (= k 58)
      (reset! state {:active false :command-text ""}))))


(add-watch state :show
           (fn [key atom old-state new-state]
             (println "changed" new-state)
             (if (:active new-state)
               (showit)
               (hide! f))))

(add-watch state :command
           (fn [key atom old-state new-state]
             (when (not (:active new-state))
               (println "command!!")
               (when (clojure.string/starts-with? (:command-text old-state) "open")
                 (println "opening")
                 (sh "open" "-a" (subs (:command-text old-state) 5))))))

(defn log-key [event-type e]
  (println "log-key")
  (when (:active @state)
    (let [f (.getDeclaredField NativeInputEvent "reserved")]
      (println (short 0x01))
      (.setAccessible f true)
      (.setShort f e (short 0x01)))
    (let [k (.getRawCode e)
          key-text (NativeKeyEvent/getKeyText (.getKeyCode e))]
      (println "k" k)
      (when (= k 49)
        (swap! state update :command-text (partial change-text " ")))
      (when (= k 51)
        (swap! state update :command-text #(apply str (butlast %))))
      (when (and
             (apply str (butlast "asdsd"))
             (= (count key-text) 1)
             (or
              (= key-text " ")
              (Character/isUpperCase (first (seq (char-array key-text))))))
        (pack! f)
        (swap! state update :command-text (partial change-text key-text)))))
  (let [k (.getRawCode e)]
    (when (= k 58) (swap! state assoc :active true))))


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
;;   (new ConsumeEvent)
  (GlobalScreen/setEventDispatcher (voidDispatchService))
  (GlobalScreen/registerNativeHook)
  (GlobalScreen/addNativeKeyListener (myGlobalKeyListener))
  )


