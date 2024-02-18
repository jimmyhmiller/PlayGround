(ns jimmyhmiller.chessnut
  (:import [org.sputnikdev.bluetooth.manager.impl BluetoothManagerBuilder]))


(let [builder (BluetoothManagerBuilder.)
      _ (.withBlueGigaTransport builder "^*.$")
      manager (.build builder)]
  ;; Use `manager` here
  )



(defn greet
  "Callable entry point to the application."
  [data]
  (println (str "Hello, " (or (:name data) "World") "!")))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (greet {:name (first args)}))
