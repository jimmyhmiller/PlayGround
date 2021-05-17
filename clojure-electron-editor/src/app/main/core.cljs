(ns app.main.core
  (:require ["electron" :refer [app BrowserWindow crashReporter] :as electron]
            [nrepl-client :as nrepl-client]
            ["nrepl-server" :default nrepl-server]))
(comment

  (.removeAllListeners electron/ipcMain "eval")
  (def server-state (atom nil))

  (.start nrepl-server  {} (fn [err server]
                             (when err
                               (println err))
                             (reset! server-state server)))
  (.stop nrepl-server @server-state)

  (def conn
    (.connect
     nrepl-client
     #js {:port "56011" :verbose true}))


  (.once conn "connect"
         (fn []
           (println "Connect")
           (.eval conn "(+ 2 2)" (fn [err result]
                                   (.reduce result (fn [result msg]
                                                     (println result)
                                                     (if (.-value msg)
                                                       (str result (.-value msg))
                                                       result))
                                            "")
                                   (println result)))))
  (.on conn "error" (fn [err]
                      (println err)))


  (.on electron/ipcMain "eval"
       (fn [event arg]
         (println "Got3" event arg (.-sender event))
         (.eval conn arg (fn [err result]
                           (println "got here" err result)
                           (.send (.-sender event) "eval-result"
                                  (.reduce result (fn [result msg]
                                                    (println result)
                                                    (if (.-value msg)
                                                      (str result (.-value msg))
                                                      result))
                                           ""))))))

  (.end conn))





(def main-window (atom nil))

(defn init-browser []
  (reset! main-window (BrowserWindow.
                        (clj->js {:width 800
                                  :height 600
                                  :webPreferences
                                  {:nodeIntegration true
                                  	:contextIsolation false,}})))
  ; Path is relative to the compiled js file (main.js in our case)
  (.loadURL ^js/electron.BrowserWindow @main-window (str "file://" js/__dirname "/public/index.html"))
  (.on ^js/electron.BrowserWindow @main-window "closed" #(reset! main-window nil)))

(defn main []
  ; CrashReporter can just be omitted
  (.start crashReporter
          (clj->js
           {:companyName "MyAwesomeCompany"
            :productName "MyAwesomeApp"
            :submitURL "https://example.com/submit-url"
            :autoSubmit false}))

  (.on app "window-all-closed" #(when-not (= js/process.platform "darwin")
                                  (.quit app)))
  (.on app "ready" init-browser))
