(ns app.main.core
  (:require ["electron" :refer [app BrowserWindow crashReporter] :as electron]
            [nrepl-client :as nrepl-client]
            ["nrepl-server" :default nrepl-server]
            [clojure.pprint]))




;; Need to actually make it so we can load up projects and open files
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
     #js {:port (.-port @server-state) :verbose true}))


  (.on conn "error" (fn [err]
                      (println err)))

  ;; not right, run manually
  (.once conn "connect"
         (fn []
           ;; Should pull out of options
           (.eval conn "(defn my-print [value writer options]
(binding [clojure.pprint/*print-right-margin* 50]
  (clojure.pprint/pprint value writer)))")))

  (do (.removeAllListeners electron/ipcMain "eval")
      (.on electron/ipcMain "eval"
           (fn [event arg]
             (.send conn
                    #js {:op "eval"
                         :code arg
                         #_#_"nrepl.middleware.print/print" "user/my-print"
                         #_#_"nrepl.middleware.print/options" #js {:print-width 40}}
                    (fn [err result]
                      
                      (.send (.-sender event) "eval-result"
                             (.reduce result (fn [result msg]
                                               (cond (.-value msg)
                                                     (str result (.-value msg))
                                                     (.-err msg)
                                                     (str result (.-err msg))
                                                     :else result))
                                      "")))))))
  
  (.end conn))

(comment

  (.send conn #js {:op "eval"
                   "nrepl.middleware.print/print" "user/my-print"
                   :code "{:a 2 :b 3 :c 3 :d 5 :e 2 :f 4 :g [{:a 2 :b 3 :c 3 :d 5 :e 2 :f 4 :g [{:a 2}]}]}"}
         (fn [err result]
           (println "Asdsad"
                    result)))

  (+ 2 2)



  (clojure.pprint/pprint {:a 2 :b 3 :c 3 :d 5 :e 2 :f 4 :g [{:a 2 :b 3 :c 3 :d 5 :e 2 :f 4 :g [{:a 2}]}]}))
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
