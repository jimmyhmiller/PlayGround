(ns shadow-debug.core)

(defn render []
  (let [worker (js/Worker. "/js/worker.js")]
    (println "Render")))
