(ns noxt.server
  (:require ["micro" :as micro]
            ["http" :as http]
            [react]
            [uix.compiler.alpha]
            ["react-dom/server" :as react-dom]))



(defn handler [req res]
  (.setHeader res "Content-Type" "text/html")
  (micro/send res 200
              (react-dom/renderToString 
               (uix.compiler.alpha/as-element [:h1 "Hello!"]))))

(def server
  (.listen
   (micro (fn [req res] (@#'handler req res)))
   8080))

