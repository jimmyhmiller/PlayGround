(ns noxt.server
  (:require ["micro" :as micro]
            ["http" :as http]
            [react]
            ["react-dom/server" :as react-dom]))

(defn handler [req res]
  (Promise/resolve "Hello!"))

(def server
  (.listen
   (micro (fn [req res] (@#'handler req res)))
   8080))


;; Fails because of loader.
;; https://clojure.atlassian.net/browse/CLJS-3067
;; Might fix it, but I have to figure out how to use latest with
;; cider or wait for release, which I think is soon anyways.
(require '[about])
