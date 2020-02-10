(ns noxt.server
  (:require ["micro" :as micro]
            ["http" :as http]
            ["react-dom/server" :as react-dom]
            [uix.compiler.alpha]
            [uix.dom.alpha]
            [serve-handler]))

;; Need to import all pages here and generate them at compile time.
;; There is no dynamic requiring in clojurescript


;; Actually, could I js/require?
;; This is a terrible idea :)
;; But it was fun to do to understand how this all works.
;; But if these pages had deps, none of this would work at all.
;; So we need to load all the pages namespaces in at compile time.
;; That means we have two separate compiles for backend and frontend.
;; Hopefully that isn't super slow and we can leverage a repl.

(comment
  (set! js/example #js {"pages" #js {}})
  (set! js/cljs.loader #js {"set_loaded_BANG_" (fn [name])})

  (js/require "../example.pages.index_loader") 
  (js/example.pages.index.main)
)


(defn handler [req res]
  (if (or (= (.-url req) "/")
          (= (.-url req) "/about"))
    (do
      (let [main (if (= (.-url req) "/") (fn [] "placeholder code routes used to be here") (fn [] "two"))]
        (.setHeader res "Content-Type" "text/html")
        (micro/send res 200
                    (react-dom/renderToString
                     (uix.compiler.alpha/as-element 
                      [:html
                       [:head
                        [:meta {:charset "UTF-8"}]]
                       [:body
                        [:div {:id "app"}
                         [:h1 "TEST"]
                         [main]]
                        [:script {:src "/js/cljs_base.js" :type "text/javascript"}]
                        [:script {:src "/js/main.js" :type "text/javascript"}]]])))))
    (serve-handler req res #js {:public "public"})))

(def server (atom nil))


(println "running server")
(reset! server
  (.listen
     (micro (fn [req res] (@#'handler req res)))
     8080))


;; Printing react elements get's ugly because 
;; there are symbols that don't have a print implementation
(comment
  (when (exists? js/Symbol)
    (extend-protocol IPrintWithWriter
      js/Symbol
      (-pr-writer [sym writer _]
        (-write writer (str "\"" (.toString sym) "\""))))))
