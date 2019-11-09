(ns noxt.server
  (:require ["micro" :as micro]
            ["http" :as http]
            [index]
            [about]
            ["react-dom/server" :as react-dom]
            [uix.compiler.alpha]
            [uix.dom.alpha]
            [serve-handler]))


(defn handler [req res]
  (if (or (= (.-url req) "/")
          (= (.-url req) "/about"))
    (do
      (let [main (if (= (.-url req) "/")  index/main about/main)]
        (.setHeader res "Content-Type" "text/html")
        (micro/send res 200
                    (react-dom/renderToString
                     (uix.compiler.alpha/as-element 
                      [:html
                       [:head
                        [:meta {:charset "UTF-8"}]]
                       [:body
                        [:div {:id "app"}
                         [main]]
                        [:script {:src "/js/cljs_base.js" :type "text/javascript"}]
                        [:script {:src "/js/main.js" :type "text/javascript"}]]])))))
    (serve-handler req res #js {:public "public"})))

(def server (atom nil))

(defn -main []
  (reset! server
    (.listen
       (micro (fn [req res] (@#'handler req res)))
       8080)))

(comment
  (when (exists? js/Symbol)
    (extend-protocol IPrintWithWriter
      js/Symbol
      (-pr-writer [sym writer _]
        (-write writer (str "\"" (.toString sym) "\""))))))
