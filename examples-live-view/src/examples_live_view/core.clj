(ns example-live-view.core
  (:require [reitit.ring]
            [live-view-server.core :as live-view]
            [cheshire.core :as json]
            [ring.adapter.jetty :as jetty]
            [ring.middleware.json]
            [ring.middleware.keyword-params]
            [clojure.java.io :as io]
            [cheshire.generate]
            [clojure.pprint :as pprint]
            [clojure.string :as string]
            [glow.html]
            [glow.parse]
            [glow.core]))


(def my-color-scheme
  {:exception "#005cc5"
   :repeat "#005cc5"
   :conditional "#005cc5"
   :variable "#df5000"
   :core-fn "#6f42c1"
   :definition "#d73a49"
   :reader-char "#032f62"
   :special-form  "#6f42c1"
   :macro "#6f42c1"
   :number "#d73a49"
   :boolean "#005cc5"
   :nil "#005cc5"
   :s-exp :black
   :keyword "#005cc5"
   :comment "#969896"
   :string "#032f62"
   :character "#032f62"
   :regex "#032f62"})

(defn pprint-str [coll]
  (let [out (java.io.StringWriter.)]
    (pprint/pprint coll out)
   (str out)))

;; To add:
;; Styles
;; Search
;; Replay request button
;; Save req/response as var in repl

(defn view [{:keys [requests shown]}]
  [:body
   [:style {:type "text/css"}
    (glow.core/generate-css my-color-scheme)]
   [:h1 "Ring Api Inspector"]
   [:div {:style {:width 800 :margin 30}}
    [:div {:style {:display "grid" :grid-template-columns "1fr 1fr 1fr 1fr"}}
     [:div "Status"]
     [:div "Method"]
     [:div "Uri"]
     [:div "Time"]]
    (map-indexed
     (fn [i {:keys [request response time]}]
       (let [{:keys [request-method uri]} request
             {:keys [status]} response]
         [:div
          [:div {:style {:display "grid"
                         :grid-template-columns "1fr 1fr 1fr 1fr"
                         :padding 10
                         :background-color (if (even? i) "rgb(249,249,249)" "white")
                         :cursor "pointer"}
                 :onclick [:toggle {:index i}]}
           [:div status]
           [:div (name request-method)]
           [:div uri]
           [:div time]]
          (when (contains? shown i)
            [:div
             [:h2 "Request"]
             [:div [:code.syntax [:pre (glow.html/hiccup-transform (glow.parse/parse (pprint-str request)))]]]
             [:h2 "Response"]
             [:div  [:code.syntax [:pre (glow.html/hiccup-transform (glow.parse/parse (pprint-str response)))]]]])]))
     requests)]])

(def state (atom {:requests ()
                  :shown #{}}))

(defn event-handler [{:keys [action]}]
  (println action)
  (let [[action-type payload] action]
    (case action-type
      :toggle (let [index (:index payload)]
                (swap! state update :shown (fn [shown]
                                             (if (contains? shown index)
                                               (disj shown index)
                                               (conj shown index)))))
      (println "not handled" action))))


(do
  (def add-tap-to-state)
  (remove-tap add-tap-to-state)

  (defn add-tap-to-state [value]
    (swap! state update :requests (fn [coll] (cons value coll))))

  (add-tap add-tap-to-state))

(comment
  (def live-view-server
    (live-view/start-live-view-server
     {:state state
      :view #'view
      :event-handler #'event-handler
      :port 12345})))





;; BEGIN REST API



(def posts
  (json/parse-string (slurp (io/resource "posts.json")) true))

(def list-of-posts (vals (:list posts)))

(def handler
  (reitit.ring/ring-handler
   (reitit.ring/router
    ["/api"
     ["/post" {:get {:handler (fn [{:keys [headers scheme] :as req}]
                                {:status 200
                                 :body (map (fn [[id _]]
                                              {:id (name id)
                                               :uri (str (name scheme) "://" (get headers "host") "/api/post/" (name id))})
                                            (:list posts))})}}]
     ["/post/:id" {:get {:handler (fn [{{:keys [id]} :path-params}]
                                    {:status 200
                                     :body {:post (get-in posts [:list (keyword id)])}})}}]
     ["/random-post" {:get {:handler (fn [_]
                                       {:status 200
                                        :body {:post (rand-nth list-of-posts)}})}}]])))



(defn tapping-middleware [handler]
  (fn [req]
    (let [response (handler req)]
      (tap> {:request req
             :response response
             :time (str (java.time.Instant/now))})
      response)))

(def app (-> #'handler
             tapping-middleware
             ring.middleware.json/wrap-json-response
             ring.middleware.json/wrap-json-body
             ring.middleware.json/wrap-json-params
             ring.middleware.keyword-params/wrap-keyword-params))



(def web-server
  (jetty/run-jetty #'app {:port 8000
                          :join? false}))




(comment

  (dotimes [_ 100]
    (app {:request-method :get
          :uri "/api/random-post"}))
  
  (.stop web-server))


