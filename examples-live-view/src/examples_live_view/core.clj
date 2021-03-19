(ns example-live-view.core
  (:require [reitit.ring]
            [live-view-server.core :as live-view]
            [cheshire.core :as json]
            [ring.adapter.jetty :as jetty]
            [ring.middleware.json]
            [ring.middleware.keyword-params]
            [clojure.java.io :as io]
            [cheshire.generate]
            [clojure.pprint :as pprint]))





(defn view [data]
  [:body
   (for [datum data]
     [:code [:pre (let [out (java.io.StringWriter.)]
                    (pprint/pprint datum out)
                    (str out))]])])


(defn event-handler [_])

(def state (atom {}))

(reset! state ())

(remove-tap add-tap-to-state)

(defn add-tap-to-state [value]
  (swap! state (fn [coll] (take 100 (cons value coll)))))

(add-tap add-tap-to-state)


(def live-view-server
  (live-view/start-live-view-server
   {:state state
    :view #'view
    :event-handler #'event-handler
    :port 12345}))




(cheshire.generate/add-encoder Object (fn [o generator]
                                        (.writeString generator (pr-str o))))

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
             :response response})
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



(app {:request-method :get
      :uri "/api/random-post"})


(comment
  (.stop web-server))
