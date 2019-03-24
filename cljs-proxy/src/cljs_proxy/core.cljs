(ns ^:figwheel-hooks cljs-proxy.core
  (:require
   [goog.dom :as gdom]
   [react :as react]
   [react-dom :as react-dom]
   [sablono.core :as sab :include-macros true]))

(println "This text is printed from src/cljs_proxy/core.cljs. Go ahead and edit it and see reloading in action.")

(defn multiply [a b] (* a b))


;; define your app data so that it doesn't get over-written on reload
(defonce app-state (atom {:text "Hello world!"}))

(defn get-app-element []
  (gdom/getElement "app"))


(defn make-obj [coll]
  (js/Proxy. #js {} 
             #js {:enumerate (fn [target]
                               (throw (ex-info "enumerate" {})))
                  :getOwnPropertyDescriptor 
                  (fn [target key]
                    (println key)
                    (if (contains? coll (keyword key))
                      #js {:value (get coll key)
                           :enumerable true
                           :configurable true}
                      (clj->js
                       (merge
                        (js->clj
                         (js/Object.getOwnPropertyDescriptor target))
                        {:enumerable false
                         :configurable true}))))
                  :has (fn [target key]
                         (contains? coll (keyword key)))
                  :preventExtensions (fn [target]
                                       true)
                  :ownKeys (fn [target]
                             (clj->js (mapv name (keys coll))))
                  :set (fn [obj prop value]
                         (println "set" obj prop value))
                  :isExtensible (fn [target]
                                  false)
                  :defineProperty (fn [target key desc]
                                    (throw (ex-info "define" {:data [target key desc]})))
                  :get (fn [obj prop]
                         (println "get" obj prop)

                         (if (= prop "hasOwnProperty")
                           (fn [prop]
                             (contains? coll (keyword prop)))
                           (let [result (get coll (keyword prop))]
                             (if (map? result)
                               (make-obj result)
                               result))))}))


;; Invariants around proxy make this terrible. Is there an alternative?
;; I think possibly.
(set! (.-freeze (.-Object js/window)) (fn [x] x))



(def p
  )





(defn hello-world [state]
  (js/console.log p)
  (react/createElement "div" (make-obj {:style {:color "green"
                                                :fontSize 120
                                                :cursor "pointer"}
                                        :id "thing"
                                        :onClick (fn [e] (println e))})
                       "Hello World"))

(defn mount [el]
  (js/ReactDOM.render (hello-world app-state) el))

(defn mount-app-element []
  (when-let [el (get-app-element)]
    (mount el)))

;; conditionally start your application based on the presence of an "app" element
;; this is particularly helpful for testing this ns without launching the app
(mount-app-element)

;; specify reload hook with ^;after-load metadata
(defn ^:after-load on-reload []
  (mount-app-element)
  ;; optionally touch your app-state to force rerendering depending on
  ;; your application
  ;; (swap! app-state update-in [:__figwheel_counter] inc)
)
