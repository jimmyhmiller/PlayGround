(ns ^:figwheel-hooks cljs-proxy.core
  (:require
   [goog.dom :as gdom]
   [goog.object :as gobj]
   [react :as react]
   [react-dom :as react-dom]
   [sablono.core :as sab :include-macros true]))

(println "This text is printed from src/cljs_proxy/core.cljs. Go ahead and edit it and see reloading in action.")

(defn multiply [a b] (* a b))


;; define your app data so that it doesn't get over-written on reload
(defonce app-state (atom {:text "Hello world!"}))

(defn get-app-element []
  (gdom/getElement "app"))



(defn unwrap
  [^js p]
  (.-__unwrap p))


(defn shallow-clj->js
  [m]
  (loop [entries (seq m)
         o #js {}]
    (if (nil? entries)
      o
      (let [entry (first entries)
            k (key entry)
            v (val entry)]
        (recur
         (next entries)
         (doto o
           (gobj/set (name k) v)))))))


(declare ueaq)

;; -- traps


(defn getter
  [o prop]
  (this-as handler
    (let [opts (.-opts ^js handler)
          context (.-context ^js handler)
          {:keys [recursive? prop->key]} opts

          v (get o (prop->key prop)
                 (gobj/get context prop))]
      (if (and recursive? (associative? v))
        (ueaq v opts)
        v))))


(defn has
  [o prop]
  (this-as handler
    (let [{:keys [prop->key]} (.-opts ^js handler)]
      (contains? o (prop->key prop)))))


(defn own-keys
  [o]
  (js/Reflect.ownKeys o))


(defn enumerate
  [o]
  (map name (keys o)))


(defn get-own-property-descriptor
  [o prop]
  (this-as handler
    (let [{:keys [prop->key]} (.-opts ^js handler)
          k (prop->key prop)]
      (if (contains? o k)
        #js {:enumerable true :configurable true
             :writable false :value (get o k)}
        (js/Object.getOwnPropertyDescriptor o prop)))))


(defn get-prototype-of
  [_]
  (this-as handler
    (.-context ^js handler)))


(defn setter
  [_ k v]
  (this-as handler
    (let [context (.-context ^js handler)]
      (gobj/set context k v))))


(defn ^js ueaq
  ([o] (ueaq o {}))
  ([o {:keys [recursive? prop->key mutable?] :as opts
       :or {recursive? false
            prop->key keyword}}]
   (let [;; this is an object to hold implementations of various protocols for
         ;; CLJS usage
         context (specify! #js {}
                   Object
                   (toString [this]
                     (pr-str* this))

                   IPrintWithWriter
                   (-pr-writer [_ writer _]
                     ;; prn is not a fast path
                     (-write writer
                             (if recursive?
                               (pr-str (clj->js o))
                               (pr-str (shallow-clj->js o))))))

         handler #js {:opts (assoc opts
                                   :prop->key prop->key)
                      :context context
                      :get getter
                      :has has
                      :ownKeys own-keys
                      :enumerate enumerate
                      :getOwnPropertyDescriptor get-own-property-descriptor
                      :getPrototypeOf get-prototype-of
                      :set setter}]
    (js/Proxy. (shallow-clj->js o) handler))))


;; Because of the invariants, target needs to actually have the properties I am talking about it seems.
(defn make-obj [coll]
  (js/Proxy. (shallow-clj->js coll)
             #js {:enumerate (fn [target]
                               (throw (ex-info "enumerate" {})))
                  :getOwnPropertyDescriptor 
                  (fn [target key]
                      (js/Object.getOwnPropertyDescriptor target key))
                  :has (fn [target key]
                         (contains? coll (keyword key)))
                  :ownKeys (fn [target]
                             (clj->js (mapv name (keys coll))))
                  :set (fn [obj prop value]
                         (println "set" obj prop value))
  
                  :get (fn [obj prop]
                         (println "get" obj prop)

                         (if (= prop "hasOwnProperty")
                           (fn [prop]
                             (contains? coll (keyword prop)))
                           (let [result (get coll (keyword prop))]
                             (if (map? result)
                               (make-obj result)
                               result))))}))




(defn hello-world [state]
  (react/createElement "div" (ueaq {:style {:color "green"
                                                :fontSize 120
                                                :cursor "pointer"}
                                        :id "thing"
                                        :onClick (fn [e] (println e))}
                                        {:recursive? true})
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
