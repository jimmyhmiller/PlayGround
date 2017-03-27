(ns cljs-forms.core
  (:require [reagent.core :as r]
            [cljs-forms.components :refer [render-form]]
            [clojure.walk :refer [postwalk postwalk-demo prewalk walk]]
            [clojure.spec :as s]
            [cljs.spec.impl.gen :as gen]
            [clojure.test.check.generators :as generators]))

(enable-console-print!)

(println "This text is printed from src/cljs-forms/core.cljs. Go ahead and edit it and see reloading in action.")

;; define your app data so that it doesn't get over-written on reload

(defonce app-state (atom {:text "Hello world!"}))


(def fields
  {:form/amount {:form/label "Enter an amount"}
   :form/another {:form/label "another"}
   :form/new-used {:form/label "New or Used?"
                   :form/values [{:form/value :form/new :form/label "New"}
                                 {:form/value :form/used :form/label "Used"}]}})

(def new-form
  [[:layout/horizontal 
    [:form/first-name] 
    [:form/middle-name] 
    [:form/last-name]]
   [:form/amount]])

(def form
  [:form/new-used
   :form/amount
   :form/another])

(defn another-component [{:keys [:form/label]} _] 
  [:div (str "Another: " label)])

(def comps 
  {:form/new-used :form/radio
   :form/another another-component})

(defn app []
  (render-form form fields comps))


(defn log [label x]
  (println label x)
  x)

(set! *recursion-limit* 2)

(s/exercise :form/component 4)


(walk (partial log "inner") (partial log "outer") new-form)

(app)

(defn ^:export run []
  (r/render [app]
            (js/document.getElementById "app")))

(run)

