(ns cljs-forms.core
  (:require [reagent.core :as r]
            [cljs-forms.components :refer [render-form]]))

(enable-console-print!)

(println "This text is printed from src/cljs-forms/core.cljs. Go ahead and edit it and see reloading in action.")

;; define your app data so that it doesn't get over-written on reload

(defonce app-state (atom {:text "Hello world!"}))

(println "hello")


(def fields
  {:form/amount {:form/label "Enter an amount"}
   :form/another {:form/label "another"}
   :form/new-used {:form/label "New or Used?"
                   :form/values [{:form/value :form/new :form/label "New"}
                                 {:form/value :form/used :form/label "Used"}]}})

(def form
  [:form/new-used
   :form/amount
   :form/another])

(defn another-component [{:keys [:form/label]} _] 
  [:div (str "Another: " label)])

(def comps 
  {:form/new-used :form/radio
   :form/another another-component})

(println (render-form form fields comps))

(defn app []
  (render-form form fields comps))

(defn ^:export run []
  (r/render [app]
            (js/document.getElementById "app")))

(run)

