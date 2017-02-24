(ns testing-stuff.form
  (:require [clojure.spec :as s]
             [testing-stuff.components :refer [render-form]]))


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
  {:form/new-used [:form/radio {:size 2}]
   :form/another another-component})



(render-form form fields comps)
