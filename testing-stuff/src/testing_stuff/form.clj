(ns testing-stuff.form
  (:require [clojure.spec :as s]
             [testing-stuff.components :refer [field->comp render-form]]))


(def fields
  {:form/amount {:form/label "Enter an amount"}
   :form/another {:form/label "t"}
   :form/new-used {:form/label "New or Used?"
               :form/values [{:form/value :form/new :form/label "New"}
                         {:form/value :form/used :form/label "Used"}]}})

(def form
  [:form/new-used
   :form/amount])


(field->comp :form/new-used :form/radio)
(field->comp :form/another (fn [{:keys [label]} _] [:div (str "Another: " label)]))




(render-form form fields)
         
