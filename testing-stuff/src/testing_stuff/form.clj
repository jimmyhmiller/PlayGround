(ns testing-stuff.form
  (:require [clojure.spec :as s]
             [testing-stuff.components :refer [field->comp render-form]]))


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


(field->comp :form/new-used :form/button-group)


(field->comp :form/another (fn [{:keys [:form/label]} _] [:div (str "Another: " label)]))




(clojure.pprint/pprint (render-form form fields))
         
