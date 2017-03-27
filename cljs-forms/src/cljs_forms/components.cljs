(ns cljs-forms.components
  (:require [clojure.spec :as s]))



(s/def :form/component (s/cat :tag keyword? :attr (s/? map?) :children (s/* :form/component)))

(s/def :form/input (s/keys :req [:form/label]))
(s/def :form/label string?)

(s/def :form/radio (s/keys :req [:form/label :form/values]))
(s/def :form/values (s/coll-of :form/value-label :kind vector?))
(s/def :form/value-label (s/keys :req [:form/value :form/label]))
(s/def :form/value any?)


(defmulti render (fn [comp field fields field-name] comp))

(defmethod render :form/input [_ {:keys [:form/label]} _ _]
  [:div.form-group
   [:label.control-label label]
   [:input.form-control {:type :text}]])


(defn radio [field-name {:keys [:form/value :form/label]}]
  [:label.radio-inline 
   [:input {:type :radio :name field-name}]
   label])


(defmethod render :form/radio [_ {:keys [:form/label :form/values]} _  field-name]
  (let [radios (map (partial radio field-name) values)]
    [:div 
     [:label.control-label label]
     radios]))


(defmethod render :default [f field fields field-name]
  (if (fn? f)
    (f field fields field-name)
    (render :form/input field fields field-name)))


(defn render-form [form fields comps]
  [:div
   (->> form
        (map (fn [field-name] 
               (render
                (field-name comps)
                (field-name fields)
                fields
                field-name))))])
