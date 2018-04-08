(ns testing-stuff.components
  (:require  [clojure.spec :as s]))



(s/def :form/input (s/keys :req [:form/label]))
(s/def :form/label string?)

(s/def :form/radio (s/keys :req [:form/label :form/values]))
(s/def :form/values (s/coll-of :form/value-label :kind vector?))
(s/def :form/value-label (s/keys :req [:form/value :form/label]))
(s/def :form/value any?)


(defmulti render (fn [comp field fields] comp))

(defmethod render :form/input [_ {:keys [:form/label]} _]
  [:div 
   [:label label]
   [:input {:type :text}]])


(defn radio [{:keys [:form/value :form/label]}]
  [:label.radio-inline 
   [:input {:type :radio}]
   label])


(defmethod render :form/radio [_ {:keys [:form/label :form/values]} _]
  (let [radios (map radio values)]
    [:div radios]))


(defmethod render :default [f field fields]
  (if (clojure.test/function? f)
    (f field fields)
    (render :form/input field fields)))


(defn render-form [form fields comps]
  [:div
   (->> form
        (map (fn [field] 
               (render 
                (field comps)
                (field fields) 
                fields))))])
