(ns testing-stuff.components
  (:require  [clojure.spec :as s]))

(defmulti choose-component (fn [field fields] field))
(defmethod choose-component :default [_ _] :form/input)


(defmacro field->comp [field comp]
  `(defmethod choose-component ~field [_# _#] ~comp))




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
    (throw (Exception. (str "No render for component " f)))))

(render (fn [x y] x) :test :test)

(defn get-and-render-comp [fields field]
  (choose-component field fields))


(defn render-form [form fields]
  [:div
   (->> form
        (map (fn [field] 
               (render 
                (get-and-render-comp fields field) 
                (field fields) 
                fields))))])
