(ns phone-number.handlers
  (:require [ring.util.response :refer [response status]]
            [phone-number.data :refer [->e164 valid-e164?] :as data]
            [clojure.spec.alpha :as s]))

(defn error [status-code error-message]
  (status (response {:error error-message}) status-code))

(def not-nil? (complement nil?))

(defn get-phone-number [phone-numbers phone-number]  
  (let [e164 (->e164 phone-number)
        entry (get phone-numbers e164)]
    (cond
      (not (valid-e164? e164)) [:invalid-format phone-number]
      (not-nil? entry) [:valid entry]
      :else [:not-found phone-number])))

(defmulti query-number-response first)

(defmethod query-number-response :valid [[_ entry]]
  (response entry))

(defmethod query-number-response :not-found [[_ number]]
  (error 404 (str number " not found")))

(defmethod query-number-response :invalid-format [[_ number]]
  (error 400 (str number " is not a valid phone number")))

(defn query-number [phone-numbers number]
  (->> number
       (get-phone-number phone-numbers)
       query-number-response))



(defn attempt-update! [phone-numbers entry]
  (let [attempt (swap! phone-numbers data/update-if-not-duplicate entry)]
    (if (nil? (data/find-entry attempt entry)) 
      [:duplicate-context entry]
      [:updated entry])))

(defn add-phone-number! [phone-numbers entry]
  (cond 
    (data/duplicate-context? @phone-numbers entry) [:duplicate-context entry]
    (not (s/valid? :phone-number.spec/entry entry)) [:invalid-entry entry]
    (not (data/valid-number-format? entry)) [:invalid-number entry]
    :else (attempt-update! phone-numbers entry)))

(defmulti add-number-response first)

(defmethod add-number-response :updated [[_ entry]]
  (status (response entry) 201))

(defmethod add-number-response :duplicate-context [[_ {:keys [number context]}]]
  (error 409
         (str "An entry for " number " with context " context " aleady exists")))

(defmethod add-number-response :invalid-number [[_ {:keys [number]}]]
  (error 400 (str number " is not a valid phone number")))

(defmethod add-number-response :invalid-entry [_]
  (error 400 (str "Invalid entry must have name, number, and context")))

(defn add-number! [phone-numbers entry]
  (->> entry
       (add-phone-number! phone-numbers)
       add-number-response))
