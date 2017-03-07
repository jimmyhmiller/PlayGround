(ns testing-stuff.promise
  (:require [promesa.core :refer [promise then bind]]))


(defmacro alet
  [bindings & body]
  (->> (reverse (partition 2 bindings))
       (reduce (fn [acc [l r]]
                 `(bind (promise ~r) (fn [~l] ~acc)))               
               `(promise (do ~@body)))))


(defn get-user [id]
  {:name "jimmy" :address 2})

(defn get-address [id]
  {:street "123 Easy Street"
   :city "Indianapolis"
   :state "Indiana"})


(defn get-user-p [id] 
  (promise (get-user id)))

(defn get-address-p [id] 
  (promise (get-address id)))


(alet [user (get-user-p 1)
        address (get-address-p (:address user))]
       [user address])
