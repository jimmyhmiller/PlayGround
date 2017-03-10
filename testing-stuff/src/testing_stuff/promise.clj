(ns testing-stuff.promise
  (:require [promesa.core :refer [promise then bind]]))


(defmacro async-let
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


(async-let [user (get-user-p 1)
            address (get-address-p (:address user))]
           [user address])




(defn getUserAndAddress [id]
  (let [user (getUser id)
        address (getAddress (user :id))]
    (merge user address)))

(defmacro fn+ [bindings & body]
  `(fn [{:keys ~bindings}] ~@body))


(def add-point2 
  (fn+ [x y] 
       (+ x y)))


(add-point2 {:x 1 :y 2 :z 3})



      
