(ns testing-stuff.weapon-example
  (:require [clojure.spec.alpha :as s]
            [expound.alpha :as expound]
            [clojure.spec.test.alpha :as stest]
            [clojure.core.match :refer [match]]
            [clojure.core.async :as async]))


(defn long-blocking-operation [x]
  (Thread/sleep 10000))

(deref (future (long-blocking-operation 1)) 1000 0)

(->> (repeatedly #(rand-int 5))
     (take 100)
     (frequencies))

(defprotocol Speak
  (speak [this]))

(defrecord Bird []
  Speak
  (speak [this] "tweet"))

(defrecord Dog []
  Speak
  (speak [this] "bark"))

(speak (Bird.)) ; tweet
(speak (Dog.)) ; bark




(import '(com.google.gson Gson JsonPrimitive JsonObject JsonArray))

(defprotocol ToJson
  (toJson [this]))

(extend-protocol ToJson
  java.lang.Number
  (toJson [this] (JsonPrimitive. this))
  
  java.lang.String
  (toJson [this] (JsonPrimitive. this))
  
  java.lang.Boolean
  (toJson [this] (JsonPrimitive. this))
  
  clojure.lang.IPersistentVector
  (toJson [this] 
    (reduce (fn [arr x]
              (doto arr (.add (toJson x))))
            (JsonArray.) this))
  
  clojure.lang.IPersistentMap
  (toJson [this] 
    (let [obj (JsonObject.)]
      (doseq [[key value] this]
        (.add obj (name key) (toJson value)))
      obj)))


(str (toJson [1 2 "1234" true]))


(toJson "asdf")

(use '[clojure.pprint :only [print-table]])

(require '[clojure.reflect :as r])


(require '[clj-java-decompiler.core :refer [decompile]])

(->> (range 100)
     (filter even?)
     (map (fn [x] (+ x 2)))
     (reduce +))


(defn wait-100ms [x]
  (Thread/sleep 100)
  x)

(time (count (pmap wait-100ms (range 100))))



(set! s/*explain-out* expound/printer)


(s/def ::weapon #{:sword :bow :staff})
(s/def ::class #{:wizard :ranger :rogue})
(s/def ::name (s/and string? not-empty))

(s/def ::player 
  (s/keys :req [::class 
                ::weapon 
                ::name]))

(s/fdef describe-weapon
        :args (s/cat :player ::player)
        :ret string?)


(defn describe-weapon [{:keys [::weapon ::class]}]
  (match [weapon class]
         [:bow :ranger] "Bow for a ranger"
         [:sword :wizard] "A wizard with a sword?"
         [w c] (str "That's a nice " 
                    (name w) " for a " 
                    (name c))))

(defn fetch-player [id]
  (Thread/sleep 100)
  (ffirst (s/exercise ::player 1)))

(def players
  (doall (pmap fetch-player (range 1000))))

(describe-weapon 
 {::weapon :sword
  ::name "asdf"
  ::class :ranger})



