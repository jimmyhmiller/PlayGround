(ns testing-stuff.weapon-example
  (:require [clojure.spec.alpha :as s]
            [expound.alpha :as expound]
            [clojure.spec.test.alpha :as stest]
            [clojure.core.match :refer [match]]
            [clojure.core.async :as async]))





(set! s/*explain-out* expound/printer)


(s/def ::weapon #{:sword :bow :staff})
(s/def ::class #{:wizard :ranger :rogue})
(s/def ::name (s/and string? not-empty))
(s/def ::strength (s/int-in 0 10))

(s/def ::player 
  (s/keys :req [::class 
                ::weapon 
                ::name
                ::strength]))

(s/fdef describe-weapon
        :args (s/cat :player ::player)
        :ret string?)


(s/exercise-fn `describe-weapon)

(defn describe-weapon [{:keys [::weapon ::class]}]
  (match [weapon class]
         [:bow :ranger] "Bow for a ranger"
         [:sword :wizard] "A wizard with a sword?"
         [w c] (str "That's a nice " 
                    (name w) " for a " 
                    (name c))))

(s/fdef attack
        :args (s/cat :weapon (s/keys :req [::weapon]))
        :ret string?)

(defmulti attack ::weapon)

(defmethod attack :sword [_]
  "Swing the sword!")

(defmethod attack :staff [_]
  "This is a magical staff")

(defmethod attack :bow [_]
  "Hits from far way")

(attack {::weapon :sword})

(defn fetch-player [id]
  (Thread/sleep 100)
  (ffirst (s/exercise ::player 1)))

(def players
  (doall (pmap fetch-player (range 1000))))

(describe-weapon 
 {::weapon :sword
  ::name "asdf"
  ::class :ranger})



