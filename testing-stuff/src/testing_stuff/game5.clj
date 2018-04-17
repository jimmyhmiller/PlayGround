(ns game5
  (:require [clojure.spec.alpha :as s]
            [expound.alpha :as expound]
            [clojure.core.match :refer [match]]))

(set! s/*explain-out* expound/printer)


(s/def ::name (s/and string? not-empty))

(s/def ::weapon
  #{:sword :bow :staff})

(s/def ::class
  #{:warrior :wizard :ranger})

(s/def ::player
  (s/keys :req [::weapon ::class ::name]))

(def player 
  {::weapon :swor
   ::class :warrior
   ::name "Sir Willamet"})


(s/fdef describe-weapon
  :args (s/cat :player ::player)
  :ret string?)

(defn describe-weapon [{:keys [::weapon ::class]}]
  (match [weapon class]
         [:staff :wizard] "You must be old."
         [:sword :warrior] "You know how to use that sword"
         :else (str "That's a nice " (name weapon) ".")))




