(ns testing-stuff.game4
  (:require [clojure.spec.alpha :as s]
            [expound.alpha :as expound]
            [clojure.core.match :refer [match]]))

(set! s/*explain-out* expound/printer)



(s/def ::name (s/and string? #(> (count %) 3)))

(s/def ::weapon #{:sword :bow :staff})

(s/def ::class #{:warrior :ranger :wizard})


(s/def ::player
  (s/keys :req [::name ::weapon ::class]))


(def player 
  {::name "Sir Willamet"
   ::weapon :sword
   ::class :warrior})


(s/fdef describe-weapon
  :args (s/cat :player ::player)
  :ret string?)

(defn describe-weapon [{:keys [::weapon ::class]}]
  (match [weapon class] 
         [:sword :warrior] "Bet you can used that thing!"
         [:staff :wizard] "You must be a wizard!"
         :else (str "That's a nice " (name weapon))))

(defmacro defmemo [name args & body]
  `(def ~name
     (memoize
      (fn ~args
        ~@body))))


(defmemo fetch-player [id]
  (Thread/sleep 100)
  (ffirst (s/exercise ::player)))

(fetch-player 100000)

(defmemo fetch-players [num]
  (pmap fetch-player (range num)))




