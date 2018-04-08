(ns game
  (:require [clojure.spec.alpha :as s]
            [clojure.string :as string]
            [expound.alpha :as expound]
            [fsm]))


(set! s/*explain-out* expound/printer)

(s/def ::name (s/and string? not-empty))
(s/def ::weapon #{:sword :bow :staff})
(s/def ::class #{:warrior :wizard :rogue})

(s/def ::player 
  (s/keys :req [::name 
                ::weapon
                ::class]))

(def player 
  {::weapon :sword
   ::name "jimmy"
   ::class :warrior})


(defn fetch-player [id]
  (Thread/sleep 100)
  (ffirst (s/exercise ::player 1)))

(defn get-all-players []
  (pmap fetch-player (range 1000)))


(let [valid-moves
      {:start {:attack :action2
               :move :action2}
       :action2 {:attack :finish}}]
  (fsm/view-fsm valid-moves))





