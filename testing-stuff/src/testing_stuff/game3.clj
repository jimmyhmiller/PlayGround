(ns game3
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [clojure.pprint :as pprint]
            [clojure.spec.test.alpha :as stest]
            [expound.alpha :as expound]
            [clojure.core.match :refer [match]]
            [fsm]))

(set! s/*explain-out* expound/printer)


(s/def ::weapon #{:sword :bow :staff})
(s/def ::class #{:ranger :warrior :wizard})
(s/def ::name (s/and string? not-empty))


(s/def ::player
  (s/keys :req [::weapon
                ::class
                ::name]))

(s/fdef describe-weapon
  :args (s/cat :player ::player)
  :ret string?)

(stest/instrument `describe-weapon)


(defn describe-weapon [{:keys [::weapon ::class]}]
  (match [class weapon]
         [:warrior :sword] "Bet you know how to use that"
         [:wizard :staff] "Old man!"
         :else (str "This is a " (name weapon))))


(defmacro defmemo [name args & body]
  `(def ~name (memoize (fn ~args ~@body))))

(defmemo fetch-player [id]
  (Thread/sleep 100)
  (ffirst (s/exercise ::player)))

(defn fetch-players [num]
  (pmap fetch-player (range num)))

(fsm/view {:start {:attack :phase-2
                   :move :phase-2
                   :run-away :end
                   :grapple :grapple-start}
           :phase-2 {:attack :end
                     :free :phase-2
                     :time-warp :start}
           :grapple-start {:attempt :resolve-attempt
                           :countered :phase2}
           :resolve-attempt {:failed :end
                             :succeeded :end}})
