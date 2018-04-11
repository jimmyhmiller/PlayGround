(ns game2
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [expound.alpha :as expound]
            [clojure.core.match :refer [match]]))


(set! s/*explain-out* expound/printer)

(s/def ::name
  (s/and string?
         not-empty))

(s/def ::class
  #{:ranger
    :warrior
    :wizard})

(s/def ::weapon
  #{:bow
    :sword
    :staff
    :short-sword})

(s/def ::player
  (s/keys :req [::name
                ::weapon
                ::class]))


(defn describe-weapon [{:keys [::weapon ::class]}]
  (match [class weapon]
         [:wizard :sword] "A wizard with sword?"
         [:warrior :bow] "Don't you want a sword?"
         [:ranger :staff] "Silly ranger!"
         [_ weapon] (str "That's a nice " (name weapon))) )


(defmacro defmemo [name args & body]
  `(def ~name
     (memoize
      (fn ~args ~@body))))

(defmemo fetch-player [id]
  (Thread/sleep 100)
  (ffirst (s/exercise ::player 1)))


(defmemo really-long [id]
  (Thread/sleep 1000)
  id)

(defn fetch-players [n]
  (pmap fetch-player (range n)))
