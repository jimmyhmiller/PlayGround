(ns clara-bug.core
  (:require [clara.rules :as rules]))

(defrecord SupportRequest [client level])

(defrecord ClientRepresentative [name client])

(rules/defrule is-important
  "Find important support requests."
  [SupportRequest (= :high level)]
  =>
  (println "High support requested!"))

(rules/defrule notify-client-rep
  "Find the client representative and send a notification of a support request."

  ;; Adding this :exists causes the firing of rules to fail with ClassNotFoundException
  [:exists [SupportRequest (= ?client client)]]


  [ClientRepresentative (= ?client client) (= ?name name)] ; Join via the ?client binding.
  =>
  (println "Notify" ?name "that"  ?client "has a new support request!"))

;; Run the rules! We can just use Clojure's threading macro to wire things up.
(-> (rules/mk-session)
    (rules/insert (->ClientRepresentative "Alice" "Acme")
                  (->SupportRequest "Acme" :high))
    (rules/fire-rules))
