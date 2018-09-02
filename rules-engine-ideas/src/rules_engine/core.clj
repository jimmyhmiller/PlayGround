(ns rules-engine.core)


;;; Desirable properties

;; Partial matches
;; Explanations why and why not
;; Pure
;; Converable to clara

(def is-important
  '[[[:type :support-request] 
     [:level :high]]
    {:type :important-request
     :request ?e}])

(def notify-client-rep
  '[[[:type :supprt-request]
     [:client ?client]]
    [?client-rep
     [:type :client-representative] 
     [:client ?client]]
    {:type :notify-client
     :client-representative ?client-rep
     :client ?client}])

