(ns integrant-enso-trial.core
  (:require [integrant.core :as ig]))

(def config 
  {[:enso/command :foo/open]
   {:name "open" :help-text "opens things" }

   [:enso/command :foo/go]
   {:name "go" :help-text "goes to things"}

   [:enso/suggestor :foo/open-applications]
   {:handler (ig/ref :foo/open-applications-handler)
    :command (ig/ref :foo/open)}

   [:enso/suggestor :foo/open-applications2]
   {:handler (ig/ref :foo/open-applications-handler)
    :command (ig/ref :foo/go)}
   
   :foo/open-applications-handler nil})




(defmethod ig/init-key :enso/command [_ data] 
  (assoc data :suggestors (atom [])))

(defmethod ig/init-key :enso/suggestor [_ {:keys [command handler] :as data}]
  (swap! (:suggestors command) conj handler)
  handler)

(defmethod ig/init-key :foo/open-applications-handler [_ {:keys [command]}]
  (fn [suggestion]
    [1 2 3 4]))

(def system (ig/init config))



