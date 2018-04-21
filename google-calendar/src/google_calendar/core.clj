(ns google-calendar.core
  (:gen-class)
  (:require [clojure.java.browse :refer [browse-url]])
  (:import
   [com.google.api.client.auth.oauth2 Credential]
   [com.google.api.client.extensions.java6.auth.oauth2 AuthorizationCodeInstalledApp]
   [com.google.api.client.extensions.jetty.auth.oauth2 LocalServerReceiver]
   [com.google.api.client.googleapis.auth.oauth2 GoogleAuthorizationCodeFlow]
   [com.google.api.client.googleapis.auth.oauth2 GoogleClientSecrets GoogleAuthorizationCodeFlow$Builder]
   [com.google.api.client.googleapis.javanet GoogleNetHttpTransport]
   [com.google.api.client.http.javanet NetHttpTransport]
   [com.google.api.client.json JsonFactory]
   [com.google.api.client.json.jackson2 JacksonFactory]
   [com.google.api.client.util DateTime]
   [com.google.api.client.util.store FileDataStoreFactory]
   [com.google.api.services.calendar Calendar Calendar$Builder]
   [com.google.api.services.calendar CalendarScopes]
   [com.google.api.services.calendar.model Event]
   [com.google.api.services.calendar.model Events]
   [java.util Collections]
   [java.io InputStreamReader]))


(def application-name "My Test App")
(def json-factory (JacksonFactory/getDefaultInstance))
(def credentials-folder "credentials")

(def scopes (Collections/singletonList CalendarScopes/CALENDAR_READONLY))
(def client-secret-dir "client-secret.json")

(defn get-credentials [^NetHttpTransport http-transport]
  (let [in (.getResourceAsStream (.getContextClassLoader (Thread/currentThread)) client-secret-dir)
        client-secrets (GoogleClientSecrets/load json-factory (InputStreamReader. in))
        flow (.. (GoogleAuthorizationCodeFlow$Builder. 
                  http-transport json-factory client-secrets scopes)
                 (setDataStoreFactory (FileDataStoreFactory. (java.io.File. credentials-folder)))
                 (setAccessType "offline")
                 (build))]
    (.. 
     (AuthorizationCodeInstalledApp. flow  (LocalServerReceiver.))
     (authorize "user"))))

(defn find-zoom-link [{:keys [:description] :as info}]
  (assoc info
         :zoom-link (re-find #"https://healthfinch.zoom.us/j/[0-9]+" description)))

(defn -main []
  (let [http-transport (GoogleNetHttpTransport/newTrustedTransport)
        service (..
                 (Calendar$Builder. http-transport json-factory (get-credentials http-transport))
                 (setApplicationName application-name)
                 (build))
        now (DateTime. (System/currentTimeMillis))
        events (.. service
                   (events)
                   (list "primary")
                   (setMaxResults (int 10))
                   (setTimeMin now)
                   (setOrderBy "startTime")
                   (setSingleEvents true)
                   (execute))]
    
    (->> events
         .getItems
         seq
         (map bean)
         (map #(select-keys % [:description :summary]))
         (filter :description)
         (map find-zoom-link)
         println)))


