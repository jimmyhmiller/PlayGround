(defproject google-calendar "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :main google-calendar.core
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [com.google.api-client/google-api-client "1.23.0"]
                 [com.google.oauth-client/google-oauth-client-jetty "1.23.0"]
                 [com.google.apis/google-api-services-calendar "v3-rev305-1.23.0"]])
