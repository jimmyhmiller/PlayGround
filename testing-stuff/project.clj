(defproject testing-stuff "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[lein-environ "1.1.0"]]
  :repositories {"xero" "https://raw.github.com/XeroAPI/Xero-Java/mvn-repo/"
                 "my.datomic.com" {:url "https://my.datomic.com/repo"
                                   :creds :gpg}}
  :dependencies [[org.clojure/math.numeric-tower "0.0.2"]
                 [org.clojure/clojure "1.9.0-alpha11"]
                 [org.clojure/core.async "0.2.385"]
                 [graphql-clj "0.1.18"]
                 [org.clojure/test.check "0.9.0"]
                 [juxt/dirwatch "0.2.3"]
                 [funcool/promesa "1.8.0"]
                 [org.clojure/core.match "0.3.0-alpha4"]
                 [camel-snake-kebab "0.4.0"]
                 [com.stripe/stripe-java "4.7.0"]
                 [org.clojure/java.data "0.1.1"]
                 [clj-http "3.5.0"]
                 [cheshire "5.7.1"]
                 [speculate "0.3.0-SNAPSHOT"]
                 [clj-time "0.13.0"]
                 [com.xero/xero-java-sdk "0.4.2"]
                 [clj-webdriver "0.7.2"]
                 [org.seleniumhq.selenium/selenium-java "3.4.0"]
                 [environ "1.1.0"]
                 [com.datomic/datomic-pro "0.9.5561.50"]
                 [com.faunadb/faunadb-java "1.0.0"]])
