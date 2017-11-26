(defproject phone-number "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :min-lein-version "2.0.0"
  :dependencies [[org.clojure/clojure "1.9.0-RC1"]
                 [compojure "1.6.0"]
                 [ring/ring-defaults "0.2.1"]
                 [org.clojure/data.csv "0.1.4"]
                 [ring/ring-json "0.4.0"]
                 [org.clojure/data.json "0.2.6"]
                 [orchestra "2017.11.12-1"]
                 [ring/ring-spec "0.0.3"]
                 [com.cognitect/transcriptor "0.1.5"]
                 [org.clojure/test.check "0.9.0"]
                 [com.googlecode.libphonenumber/libphonenumber "7.4.4"]]
  :plugins [[lein-ring "0.9.7"]]
  :ring {:handler phone-number.routes/app}
  :profiles
  {:dev {:dependencies [[javax.servlet/servlet-api "2.5"]
                        [ring/ring-mock "0.3.0"]]}})
