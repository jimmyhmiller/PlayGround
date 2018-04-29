(defproject s3-instant-rest "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [com.amazonaws/aws-java-sdk "1.11.318"]
                 [amazonica "0.3.121" :exclusions
                  [[com.amazonaws/aws-java-sdk]]]
                 [honeysql "0.9.2"]
                 [cheshire "5.8.0"]
                 [org.clojure/java.data "0.1.1"]])
