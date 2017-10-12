(defproject jmx-clojure "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/java.jmx "0.3.4"]
                 [incanter "1.5.7"]
                 [clj-time "0.14.0"]
                 [live-chart "0.1.1"]
                 [dorothy "0.0.6"]]
  :jvm-opts ["-XX:+UseG1GC"])
