(defproject ping-experiment "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [codax "1.1.0-SNAPSHOT"]
                 [clj-time "0.14.0"]
                 [incanter "1.9.1"]
                 [incanter "1.9.1"]
                 [incanter "1.5.7"]]
  :main ^:skip-aot ping-experiment.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
