(defproject account-number "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-beta4"]
                 [org.clojure/test.check "0.10.0-alpha2"]
                 [orchestra "2017.08.13"]
                 [funcool/cuerdas "2.0.4"]]
  :main ^:skip-aot account-number.main
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
