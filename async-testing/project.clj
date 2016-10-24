(defproject async-testing "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha10"]
                 [org.clojure/core.async "0.2.385"]
                 [http-kit "2.1.18"]]
  :main ^:skip-aot async-testing.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
