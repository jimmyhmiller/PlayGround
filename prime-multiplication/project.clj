(defproject prime-multiplication "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-beta1"]
                 [org.clojure/spec.alpha "0.1.123"]
                 [org.clojure/test.check "0.10.0-alpha2"]
                 [orchestra "2017.08.13"]
                 [expound "0.3.0"]]
  :main ^:skip-aot prime-multiplication.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
