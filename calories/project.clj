(defproject calories "0.1.0-SNAPSHOT"
  :dependencies [[org.clojure/clojure "1.10.0-alpha6"]
                 [cheshire "5.8.0"]]
  :main ^:skip-aot calories.core
  :profiles {:uberjar {:aot :all}})
