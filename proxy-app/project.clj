(defproject proxy-app "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :repositories [["jitpack" "https://jitpack.io"]]
  :dependencies [[org.clojure/clojure "1.9.0-beta2"]
                 [org.littleshoot/littleproxy "1.1.2" :exclusions [io.netty/netty-all]]
                 [com.github.oubiwann/net "95a1bdb74b70f5b90a414335e841a8a57a6d4628"]]
  :main ^:skip-aot proxy-app.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
