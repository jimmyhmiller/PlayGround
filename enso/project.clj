(defproject enso "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.1stleg/jnativehook "2.0.3"]
                 [seesaw "1.4.4"]]
  :source-paths      ["src/clojure"]
  :java-source-paths ["src/java"]
  :main enso.core
  :plugins [[lein-exec "0.3.3"]])
