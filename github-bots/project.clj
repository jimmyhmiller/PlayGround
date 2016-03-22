(defproject github-bots "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[lein-git-deps "0.0.1-SNAPSHOT"]]
  :git-dependencies [["https://github.com/fversnel/DnDDice.git"]]
  :source-paths [".lein-git-deps/DnDDice/src" "src"]
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [instaparse "1.4.1"]
                 [tentacles "0.5.1"]
                 [clj-yaml "0.4.0"]])
