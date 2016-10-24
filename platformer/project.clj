(defproject platformer "1.2.3"
  :plugins [[lein-cljsbuild "1.1.2"]]
    :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/clojurescript "1.9.89"]
                 [org.clojure/core.async "0.1.267.0-0d7780-alpha"]
                 [om "0.5.0"]
                 [sablono "0.2.16"]]
  :cljsbuild {
    :builds [{
        ; The path to the top-level ClojureScript source directory:
        :source-paths ["src"]
        ; The standard ClojureScript compiler options:
        ; (See the ClojureScript compiler documentation for details.)
        :compiler {
          :optimizations :whitespace
          :pretty-print true}}]})
