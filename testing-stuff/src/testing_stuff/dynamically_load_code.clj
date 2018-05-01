(ns testing-stuff.dynamically-load-code
  (:require [clojure.tools.deps.alpha :as deps]
            [clojure.java.io :as io]
            [clojure.string :as string]))


(def my-deps
  '{:deps 
   {hello-clojure 
    {:git/url 
     "https://gist.github.com/ce5b487d92d0a9f594346da94a299f57.git" 
     :sha "b9527df7ab308ceb5116f4e04083dbf7608ed857"}}})

(def resolved
  (deps/resolve-deps my-deps {}))

(def classpath
  (-> resolved
      (deps/make-classpath "." {})
      (string/replace ".:" "")))

(->> classpath
     io/file
     file-seq
     (filter #(string/ends-with? % ".clj"))
     (map #(.getAbsolutePath %))
     (map load-file)
     doall)


