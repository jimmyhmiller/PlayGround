(ns noxt.core
  (:require [cljs.build.api :as b]
            [clojure.java.io :as io]
            [clojure.string :as string]))


(defn get-modules []
  (->> "pages"
       io/file
       file-seq
       (filter #(.isFile %))
       (map #(.getPath %))
       (map #(string/replace (string/replace (string/replace % #"^pages/" "") #"/" ".") #"\.cljs" ""))
       (map (fn [page] {(keyword page) {:entries #{(symbol page)}
                                        :output-to (str "out/" page ".js")}}))
       (reduce merge {})))

(def opts
  {:output-dir "out"
   :asset-path "/out"
   :optimizations :advanced
   :modules (merge
             {:main {:entries '#{noxt.main}
                     :output-to "out/main.js"}}
            (get-modules))
   :source-maps true})


(b/build #{"src" "pages"} opts)

