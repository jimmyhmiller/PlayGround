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
       ;; Need to generate files instead of finding them
       ;; Can I do that in memory instead of on the file system?
       (filter (fn [p] (re-matches #".*_loader.cljs$" p)))
       (map #(string/replace (string/replace (string/replace % #"^pages/" "") #"/" ".") #"\.cljs" ""))
       (map (fn [page] {(keyword (string/replace page #"_loader" ""))
                        {:entries #{(symbol (string/replace page #"_" "-"))}
                         :output-to (str "public/js/page/" page ".js")}}))
       (reduce merge {})))

(def opts
  {:output-dir "public/js"
   :asset-path "js"
   :optimizations :advanced
   :main "noxt.main"
   :verbose false
   :warnings {:single-segment-namespace false}
   :modules (merge
             {:main {:entries '#{noxt.main}
                     :output-to "public/js/main.js"}}
            (get-modules))})


(defn -main []
  (b/watch (b/inputs "src" "pages") opts))

(comment 
  (b/build (b/inputs "src" "pages" ) opts)
)
