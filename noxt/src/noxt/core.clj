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
       (filter (fn [p] (re-matches #".*cljs$" p)))
       (map #(string/replace (string/replace (string/replace % #"^pages/" "") #"/" ".") #"\.cljs" ""))
       (map (fn [page] {(keyword page) {:entries #{(symbol page)}
                                        :output-to (str "public/js/" page ".js")}}))
       (reduce merge {})))

(def opts
  {:output-dir "public/js"
   :asset-path "js/"
   :optimizations :advanced
   :warnings {:single-segment-namsespace false}
   :modules (merge
             {:main {:entries '#{noxt.main}
                     :output-to "public/js/main.js"}}
            (get-modules))})


(defn -main []
  (b/watch (b/inputs "src" "pages") opts))

