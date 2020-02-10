(ns noxt.main
  (:require [cljs.build.api :as b]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.set :as set]))

(defn get-path-from-pages [namespace]
  (-> (subs namespace (+ 6 (string/index-of namespace "pages.")))
      (string/replace "." "/")))

(defn generate-loader [file-namespace]
  `[(~'ns ~(symbol (str file-namespace "-loader"))
      (:require [cljs.loader :as ~'loader]
                [~(symbol file-namespace)]))

    (~'defn ~(vary-meta 'main assoc :export true) []
     (~(symbol file-namespace "main")))

    (loader/set-loaded! ~(keyword (get-path-from-pages file-namespace)))])


;; Maybe reading the file is the right answer?
;; Should I really rely on the path?
(defn derive-namespace [path]
  (let [[_ file-path] (string/split path #"/src/")]
    (-> file-path
        (string/replace "/" ".")
        (string/replace ".cljs" ""))))

(defn get-all-page-paths []
  (->> (io/file ".")
       file-seq
       (map #(.getPath %))
       (remove (fn [p] (re-matches #".*\.noxt/.*" p)))
       (filter (fn [p] (re-matches #".*pages/.*.cljs" p)))))

(defn get-all-page-namespaces [paths]
   (map derive-namespace paths))

(defn get-all-page-roots [paths]
  (set
   (map (fn [path]
          (first (string/split path #"/src/")))
        paths)))


(defn derive-modules [namespaces]
  (into {}
        (map (fn [namespace]
               [(keyword (get-path-from-pages namespace)) 
                {:entries #{(symbol (str namespace "-loader"))}
                 :output-to (str ".noxt/" namespace ".js")}])
             namespaces)))



(defn generate-opts [namespaces {:keys [optimizations]}]
  {:output-dir ".noxt/"
   :asset-path ".noxt/"
   :optimizations (or optimizations :none)
   :verbose false
   ;; Consider using closure-defines to dead code eliminate server stuff
   :closure-defines {'noxt.main.path_prefix (subs (first namespaces)
                                                  0
                                                  (+ 5 (string/index-of (first namespaces) "pages.")))}
   :modules (merge
              {:main {:entries '#{noxt.main}
                     :output-to ".noxt/main.js"}}
             (derive-modules namespaces))})

(defn build-project [{:keys [optimizations]}]
  (let [paths (get-all-page-paths)
        namespaces (get-all-page-namespaces paths)
        roots (get-all-page-roots paths)
        loaders (mapv (juxt identity generate-loader) namespaces)
        opts (generate-opts namespaces {:optimizations optimizations})]

    (doseq [[namespace loader] loaders]
      (let [paths (string/split namespace #"\.")]
        (.mkdirs (io/file ".noxt/loaders/src/" (string/join "/" (butlast paths))))
        (spit (str ".noxt/loaders/src/" (string/join "/" paths)  "_loader.cljs")
              (binding [*print-meta* true]
                (string/join "\n" (map str loader))))))
    ;; Need to copy main.cljs and lib.cljs from resources or
    ;; something, because it won't work work when run as main from
    ;; another project.
    ;; Or not? I'm really not sure because this seems to magically
    ;; work in a way I didn't expect.
    
    (b/build (set/union #{".noxt/loaders/"} roots) opts)))


(comment
  (build-project {:optimizations :none})
)



