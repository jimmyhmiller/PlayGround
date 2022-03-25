

(require '[clojure.tools.reader.reader-types :as t])
(require '[clojure.tools.reader :as treader])
(require '[cljs.reader :as reader])
(require '[cljs.tagged-literals :as cljs-tags])
(def fs (js/require "fs"))



(defn read-all
  [input]
  (binding [treader/*data-readers* cljs-tags/*cljs-data-readers*]
    (let [eof #js {}]
      (doall
       (take-while #(not= % eof) (repeatedly #(treader/read input false eof)))))))


(defn read-file [path f]
  (.readFile fs path "utf8"
               (fn [err data]
                 (if err
                   (js/console.error err)
                   (f data)))))


(do (js/console.time "read")
    (read-file "/Users/jimmyhmiller/Downloads/core.cljs"
               (fn [data]
                 (println
                  (count (read-all (t/string-push-back-reader data))))
                 (js/console.timeEnd "read"))))

