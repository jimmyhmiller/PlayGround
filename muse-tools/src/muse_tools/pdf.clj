(ns muse-tools.pdf
  (:require [pdf-to-images.core :as pdf]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.java.shell :as sh]))


(defn export-first-page [file]
  (try
    (pdf/pdf-to-images file
                       pdf/image-to-file
                       :start-page 0
                       :end-page 1
                       :dpi 120)
    nil
    (catch Throwable e
      (println "failed" file)
      file)))


(do
  (def failed-pdf-conversions
    (->> (file-seq (io/file "/Users/jimmyhmiller/Downloads/Programming 6/files/"))
         (filter (fn [f] (string/ends-with? (.getName f) ".pdf")))
         (map export-first-page)
         (filter identity)
         doall
         (map #(.getAbsolutePath %))
         set))



  (def fail-pngs
    (->> (file-seq (io/file "/Users/jimmyhmiller/Downloads/Programming 6/files/"))
         (filter (fn [f] (string/ends-with? (.getName f) ".png")))
         (map #(sh/sh "pngquant" "--force" "--ext=.png" (.getAbsolutePath %)))
         (remove (comp zero? :exit))
         set
         doall))


  (def failed-deleted-pdfs
    (->> (file-seq (io/file "/Users/jimmyhmiller/Downloads/Programming 6/files/"))
         (filter (fn [f] (string/ends-with? (.getName f) ".pdf")))
         (remove (comp failed-pdf-conversions #(.getAbsolutePath %)))
         (map #(sh/sh "rm" (.getAbsolutePath %)))
         (remove (comp zero? :exit))
         doall)))

;; TODO
;; abstract out path
;; move files after done




