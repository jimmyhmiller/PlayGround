(ns image-transformations.core
  (:require [mikera.image.core :as image]
            [mikera.image.filters :as filters]
            [mikera.image.colours :as col])
  (:import [org.jtransforms.dct DoubleDCT_2D]))

(defn transform-image [image-path]
  (-> image-path
      (image/load-image)
      (image/resize 32 32)
      (image/filter-image (filters/grayscale))))

(defn image->array [image]
  (->> image
       image/get-pixels
       (partition 32)
       (map double-array)
       into-array))

(defn dct [image-array] 
  (. (DoubleDCT_2D. 32 32) forward image-array true)
  image-array)

(defn reduce-image [image-array] 
  (map #(take 8 %) (take 8 image-array)))
  
(defn average [coll]
  (/ (reduce + coll) (count coll)))

(defn find-average [reduced-image] 
  (average (flatten reduced-image)))

(defn to-bits [avg num]
  (if (> avg num) 1 0))

(defn calculate-hash [reduced-image avg]
  (->> reduced-image
       flatten
       (map (partial to-bits avg))))

(defn phash [image-path]
  (let [reduced-image 
        (->> image-path
             transform-image
             image->array
             dct
             reduce-image)
        avg (find-average reduced-image)]
    (calculate-hash reduced-image avg)))

; https://stackoverflow.com/questions/13063594/how-to-filter-a-directory-listing-with-a-regular-expression-in-clojure

(defn regex-file-seq
  [re dir]
  (filter #(re-find re (.getPath %)) (file-seq dir)))

(comment
  (time
   (->> "/Users/jimmy/Desktop/images/" 
        clojure.java.io/file
        (regex-file-seq #".*\.jpg")
        (map phash)
        count)))

(defn -main [& args]
  (->> "/Users/jimmy.miller/Desktop/images/" 
       clojure.java.io/file
       (regex-file-seq #".*\.jpg")
       (pmap phash)
       last)
  (shutdown-agents))

(comment
  (frequencies 
   (map bit-xor
        (phash "/Users/jimmy/Desktop/wadler.jpg")
        (phash "/Users/jimmy/Desktop/wadler2.jpg"))))
  
