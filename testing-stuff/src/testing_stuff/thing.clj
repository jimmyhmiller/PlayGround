(ns testing-stuff.thing
  (:require [clojure.string :as string]
            [clj-fuzzy.metrics :as match]))

(defn clean-image [image]
  (-> image
      (string/replace #"\.[a-zA-Z]{3}$" "")
      (string/replace #"[_-]" " ")))

(defn read-file [file-name]
 (string/split (slurp file-name) #"\n"))

(defn get-all-distances [names images]
  (for [name names
        image images]
    {:name name
     :image image
     :similarity (match/dice name (clean-image image))}))

(defn find-most-similar [coll]
  (apply max-key :similarity coll))

(defn find-images [names images]
  (->> (get-all-distances names images)
       (group-by :name)
       (map (juxt first (comp find-most-similar second)))
       (map second)
       (filter (fn [x] (> (:similarity x) 0.9)))))

(defn format-row [{:keys [name image similarity]}]
  (str name "," image "," similarity))

(defn -main [name-file image-file output-file]
  (let [names (read-file name-file)
        images (read-file image-file)]

    (->> (find-images name-file image-file)
         (map format-row)
         (string/join "\n")
         (spit output-file))))

(comment
  (def root "/Users/jimmyhmiller/Desktop/tmp/")
  (def name-file (str root "/school-names.txt"))
  (def image-file (str root "/school-images.txt"))


  (-main name-file image-file (str root "/names-and-images.txt")))
