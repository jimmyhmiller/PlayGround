
(require '[clojure.string :as string])

(defn get-categories [data]
  (let [categories 
        (-> data
            (string/replace #"(?s)DM-A.*?Fumbles\W+" "-------")
            (string/replace #"\n+" "\n")
            (string/split #"-------"))]
    (map #(string/split % #"\n") categories)))

(defn get-category-name [category]
  (-> category
      first
      (string/split #" ")
      first
      string/lower-case
      keyword))

(defn determine-punctuation [name punct]
  (if (= punct "-")
    name
    (str name punct)))

(defn get-entry-info [line]
  (let [[_ name punct description] 
        (re-find #"^[0-9]+\. (.*?)\s*([.,\/#!$%?\^&\*;:{}=\-_`~()]+)\s*(.*?)$" line)]
    {:name (determine-punctuation name punct)
     :description description}))

(defn get-entries [category]
  (->> category
       (drop 1)
       (map get-entry-info)))

(defn get-category-info [category]
  (let [name (get-category-name category)
        entries (get-entries category)]
    {name entries}))

(defn get-major-sections [categories]
  (->> categories
       (map get-category-info)
       (partition 4)
       (map #(apply merge %))))

(defn main [path]
  (let [[criticals fumbles]
        (->> path
             slurp
             (get-categories)
             (get-major-sections))]
    {:criticals criticals
     :fumbles fumbles}))

(def file-location "/Users/jimmy/Downloads/criticalhits.txt")

(main file-location)
