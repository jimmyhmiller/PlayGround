(ns wander.core22
  (:require [cheshire.core :as json]
            [clojure.string :as string]
            [clojure.data]))




(->> (slurp "/Users/jimmyhmiller/Documents/Code/yjit-bench/connections.json")
     (string/split-lines)
     (map #(json/decode % true))
     (mapcat (fn [[from connections]]
               (for [connection connections]
                 (when connection
                   (str (:iseq from) "-" (:idx from)
                        " -> "
                        (:iseq connection) "-" (:idx connection))))))
     (filter identity)
     (string/join "\n")
     (spit "/Users/jimmyhmiller/Documents/Code/yjit-bench/connections.txt"))



(def data
  (->> (slurp "/Users/jimmyhmiller/Documents/Code/yjit-bench/yjit.log")
       (string/split-lines)
       (map #(json/decode % true))))


(->> data
     (mapcat :cme_dependencies)
     (map :receiver_klass)
     (frequencies)
     (sort-by second)
     reverse)

(->> data
     (mapcat :class_names)
     frequencies
     (sort-by second)
     reverse)

(->> data
     (map (comp hash :ctx))
     (frequencies)
     (sort-by second)
     reverse)

(->> data
     (map (comp :location))
     frequencies
     (sort-by second)
     reverse)


(defn diff-lines [[string1 string2]]
  (clojure.data/diff (string/split-lines string1)
                     (string/split-lines string2)))


(->> data
     (filter (comp #{7} count :outgoing)))


(->> data
     (filter (comp #{3} count :cme_dependencies))
     (take 2))


(->> data
     (filter (comp #{{:iseq 4790796160, :idx 3}} :block_id))
     (filter (comp #{2644} count :disasm))
     (take 2)
     (apply clojure.data/diff))

(->> data
     (filter (comp #{{:iseq 4790796160, :idx 3}} :block_id))
     (filter (comp #{2648} count :disasm))
     (take 2))
