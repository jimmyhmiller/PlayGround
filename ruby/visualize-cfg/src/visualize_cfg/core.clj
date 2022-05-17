(ns visualize-cfg.core
  (:require [cheshire.core :as json]
            [clojure.string :as string])
  (:import [java.awt.datatransfer DataFlavor StringSelection Transferable]))


(def cfgs (->> "/Users/jimmyhmiller/Downloads/control_flow2.json"
              slurp
              string/split-lines
              (map #(json/parse-string % true))))

(defn clipboard []
  (.getSystemClipboard (java.awt.Toolkit/getDefaultToolkit)))


(defn spit-clip [_file text]
  (let [selection (StringSelection. text)]
    (.setContents (clipboard) selection selection)))

(defn q [x]
  (str "\"" x "\""))


(do

  (defn make-graph [cfg]
    (str "\n"
         "\n"
         (string/join "\n" (map (fn [{:keys [hash instructions]}]
                                  (str (q hash) "[label=" (q(get-instruction-text instructions)) ", shape=\"square\"]"))
                                (:blocks cfg)))

         "\n"
         (string/join "\n"
                      (map (fn [x]
                             (str (q x) "[label=stub]"))
                       (filter (comp #{12} count) (map second (mapcat :edge_list (:blocks cfg))))))


         "\n\n"
         (string/join "\n"
                      (map (fn [[x y]] (str (q x) " -> " (q y) )) (mapcat :edge_list (:blocks cfg))))

         

         ))
  
  (defn get-instruction-text [instructions]
    (str
     (string/join "\\l" (map (fn [{:keys [type value]}]
                               (case type
                                 "comment" (str "# " value)
                                 value))
                             instructions))
     "\\l"))


  (def cfg (second cfgs))

  (spit "/Users/jimmyhmiller/Downloads/cfg.dot"
        (str
         "digraph CFG {\n"
         "graph [splines=ortho, nodesep=2, ranksep=1]\n"
         "pad=1\n"
         "bgcolor=white\n"
         (string/join "\n" (map make-graph cfgs))
         "\n}")
        ))


(first cfg)

(count cfg)
