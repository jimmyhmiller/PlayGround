(ns graph.core
  (:use rhizome.viz)
  (:use rhizome.dot)
  (:require [fsmviz.core :as fsmviz]))


(defn conj-cat [coll elem]
  (if (sequential? elem)
    (concat coll elem)
    (conj coll elem)))


(defn add-node [graph node]
  (if (sequential? node)
    (reduce add-node graph node)
    (let [old-node (node graph)]
      (if (nil? old-node)
        (assoc graph node [])
        graph))))

(defn --> [graph from to]
  (let [old-node (from graph)]
    (-> graph
        (add-node to)
        (assoc from (conj-cat old-node to)))))




(spit "fsm.dot" (fsmviz/generate-image g "example-map"))

(comment (def g 
           (-> {}
               (--> :if1 [:end :take-stuff])
               (--> :take-stuff [:call-get :call-take])
               (--> :call-get [:if1])
               (--> :call-take [:if1]))))

(comment (save-image
          (graph->image (keys g) g
                        :node->descriptor (fn [n] {:label (name n) :shape "plaintext"})
                        :options {:layout "dot"})
          "graph2.png"))

(view-graph (keys g) g
            :node->descriptor (fn [n] {:label (name n)})
            :options {:layout "dot"})
