(ns graph.core
  (:use rhizome.viz)
  (:use rhizome.dot))


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




(comment
 (def g
  {:unmatched [:validate :exception]
   :validate [:exception :inventory]
   :exception [:validate :shipToState]
   :inventory [:exception :validate :reassign]
   :reassign [:exception :shipToBuyer]
   :shipToState [:validate]
   :shipToBuyer [:archive]
   :archive []}))



(def g
  (-> {}
      (--> :CounterRelease [:Released-at-Counter :Exception])
      (--> :Exception [:Ship-to-State])
      (--> :Inventory [:Exception :Reassign :Ship-to-Seller])
      (--> :Pre-Receipt [:Validate])
      (--> :Reassign [:Exception :Ship-to-Buyer :CounterRelease])
      (--> :Ship-to-Buyer [:Shipped-to-Buyer :Exception])
      (--> :Ship-to-Seller [:Shipped-to-Seller :Exception])
      (--> :Unmatched [:Validate])
      (--> :Validate [:Inventory :Exception])
      (--> :Verify [:Unmatched :Validate])
      (--> :Shipped-to-State [:Validate])
      (--> :Ship-to-State [:Shipped-to-State :Exception])
      (--> :Shipped-to-Buyer [:Archive])
      (--> :Shipped-to-Seller [:Archive])
      (--> :Released-at-Counter [:Archive])))


(save-image
 (graph->image (keys g) g
               :node->descriptor (fn [n] {:label (name n) :shape "plaintext"})
               :options {:layout "dot"})
 "graph2.png")

(view-graph (keys g) g
            :node->descriptor (fn [n] {:label (name n)})
            :options {:layout "circo"})
