;; This buffer is for text that is not saved, and for Lisp evaluation.
;; To create a file, visit it with C-x C-f and enter text in its buffer.

(def current-state (atom :test))

(def connections 
  [[:test :thing :test1]
   [:test1 :thing :test2]])

(defn start-points [connections]
  (clojure.set/difference 
   (set (map first connections))
   (set (map last connections))))

(defn end-points [connections]
  (clojure.set/difference 
   (set (map second connections))
   (set (map first connections))))

(defn transform-connection [[_ val to]]
  {val to})

(defn combine-connections [connections]
  (->> connections
       (map transform-connections)
       (reduce merge)))

(defn map-vals [f [key val]]
  [key (f val)])

(defn connections->map [connections]
  (->> connections
       (group-by first)
       (map (partial map-vals combine-connections))
       (into {})))

(defn find-transitions [connection-map current-state]
  (keys (current-state connection-map)))

(defn transition [connection-map current-state val]
  (-> connection-map current-state val))




(def connections-map 
  (connections->map connections))


