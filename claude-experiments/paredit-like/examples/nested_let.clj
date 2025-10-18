; Example with nested let forms that can be merged

(defn calculate [x y]
  (let [sum (+ x y)]
    (let [product (* x y)]
      (let [difference (- sum product)]
        {:sum sum
         :product product
         :difference difference}))))

(defn process-data [data]
  (let [cleaned (remove nil? data)]
    (let [sorted (sort cleaned)]
      (let [unique (distinct sorted)]
        (vec unique)))))
