(ns datafrog-clj.core)

;https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md

(defn gallop [input pred]
  (drop-while pred input)) ; TODO

(defn join-helper [input1 input2 results n] 
  (if (or (empty? input1) 
          (empty? input2)) results
      (case (compare (ffirst input1) (ffirst input2))
        -1 (recur (gallop input1 (fn [x] (< (first x) (ffirst input2))))
                  input2
                  results
                  (inc n))
        0 (let [coll1 (take-while (fn [x] (= (first x) (ffirst input1))) input1)
                coll2 (take-while (fn [x] (= (first x) (ffirst input2))) input2)]
            (recur (drop (count coll1) input1)
                   (drop (count coll2) input2)
                   (concat results 
                           (for [x coll1 y coll2] 
                             [(ffirst input1) (second x) (second y)]))
                   (inc n)))
        1 (recur input1 (gallop input2 (fn [x] (< (first x) (ffirst input1)))) results (inc n)))))



(join-helper [[:user 2] [:user 4]] [[:user 3] [:thing 2]] [] 0) 

(defn join-into [input1 input2 f]
  (let [coll1 (map f (mapcat #(join-helper (:recent input1) % []) (:stable input2)))
        coll2 (map f (mapcat #(join-helper (:recent input2) % []) (:stable input1)))
        coll3 (map f (join-helper (:recent input1) (:recent input2)))]
    (concat coll1 coll2 coll3)))

