(ns query.core2
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.column :as col]
            [tech.v3.dataset.column-index-structure :as index]
            [tech.v3.dataset.join :as join]))


(def dataset (ds/->dataset (into  (into [] (repeat 1000 {:a 1 :b 2})) (repeat 1000 {:a 2 :b 1}))))

(def dataset2 (ds/->dataset [{:a 10 :b 3} ]))

(time
 (ds/filter-column 
  dataset
  :a
  10))

(ds/unroll-column
 (ds/unroll-column
  dataset
  :b)
 :b)



[:id :col :val]
(=
 (from-index :id :id)
 (from-index :col :val))

[:id :col _]
(from-index :id :id)



(col/index-structure (ds/column (ds/concat-inplace dataset dataset2) :a)
                     )

(time
 (index/select-from-index
  (col/index-structure (ds/column dataset :a))
  :pick [1]))

(time
 (do
   (ds/concat dataset dataset)
   nil))

(time
 
 (do
   (ds/row-count
    (join/left-join [:a :b] dataset dataset))
   nil))
