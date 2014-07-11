(ns three-d.core
  (:use scad-clj.model scad-clj.scad))


(defn column [radius height pos]
  (->>
    (cylinder radius height)
    (translate pos)))



(def primitives
  (union
   (map #(column 10 160 [% 0 0]) (range 0 500 50))
   (map #(column 10 160 [0 % 0]) (range 0 500 50))
   (map #(column 10 160 [450 % 0]) (range 0 500 50))
   (map #(column 10 160 [% 450 0]) (range 0 500 50))
   (cube 450 450 10)))

(spit "post-demo.scad"
      (write-scad primitives))

