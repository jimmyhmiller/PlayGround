(defmodule MyApp
  "This is a module docstring
  that spans multiple lines"
  (defn calculate [x y]
    (let [sum (+ x y)
          product (* x y
      (if (> sum 10)
        {:result sum
         :status "large"
        {:result product
         :status "small"
