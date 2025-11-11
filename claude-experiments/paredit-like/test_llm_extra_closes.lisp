(defn process-data [items]
  (map (fn [x]
         (* x 2))
       items)))))))
