(ns sieve)


(defn sieve [n]
  (let [sieve (boolean-array (inc n) true)]
    (doseq [p (range 3 (int (Math/ceil (Math/sqrt (inc n)))) 2)]
      (when (aget sieve p)
        (doseq [i (range (* p p) n (* p 2))]
          (aset sieve i false))))
    (cons 2 (filter #(aget sieve %)(range 3 (inc n) 2)))))
