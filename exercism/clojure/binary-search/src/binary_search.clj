(ns binary-search)

(defn middle 
  ([coll]
   (int (/ (count coll) 2)))
  ([lower upper]
   (int (Math/ceil (/ (+ upper lower) 2)))))

(defn search-for 
  ([val coll] (search-for val coll 0 (dec (count coll))))
  ([val coll lower upper]
   (let [index (middle lower upper)
         elem (nth coll index)]
     (cond
       (= val elem) index
       (= lower upper) (throw (Exception. "not found"))
       (> val elem) (search-for val coll index upper)
       (< val elem) (search-for val coll lower index)))))



