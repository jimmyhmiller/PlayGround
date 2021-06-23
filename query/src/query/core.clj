(ns query.core)

;; https://scattered-thoughts.net/writing/a-practical-relational-query-compiler-in-500-lines/


(defn gallop [coll val]
  (if (and (not (empty? coll))
           (< (first coll) val))
    (let [step 1
          result
          (loop [coll coll val val step step]
            (if (and (< step (count coll))
                     (< (nth coll step) val))
              (recur (subvec coll step)
                     val
                     (bit-shift-left step 1))
              ;; Would love to get rid of this allocation
              ;; I could consider deftype with a mutable variable for this stuff
              [coll step]))
          coll (nth result 0)
          step (nth result 1)
          step (bit-shift-right step 1)
          coll (loop [coll coll
                      val val
                      step step]
                 (if (> step 0)
                   (if (and (< step (count coll))
                            (< (nth coll step) val))
                     (recur (subvec coll step)
                            val
                            (bit-shift-right step 1))
                     (recur coll val (bit-shift-right step 1)))
                   coll))]
      (subvec coll 1))
    coll))




