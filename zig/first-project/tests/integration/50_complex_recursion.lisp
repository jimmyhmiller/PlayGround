(def sum_array_recursive (: (-> [(Pointer Int) Int Int] Int))
  (fn [arr idx len]
    (if (= idx len)
      0
      (+ (pointer-index-read arr idx)
         (sum_array_recursive arr (+ idx 1) len)))))

(def arr (: (Pointer Int)) (allocate-array Int 5 0))
(pointer-index-write! arr 0 1)
(pointer-index-write! arr 1 2)
(pointer-index-write! arr 2 3)
(pointer-index-write! arr 3 4)
(pointer-index-write! arr 4 5)

(def result (: Int) (sum_array_recursive arr 0 5))
(deallocate-array arr)
(printf (c-str "%lld\n") result)
