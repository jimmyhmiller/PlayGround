;; Test inline multiple closure calls
;; This was failing before the fix because the second closure call
;; would clobber the first call's result stored in a volatile register

(let [add1 (fn [x] (+ x 1))
      mul2 (fn [x] (* x 2))]
  (+ (add1 5) (mul2 3)))

;; Expected result:
;; add1(5) = 6
;; mul2(3) = 6
;; 6 + 6 = 12
;; Tagged: 12 << 3 = 96
