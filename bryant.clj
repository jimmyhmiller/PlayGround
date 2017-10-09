;; This buffer is for text that is not saved, and for Lisp evaluation.
;; To create a file, visit it with C-x C-f and enter text in its buffer.



(defn find-divisor' [n]
  (if (= n 1)
    1
    (inc (- (dec (* n n)) (find-divisor' (dec n))))))


(def find-divisor (memoize find-divisor'))

(defn f' [n]
  (if (= n 1)
    2017
    (+ (f' (dec n)) (/ 2017 (find-divisor n)))))

(def f (memoize f'))


(defn are-equal [n]
  (= (f n)
   (* (* n n) (/ 2017 (find-divisor n)))))

(are-equal 3)

(f 2017)

(every? are-equal (range 2 2018))




(comment
  2^2f(2) = f(1) + f(2)
  4f(2) = f(1) + f(2)
  3f(2) = f(1)
  3f(2) = 2017

  f(1) = 2^2f(2) - f(2)


  3^2f(3) = f(1) + f(2) + f(3)
  9f(3) = f(1) + f(2) + f(3)
  8f(3) = f(1) + f(2)
  8f(3) = 3f(2) + f(2)
  8f(3) = 4f(2)

  4^2f(4) = f(1) + f(2) + f(3) + f(4)
  16f(4) = f(1) + f(2) + f(3) + f(4)
  15f(4) = 4f(2) + f(3)
  15f(4) = 8f(3) + f(3)
  15f(4) = 9f(3)


  (/ 2017 3)
  2017/3
  (/ (* 4 (/ 2017 3)) 8)
  2017/6
  (/ (* 9 (/ (* 4 (/ 2017 3)) 8)) 15)
  2017/10
  (/ (* 9 (/ (* 4 (/ 2017 3)) 8)) 15)
  2017/10
  (+ 2017 (/ 2017 3))
  8068/3
  (* (* 2 2) (/ 2017 3))
  8068/3
  (/ 2017 10)
  2017/10
  (+ 2017 (/ 2017 3) (/ 2017 6))
  6051/2
  (+ 2017 (/ 2017 3) (/ 2017 6))
  6051/2
  (* (* 3 3) (/ 2017 6))
  6051/2)


