; Example of unbalanced Clojure code that needs fixing

(defn fibonacci [n]
  (if (<= n 1)
    n
    (+ (fibonacci (- n 1))
       (fibonacci (- n 2

(defn factorial [n]
  (if (= n 0
    1
    (* n (factorial (- n 1)))

(let [x 1
      y 2
      z 3
  (+ x y z

; Missing closing paren on map
(def my-map {:a 1
             :b 2
             :c 3

; Extra closing parens
(defn foo [x])
  (+ x 1)))
