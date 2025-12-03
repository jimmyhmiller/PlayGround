;; Basic Clojure Compatibility Tests
;; Format: ;; TEST: description followed by expression

;; TEST: nil literal
nil

;; TEST: true literal
true

;; TEST: false literal
false

;; TEST: integer zero
0

;; TEST: positive integer
42

;; TEST: nil not equal to zero
(= nil 0)

;; TEST: nil not equal to false
(= nil false)

;; TEST: false not equal to zero
(= false 0)

;; TEST: true not equal to false
(= true false)

;; TEST: integers equal
(= 5 5)

;; TEST: integers not equal
(= 5 3)

;; TEST: less than comparison
(< 1 2)

;; TEST: greater than comparison
(> 2 1)

;; TEST: simple addition
(+ 1 2)

;; TEST: simple multiplication
(* 2 3)

;; TEST: empty let body
(let [x 2])

;; TEST: let with body
(let [x 5] x)

;; TEST: let with arithmetic
(let [x 2 y 3] (+ x y))
