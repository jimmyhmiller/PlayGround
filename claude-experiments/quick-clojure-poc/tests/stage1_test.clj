;; Stage 1 Test Cases - Evaluation Tests
;; These should all evaluate correctly

;; Basic literals
42      ; => 42
true    ; => true
false   ; => false
nil     ; => nil

;; def
(def x 42)        ; => 42
x                 ; => 42

;; Arithmetic
(+ 1 2)           ; => 3
(- 10 4)          ; => 6
(* 2 3)           ; => 6
(/ 20 5)          ; => 4
(+ 1.5 2.5)       ; => 4.0

;; Comparisons
(< 1 2)           ; => true
(> 5 3)           ; => true
(= 2 2)           ; => true
(= 1 2)           ; => false

;; if
(if true 1 2)     ; => 1
(if false 1 2)    ; => 2
(if nil 1 2)      ; => 2

;; Nested expressions
(+ (* 2 3) 4)                    ; => 10
(if (< 5 10) (+ 1 2) (- 10 5))  ; => 3

;; Multiple defs and references
(def y 100)
(def z (+ y 50))
z                                ; => 150

;; Complex example
(def n 42)
(if (< n 100)
  (+ n 1)
  n)                             ; => 43
