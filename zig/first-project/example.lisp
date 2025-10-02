;; Example Lisp file for testing Emacs integration
;; Use C-x C-e after each expression to evaluate it in the REPL

(ns example.core)

;; Simple arithmetic
(+ 1 2)

;; Define a function
(def square
  (: (-> [Int] Int))
  (fn [x]
    (* x x)))

;; Use the function
(square 7)

;; Define another function that uses the first
(def cube
  (: (-> [Int] Int))
  (fn [x]
    (* x (square x))))

(cube 3)

;; Recursive function
(def fib
  (: (-> [Int] Int))
  (fn [n]
    (if (< n 2)
        n
        (+ (fib (- n 1))
           (fib (- n 2))))))

(fib 10)

;; Multiple definitions and expressions
(def answer (: Int) 42)
(def x (: Int) 100)
(+ answer x)


(def Point (: Type)
  (Struct [x Int] [y Int]))

(def p (Point 2 3))

(def y (: Int) (. p y))

