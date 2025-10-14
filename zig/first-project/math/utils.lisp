(ns math.utils)

;; Value definition
(def value (: Int) 42)

;; Function definition - single parameter
(def add-one (: (-> [Int] Int))
  (fn [x] (+ x 1)))

;; Multi-parameter function
(def add (: (-> [Int Int] Int))
  (fn [a b] (+ a b)))

;; Struct definition
(def Point (: Type) (Struct [x Int] [y Int]))

;; Function that returns a struct
(def make-point (: (-> [Int Int] Point))
  (fn [x y] (Point x y)))

;; Enum definition
(def Color (: Type) (Enum Red Green Blue))

;; Function that returns first variant value
(def get-red-value (: (-> [] Int))
  (fn [] 255))

;; Another value for testing multiple definitions
(def magic-number (: Int) 7)
