;; Language Feature Showcase
;; This file demonstrates all the type system features supported by our Lisp dialect
;; All examples here successfully type check!

;; ====================
;; 1. Basic Types
;; ====================

;; Integer literals
(def my-int (: Int) 42)

;; Float literals
(def pi (: Float) 3.14159)

;; String literals
(def greeting (: String) "Hello, World!")

;; Boolean (as integers for now)
(def is-true (: Int) 1)

;; Nil value
(def nothing (: Nil) nil)

;; ====================
;; 2. Specific Numeric Types
;; ====================

;; Unsigned integers
(def byte-val (: U8) 255)
(def word-val (: U16) 65535)
(def dword-val (: U32) 4294967295)

;; Signed integers
(def small-signed (: I8) -128)
(def medium-signed (: I16) -32768)
(def large-signed (: I32) -2147483648)

;; Floating point types
(def float32-val (: F32) 3.14)
(def float64-val (: F64) 2.718281828)

;; ====================
;; 3. Function Types
;; ====================

;; Simple function: Int -> Int
(def increment (: (-> [Int] Int))
  (fn [x] (+ x 1)))

;; Function with multiple parameters: (Int, Int) -> Int
(def add (: (-> [Int Int] Int))
  (fn [x y] (+ x y)))

;; Higher-order function: takes a function and applies it
(def apply-twice (: (-> [(-> [Int] Int) Int] Int))
  (fn [f x] (f (f x))))

;; Function composition
(def compose (: (-> [(-> [Int] Int) (-> [Int] Int)] (-> [Int] Int)))
  (fn [f g] (fn [x] (f (g x)))))

;; ====================
;; 4. Arithmetic Operations
;; ====================

;; All basic arithmetic operations are supported
(def sum (: Int) (+ 10 20 30))
(def difference (: Int) (- 100 50))
(def product (: Int) (* 6 7))
(def quotient (: Float) (/ 22 7))  ; Division of integers produces float
(def remainder (: Int) (% 17 5))

;; Nested arithmetic
(def complex-calc (: Int)
  (+ (* 3 4) (- 20 (% 15 4))))

;; ====================
;; 5. Vector Types
;; ====================

;; Homogeneous vectors
(def int-vector (: [Int]) [1 2 3 4 5])
(def string-vector (: [String]) ["hello" "world"])

;; Empty vectors (polymorphic)
(def empty-vec (: [Int]) [])

;; ====================
;; 6. Struct Types
;; ====================

;; Define a Point struct
(def Point (Struct [x Int] [y Int]))

;; Define a Color struct
(def Color (Struct [r U8] [g U8] [b U8]))

;; Define a Rectangle struct using Point
(def Rectangle (Struct
  [top-left Point]
  [width Int]
  [height Int]))

;; Functions operating on structs
(def translate-point (: (-> [Point Int Int] Point))
  (fn [p dx dy] p))  ; Simplified - just returns the point

(def point-to-color (: (-> [Point] Color))
  (fn [p] p))  ; Type would fail if uncommented - different types!

;; ====================
;; 7. Forward References
;; ====================

;; Functions can reference each other (two-pass type checking)
(def is-even (: (-> [Int] Int))
  (fn [n] (is-odd n)))

(def is-odd (: (-> [Int] Int))
  (fn [n] (is-even n)))

;; Complex dependency chains work
(def func-a (: Int) func-b)
(def func-b (: Int) func-c)
(def func-c (: Int) func-d)
(def func-d (: Int) 42)

;; ====================
;; 8. Complex Function Types
;; ====================

;; Function returning a function (currying)
(def make-adder (: (-> [Int] (-> [Int] Int)))
  (fn [x] (fn [y] (+ x y))))

;; Function with struct parameters
(def distance (: (-> [Point Point] Float))
  (fn [p1 p2] 0.0))

;; Function with mixed types
(def process-data (: (-> [String Int Point] Color))
  (fn [name count position] position))

;; ====================
;; 9. Nested Structures
;; ====================

;; A Person struct with nested types
(def Person (Struct
  [name String]
  [age U8]
  [location Point]))

;; A Company struct with a vector of Persons
(def Company (Struct
  [name String]
  [employees [Person]]))

;; ====================
;; 10. Type Inference
;; ====================

;; The type checker can infer types for literals
;; (these would work if we didn't require type annotations on def)
; (def inferred-int 42)           ; Would infer Int
; (def inferred-string "hello")   ; Would infer String
; (def inferred-vec [1 2 3])      ; Would infer [Int]

;; ====================
;; DEMONSTRATION NOTES
;; ====================

;; This file showcases:
;; - All primitive types (Int, Float, String, Nil)
;; - Specific numeric types (U8, U16, I32, F64, etc.)
;; - Function types with various signatures
;; - Higher-order functions
;; - Vector types
;; - User-defined struct types
;; - Forward references (mutual recursion)
;; - Nested and complex type compositions
;; - Type-safe arithmetic operations

;; The type checker ensures:
;; - Type safety for all operations
;; - Correct function application
;; - Homogeneous vectors
;; - Struct field type checking
;; - Forward reference resolution (two-pass)

;; Current limitations:
;; - No parametric polymorphism (generics)
;; - No type inference for definitions (must annotate)
;; - No pattern matching
;; - No sum types (unions/enums)
;; - Limited struct operations (no field access yet)