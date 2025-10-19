;; Example 11: Modules and Require
;;
;; This example demonstrates how to use the module system with `require`.
;; The `require` form imports definitions from other namespaces, making them
;; accessible through qualified names.
;;
;; Prerequisites:
;; - This example relies on the math.utils module (located at math/utils.lisp)
;; - The compiler resolves module paths relative to the working directory

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(ns example.modules)

;; Import the math.utils namespace with alias 'mu'
;; This makes all public definitions from math.utils available as mu/name
(require [math.utils :as mu])

;; BASIC USAGE: Access imported values
;; The math.utils module defines a constant 'value' (42)
(def imported-value (: Int) mu/value)

;; USING IMPORTED FUNCTIONS
;; Call single-parameter functions from the imported namespace
(def incremented (: Int) (mu/add-one 10))  ; => 11

;; Call multi-parameter functions
(def sum (: Int) (mu/add 5 7))  ; => 12

;; Compose multiple imported functions
(def composed (: Int)
  (mu/add-one (mu/add imported-value incremented)))  ; => (42 + 11) + 1 = 54

;; USING IMPORTED FUNCTIONS WITH TYPES
;; Imported modules can export functions that work with custom types

;; Use imported function that returns a value
(def red-value (: Int) (mu/get-red-value))  ; => 255

;; Access the magic number constant
(def magic (: Int) mu/magic-number)  ; => 7

;; CHAINED REQUIRES
;; Modules can require other modules. If module A requires module B,
;; and you require module A, you can use A's exports (which may internally
;; use B), but you cannot directly access B's exports.
;;
;; See test case: tests/integration/65_require_chained.lisp

;; QUALIFIED NAME SYNTAX
;; All imported definitions use the alias prefix: alias/name
;; - mu/value         - imported constant
;; - mu/add-one       - imported function
;; - mu/Point         - imported type
;; - mu/make-point    - imported constructor function

;; ERROR CASES
;; The compiler will catch these errors:
;;
;; 1. Undefined namespace:
;;    (require [nonexistent.module :as nm])
;;    => ERROR: Module not found
;;
;; 2. Undefined qualified name:
;;    (require [math.utils :as mu])
;;    (def x (: Int) mu/nonexistent)
;;    => ERROR: Undefined qualified name 'mu/nonexistent'
;;
;; 3. Missing require:
;;    (def x (: Int) mu/value)
;;    => ERROR: Unbound variable 'mu/value' (if mu not required)

;; Main function to demonstrate all features
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "=== Module System Demo ===\n"))
    (printf (c-str "Imported value: %lld\n") imported-value)
    (printf (c-str "Function result: %lld\n") incremented)
    (printf (c-str "Sum: %lld\n") sum)
    (printf (c-str "Composed: %lld\n") composed)
    (printf (c-str "Red value: %lld\n") red-value)
    (printf (c-str "Magic number: %lld\n") magic)
    0))

;; Call main-fn at top level
(main-fn)

;; NOTES:
;; - Module paths use dot notation: math.utils -> math/utils.lisp
;; - Aliases are required: [namespace :as alias] is the only supported syntax
;; - All imports are qualified; no unqualified imports are supported
;; - Circular requires are not allowed
;; - Requires must appear after (ns ...) but before any code that uses them
