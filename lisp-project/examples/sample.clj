;; Sample Clojure-style code that the reader can parse
;; This demonstrates the various forms supported by the reader

;; Simple number
42

;; Simple symbol
hello

;; String
"Hello, World!"

;; Simple list
(+ 1 2)

;; List with symbols
(def x 10)

;; Vector
[1 2 3 4 5]

;; Vector with mixed types
[:keyword "string" 42 symbol]

;; Map
{:name "John" :age 30}

;; Nested structures
(def user
  {:name "Alice"
   :email "alice@example.com"
   :roles [:admin :developer]
   :metadata {:created-at 1234567890
              :updated-at 1234567900}})

;; Function definition style
(defn add [a b]
  (+ a b))

;; More complex nested structure
(def data
  {:users [{:id 1 :name "Alice"}
           {:id 2 :name "Bob"}
           {:id 3 :name "Charlie"}]
   :settings {:theme "dark"
              :language "en"
              :notifications {:email true :sms false}}})

;; Arithmetic expressions
(* (+ 1 2) (- 5 3))

;; Nested lists
(let [x 10
      y 20]
  (+ x y))

;; Comments are supported
(def z 100) ; This is a comment
