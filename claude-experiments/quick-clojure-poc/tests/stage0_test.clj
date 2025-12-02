;; Stage 0 Test Cases
;; These should all be readable and printable (no evaluation yet)

;; Scalars
nil
true
false
42
3.14
"hello world"

;; Symbols and Keywords
foo
bar
:keyword
:namespaced/keyword

;; Lists
(+ 1 2)
(defn foo [x] x)

;; Vectors
[1 2 3]
[true false nil]
[x y z]

;; Maps
{:a 1 :b 2}
{:name "Alice" :age 30}
{"key" "value"}

;; Sets
#{1 2 3}
#{:a :b :c}

;; Nested structures
[1 2 {:a 3}]
{:vec [1 2 3] :map {:nested true}}
(def x [1 2 {:key "value"}])
